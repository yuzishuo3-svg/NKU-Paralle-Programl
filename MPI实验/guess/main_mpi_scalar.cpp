#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
#include <vector>
#include <string>
#include <cstring>

using namespace std;
using namespace chrono;

// MPI + Serial MD5 编译指令：
// mpic++ -O2 -std=c++11 main_mpi_scalar.cpp train.cpp guessing.cpp md5.cpp -o main

static int md5_self_test(int rank) {
    if (rank != 0) return 0;

    cout << "Testing MD5Hash correctness..." << endl;

    string test_pws[8] = {
        "123456", "password", "12345678", "qwerty",
        "123456789", "12345", "1234", "111111"
    };

    string test_hashes[8] = {
        "e10adc3949ba59abbe56e057f20f883e",
        "5f4dcc3b5aa765d61d8327deb882cf99",
        "25d55ad283aa400af464c76d713c07ad",
        "d8578edf8458ce06fbc5bb76a58c5ca4",
        "25f9e794323b453885f5181f1b624d0b",
        "827ccb0eea8a706c4c34a16891f84e7b",
        "81dc9bdb52d04dc20036dbd8313ed055",
        "96e79218965eb72c92a549dd5a330112"
    };

    for (int i = 0; i < 8; i++) {
        bit32 state[4];
        MD5Hash(test_pws[i], state);

        stringstream ss;
        for (int i1 = 0; i1 < 4; i1++) {
            ss << setw(8) << setfill('0') << hex << state[i1];
        }

        if (ss.str() != test_hashes[i]) {
            cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
            cout << "Expected: " << test_hashes[i] << "\nGot:      " << ss.str() << endl;
            return 1;
        }
    }

    cout << "MD5Hash test passed!" << endl; // 请不要修改这一行
    return 0;
}

// MPI + Serial MD5：所有 MPI 进程共同处理一批 guesses。
// rank 0 拥有完整 guesses，其它 rank 通过 MPI_Bcast 接收。
// 每个 rank 只处理自己负责的区间。
// 与 SIMD 版本不同：这里每个口令都用普通 MD5Hash 逐个计算。
static double mpi_hash_one_batch(const vector<string>* root_guesses, int rank, int world_size) {
    int total = 0;

    if (rank == 0) {
        total = (int)root_guesses->size();
    }

    MPI_Bcast(&total, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> lens(total);
    int packed_size = 0;

    if (rank == 0) {
        for (int i = 0; i < total; i++) {
            lens[i] = (int)(*root_guesses)[i].size();
            packed_size += lens[i];
        }
    }

    MPI_Bcast(lens.data(), total, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&packed_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> offsets(total);
    for (int i = 1; i < total; i++) {
        offsets[i] = offsets[i - 1] + lens[i - 1];
    }

    vector<char> packed(packed_size);

    if (rank == 0) {
        int pos = 0;
        for (int i = 0; i < total; i++) {
            const string& s = (*root_guesses)[i];
            memcpy(packed.data() + pos, s.data(), s.size());
            pos += (int)s.size();
        }
    }

    MPI_Bcast(packed.data(), packed_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    int start = total * rank / world_size;
    int end = total * (rank + 1) / world_size;

    double local_start = MPI_Wtime();

    // 这里是 Serial MD5 版本：不用 MD5Hash_SIMD，逐个计算
    for (int i = start; i < end; i++) {
        string pw(packed.data() + offsets[i], lens[i]);
        bit32 state[4];
        MD5Hash(pw, state);
    }

    double local_end = MPI_Wtime();
    double local_time = local_end - local_start;
    double max_time = 0.0;

    // 一批任务真正完成的时间取所有进程中最慢的那个
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        return max_time;
    }

    return 0.0;
}

// 非 0 号进程一直等待 rank 0 分发任务。
// cmd = 1 表示接收一批口令并 hash。
// cmd = 0 表示实验结束。
static void worker_loop(int rank, int world_size) {
    while (true) {
        int cmd = 0;
        MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (cmd == 0) {
            break;
        }

        if (cmd == 1) {
            mpi_hash_one_batch(nullptr, rank, world_size);
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0) {
        cout << "MPI password guessing experiment" << endl;
        cout << "MPI world_size = " << world_size << endl;
        cout << "Hash mode = Serial MD5" << endl;
    }

    int test_result = md5_self_test(rank);
    MPI_Bcast(&test_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (test_result != 0) {
        MPI_Finalize();
        return 1;
    }

    // 非 0 号进程不训练、不维护优先队列，只等待 rank 0 分发 hash 任务
    if (rank != 0) {
        worker_loop(rank, world_size);
        MPI_Finalize();
        return 0;
    }

    double time_hash = 0.0;   // 用于 MD5 哈希的并行墙钟时间
    double time_guess = 0.0;  // 猜测生成总时间，不含 hash
    double time_train = 0.0;  // 模型训练时间

    PriorityQueue q;

    double start_train = MPI_Wtime();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    double end_train = MPI_Wtime();
    time_train = end_train - start_train;

    q.init();

    cout << "here" << endl;

    const int GENERATE_LIMIT = 10000000;
    const int HASH_BATCH_SIZE = 1000000;
    const int PRINT_STEP = 100000;

    int curr_num = 0;
    int history = 0;

    double start = MPI_Wtime();

    while (!q.priority.empty()) {
        q.PopNext();

        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= PRINT_STEP) {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;
        }

        // 达到一批后，rank 0 分发给所有 MPI 进程共同 hash
        if ((int)q.guesses.size() >= HASH_BATCH_SIZE ||
            history + (int)q.guesses.size() >= GENERATE_LIMIT) {

            int cmd = 1;
            MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);

            time_hash += mpi_hash_one_batch(&q.guesses, rank, world_size);

            history += q.guesses.size();
            curr_num = 0;
            q.guesses.clear();

            if (history >= GENERATE_LIMIT) {
                break;
            }
        }
    }

    double end = MPI_Wtime();
    double total_guess_and_hash = end - start;
    time_guess = total_guess_and_hash - time_hash;

    // 通知其它 MPI 进程结束
    int cmd = 0;
    MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);

    cout << "Guess time:" << time_guess << "seconds" << endl; // 请不要修改这一行
    cout << "Hash time:" << time_hash << "seconds" << endl;   // 请不要修改这一行
    cout << "Train time:" << time_train << "seconds" << endl; // 请不要修改这一行

    MPI_Finalize();
    return 0;
}