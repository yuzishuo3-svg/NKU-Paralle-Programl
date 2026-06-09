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
        for (int j = 0; j < 4; j++) {
            ss << setw(8) << setfill('0') << hex << state[j];
        }

        if (ss.str() != test_hashes[i]) {
            cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
            cout << "Expected: " << test_hashes[i] << "\nGot:      " << ss.str() << endl;
            return 1;
        }
    }

    cout << "MD5Hash test passed!" << endl;
    return 0;
}

static void pack_guesses(const vector<string>& guesses,
                         vector<int>& lens,
                         vector<char>& packed) {
    int total = (int)guesses.size();
    lens.resize(total);

    int packed_size = 0;
    for (int i = 0; i < total; i++) {
        lens[i] = (int)guesses[i].size();
        packed_size += lens[i];
    }

    packed.resize(packed_size);

    int pos = 0;
    for (int i = 0; i < total; i++) {
        memcpy(packed.data() + pos, guesses[i].data(), guesses[i].size());
        pos += (int)guesses[i].size();
    }
}

static double hash_packed_batch_simd(const vector<int>& lens,
                                     const vector<char>& packed) {
    double start = MPI_Wtime();

    int pos = 0;
    string pw_batch[4];
    bit32 state_batch[4][4];
    int batch_count = 0;

    for (int i = 0; i < (int)lens.size(); i++) {
        string pw(packed.data() + pos, lens[i]);
        pos += lens[i];

        pw_batch[batch_count] = pw;
        batch_count++;

        if (batch_count == 4) {
            MD5Hash_SIMD(pw_batch, state_batch);
            batch_count = 0;
        }
    }

    for (int i = 0; i < batch_count; i++) {
        bit32 state[4];
        MD5Hash(pw_batch[i], state);
    }

    double end = MPI_Wtime();
    return end - start;
}

static void send_batch_to_worker(int worker, const vector<string>& guesses) {
    int cmd = 1;
    int total = (int)guesses.size();

    vector<int> lens;
    vector<char> packed;
    pack_guesses(guesses, lens, packed);

    int packed_size = (int)packed.size();

    MPI_Send(&cmd, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
    MPI_Send(&total, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
    MPI_Send(lens.data(), total, MPI_INT, worker, 2, MPI_COMM_WORLD);
    MPI_Send(&packed_size, 1, MPI_INT, worker, 3, MPI_COMM_WORLD);
    MPI_Send(packed.data(), packed_size, MPI_CHAR, worker, 4, MPI_COMM_WORLD);
}

static void stop_worker(int worker) {
    int cmd = 0;
    MPI_Send(&cmd, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
}

static void worker_loop_pipeline(int rank) {
    double local_hash_time = 0.0;
    long long local_hash_count = 0;
    int batch_count = 0;

    while (true) {
        int cmd = 0;
        MPI_Recv(&cmd, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (cmd == 0) {
            break;
        }

        int total = 0;
        MPI_Recv(&total, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        vector<int> lens(total);
        MPI_Recv(lens.data(), total, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int packed_size = 0;
        MPI_Recv(&packed_size, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        vector<char> packed(packed_size);
        MPI_Recv(packed.data(), packed_size, MPI_CHAR, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        local_hash_time += hash_packed_batch_simd(lens, packed);
        local_hash_count += total;
        batch_count++;
    }

    MPI_Send(&local_hash_time, 1, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
    MPI_Send(&local_hash_count, 1, MPI_LONG_LONG, 0, 11, MPI_COMM_WORLD);
    MPI_Send(&batch_count, 1, MPI_INT, 0, 12, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0) {
        cout << "MPI password guessing experiment" << endl;
        cout << "Advanced mode = generation-hash pipeline" << endl;
        cout << "MPI world_size = " << world_size << endl;
    }

    int test_result = md5_self_test(rank);
    MPI_Bcast(&test_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (test_result != 0) {
        MPI_Finalize();
        return 1;
    }

    if (world_size < 2) {
        if (rank == 0) {
            cout << "Pipeline mode needs at least 2 MPI processes." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    if (rank != 0) {
        worker_loop_pipeline(rank);
        MPI_Finalize();
        return 0;
    }

    double time_train = 0.0;
    double time_hash_max = 0.0;
    double time_pipeline_total = 0.0;
    double time_generate_dispatch = 0.0;

    PriorityQueue q;

    double start_train = MPI_Wtime();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    double end_train = MPI_Wtime();
    time_train = end_train - start_train;

    q.init();

    cout << "here" << endl;

    const int GENERATE_LIMIT = 10000000;
    const int PIPELINE_BATCH_SIZE = 500000;
    const int PRINT_STEP = 100000;

    int curr_num = 0;
    int history = 0;
    int next_worker = 1;
    int total_batches = 0;

    double pipeline_start = MPI_Wtime();

    while (!q.priority.empty()) {
        q.PopNext();
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= PRINT_STEP) {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;
        }

        if ((int)q.guesses.size() >= PIPELINE_BATCH_SIZE ||
            history + (int)q.guesses.size() >= GENERATE_LIMIT) {

            send_batch_to_worker(next_worker, q.guesses);

            history += q.guesses.size();
            q.guesses.clear();
            curr_num = 0;
            total_batches++;

            next_worker++;
            if (next_worker >= world_size) {
                next_worker = 1;
            }

            if (history >= GENERATE_LIMIT) {
                break;
            }
        }
    }

    double dispatch_end = MPI_Wtime();
    time_generate_dispatch = dispatch_end - pipeline_start;

    for (int worker = 1; worker < world_size; worker++) {
        stop_worker(worker);
    }

    long long total_hashed = 0;

    for (int worker = 1; worker < world_size; worker++) {
        double worker_hash_time = 0.0;
        long long worker_hash_count = 0;
        int worker_batch_count = 0;

        MPI_Recv(&worker_hash_time, 1, MPI_DOUBLE, worker, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&worker_hash_count, 1, MPI_LONG_LONG, worker, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&worker_batch_count, 1, MPI_INT, worker, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (worker_hash_time > time_hash_max) {
            time_hash_max = worker_hash_time;
        }

        total_hashed += worker_hash_count;

        cout << "Worker " << worker
             << " batches=" << worker_batch_count
             << ", hashed=" << worker_hash_count
             << ", hash_time=" << worker_hash_time
             << " seconds" << endl;
    }

    double pipeline_end = MPI_Wtime();
    time_pipeline_total = pipeline_end - pipeline_start;

    cout << "Total batches:" << total_batches << endl;
    cout << "Total hashed:" << total_hashed << endl;

    cout << "Guess time:" << time_generate_dispatch << "seconds" << endl;
    cout << "Hash time:" << time_hash_max << "seconds" << endl;
    cout << "Train time:" << time_train << "seconds" << endl;
    cout << "Pipeline total time:" << time_pipeline_total << "seconds" << endl;

    MPI_Finalize();
    return 0;
}