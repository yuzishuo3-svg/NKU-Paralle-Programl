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

struct PTTaskPayload {
    string prefix;
    vector<string> suffixes;
};

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

static segment* get_segment_ptr(PriorityQueue& q, const segment& seg) {
    if (seg.type == 1) {
        return &q.m.letters[q.m.FindLetter(seg)];
    }
    if (seg.type == 2) {
        return &q.m.digits[q.m.FindDigit(seg)];
    }
    if (seg.type == 3) {
        return &q.m.symbols[q.m.FindSymbol(seg)];
    }
    return nullptr;
}

static PTTaskPayload build_task_from_pt(PriorityQueue& q, const PT& pt) {
    PTTaskPayload task;

    int seg_count = (int)pt.content.size();

    if (seg_count == 1) {
        segment* last_seg = get_segment_ptr(q, pt.content[0]);
        if (last_seg != nullptr) {
            task.prefix = "";
            task.suffixes = last_seg->ordered_values;
        }
        return task;
    }

    string prefix;
    int seg_idx = 0;

    for (int idx : pt.curr_indices) {
        if (seg_idx == seg_count - 1) {
            break;
        }

        segment* cur_seg = get_segment_ptr(q, pt.content[seg_idx]);
        if (cur_seg != nullptr) {
            prefix += cur_seg->ordered_values[idx];
        }

        seg_idx++;
    }

    segment* last_seg = get_segment_ptr(q, pt.content[seg_count - 1]);
    if (last_seg != nullptr) {
        task.prefix = prefix;
        task.suffixes = last_seg->ordered_values;
    }

    return task;
}

static void insert_pt_by_prob(PriorityQueue& q, PT pt) {
    q.CalProb(pt);

    if (q.priority.empty()) {
        q.priority.emplace_back(pt);
        return;
    }

    auto iter = q.priority.begin();
    while (iter != q.priority.end() && iter->prob >= pt.prob) {
        ++iter;
    }

    q.priority.emplace(iter, pt);
}

static void pack_strings(const vector<string>& values,
                         vector<int>& lens,
                         vector<char>& packed) {
    int total = (int)values.size();
    lens.resize(total);

    int packed_size = 0;
    for (int i = 0; i < total; i++) {
        lens[i] = (int)values[i].size();
        packed_size += lens[i];
    }

    packed.resize(packed_size);

    int pos = 0;
    for (int i = 0; i < total; i++) {
        memcpy(packed.data() + pos, values[i].data(), values[i].size());
        pos += (int)values[i].size();
    }
}

static void send_empty_task(int worker) {
    int cmd = 1;
    int total = 0;

    MPI_Send(&cmd, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
    MPI_Send(&total, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
}

static void send_pt_task(int worker, const PTTaskPayload& task) {
    int cmd = 1;
    int total = (int)task.suffixes.size();

    MPI_Send(&cmd, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
    MPI_Send(&total, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);

    if (total == 0) {
        return;
    }

    int prefix_len = (int)task.prefix.size();

    vector<int> lens;
    vector<char> packed;
    pack_strings(task.suffixes, lens, packed);

    int packed_size = (int)packed.size();

    MPI_Send(&prefix_len, 1, MPI_INT, worker, 2, MPI_COMM_WORLD);

    if (prefix_len > 0) {
        MPI_Send(task.prefix.data(), prefix_len, MPI_CHAR, worker, 3, MPI_COMM_WORLD);
    }

    MPI_Send(lens.data(), total, MPI_INT, worker, 4, MPI_COMM_WORLD);
    MPI_Send(&packed_size, 1, MPI_INT, worker, 5, MPI_COMM_WORLD);

    if (packed_size > 0) {
        MPI_Send(packed.data(), packed_size, MPI_CHAR, worker, 6, MPI_COMM_WORLD);
    }
}

static void stop_worker(int worker) {
    int cmd = 0;
    MPI_Send(&cmd, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
}

static double hash_task_simd(const string& prefix,
                             const vector<int>& lens,
                             const vector<char>& packed) {
    double start = MPI_Wtime();

    int pos = 0;
    string pw_batch[4];
    bit32 state_batch[4][4];
    int batch_count = 0;

    for (int i = 0; i < (int)lens.size(); i++) {
        string suffix(packed.data() + pos, lens[i]);
        pos += lens[i];

        string guess = prefix + suffix;

        pw_batch[batch_count] = guess;
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

static void worker_loop_ptbatch(int rank) {
    double total_hash_time = 0.0;
    long long total_hashed = 0;
    int task_count = 0;

    while (true) {
        int cmd = 0;
        MPI_Recv(&cmd, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (cmd == 0) {
            break;
        }

        int total = 0;
        MPI_Recv(&total, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (total == 0) {
            double hash_time = 0.0;
            long long hashed = 0;
            MPI_Send(&hash_time, 1, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
            MPI_Send(&hashed, 1, MPI_LONG_LONG, 0, 11, MPI_COMM_WORLD);
            continue;
        }

        int prefix_len = 0;
        MPI_Recv(&prefix_len, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        string prefix;
        if (prefix_len > 0) {
            prefix.resize(prefix_len);
            MPI_Recv(&prefix[0], prefix_len, MPI_CHAR, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        vector<int> lens(total);
        MPI_Recv(lens.data(), total, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int packed_size = 0;
        MPI_Recv(&packed_size, 1, MPI_INT, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        vector<char> packed(packed_size);
        if (packed_size > 0) {
            MPI_Recv(packed.data(), packed_size, MPI_CHAR, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double hash_time = hash_task_simd(prefix, lens, packed);
        long long hashed = total;

        total_hash_time += hash_time;
        total_hashed += hashed;
        task_count++;

        MPI_Send(&hash_time, 1, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
        MPI_Send(&hashed, 1, MPI_LONG_LONG, 0, 11, MPI_COMM_WORLD);
    }

    MPI_Send(&total_hash_time, 1, MPI_DOUBLE, 0, 20, MPI_COMM_WORLD);
    MPI_Send(&total_hashed, 1, MPI_LONG_LONG, 0, 21, MPI_COMM_WORLD);
    MPI_Send(&task_count, 1, MPI_INT, 0, 22, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0) {
        cout << "MPI password guessing experiment" << endl;
        cout << "Advanced mode = PT batch generation and hash" << endl;
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
            cout << "PT batch mode needs at least 2 MPI processes." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    if (rank != 0) {
        worker_loop_ptbatch(rank);
        MPI_Finalize();
        return 0;
    }

    double time_train = 0.0;
    double time_total = 0.0;

    PriorityQueue q;

    double start_train = MPI_Wtime();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    double end_train = MPI_Wtime();
    time_train = end_train - start_train;

    q.init();

    cout << "here" << endl;

    const int GENERATE_LIMIT = 10000000;
    const int PRINT_STEP = 100000;

    int worker_count = world_size - 1;
    long long history = 0;
    long long last_print = 0;

    long long total_pt_tasks = 0;
    long long total_rounds = 0;

    vector<double> worker_hash_sum(world_size, 0.0);
    vector<long long> worker_hashed_sum(world_size, 0);
    vector<int> worker_task_sum(world_size, 0);

    double total_start = MPI_Wtime();

    while (!q.priority.empty() && history < GENERATE_LIMIT) {
        vector<PT> batch_pts;
        vector<vector<PT>> batch_new_pts;
        vector<PTTaskPayload> tasks;

        int take_num = worker_count;
        if ((int)q.priority.size() < take_num) {
            take_num = (int)q.priority.size();
        }

        for (int i = 0; i < take_num; i++) {
            PT pt = q.priority.front();
            q.priority.erase(q.priority.begin());

            vector<PT> new_pts = pt.NewPTs();

            batch_pts.emplace_back(pt);
            batch_new_pts.emplace_back(new_pts);
            tasks.emplace_back(build_task_from_pt(q, pt));
        }

        for (int worker = 1; worker <= worker_count; worker++) {
            int idx = worker - 1;
            if (idx < (int)tasks.size()) {
                send_pt_task(worker, tasks[idx]);
            } else {
                send_empty_task(worker);
            }
        }

        long long round_hashed = 0;
        double round_max_hash_time = 0.0;

        for (int worker = 1; worker <= worker_count; worker++) {
            double hash_time = 0.0;
            long long hashed = 0;

            MPI_Recv(&hash_time, 1, MPI_DOUBLE, worker, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&hashed, 1, MPI_LONG_LONG, worker, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            worker_hash_sum[worker] += hash_time;
            worker_hashed_sum[worker] += hashed;

            if (hashed > 0) {
                worker_task_sum[worker] += 1;
            }

            round_hashed += hashed;
            if (hash_time > round_max_hash_time) {
                round_max_hash_time = hash_time;
            }
        }

        history += round_hashed;
        total_pt_tasks += (int)tasks.size();
        total_rounds++;

        for (auto& new_pts : batch_new_pts) {
            for (PT new_pt : new_pts) {
                insert_pt_by_prob(q, new_pt);
            }
        }

        if (history - last_print >= PRINT_STEP) {
            cout << "Guesses generated: " << history << endl;
            last_print = history;
        }
    }

    double total_end = MPI_Wtime();
    time_total = total_end - total_start;

    for (int worker = 1; worker <= worker_count; worker++) {
        stop_worker(worker);
    }

    for (int worker = 1; worker <= worker_count; worker++) {
        double final_hash_time = 0.0;
        long long final_hashed = 0;
        int final_tasks = 0;

        MPI_Recv(&final_hash_time, 1, MPI_DOUBLE, worker, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&final_hashed, 1, MPI_LONG_LONG, worker, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&final_tasks, 1, MPI_INT, worker, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        cout << "Worker " << worker
             << " tasks=" << final_tasks
             << ", hashed=" << final_hashed
             << ", hash_time=" << final_hash_time
             << " seconds" << endl;
    }

    cout << "Total PT tasks:" << total_pt_tasks << endl;
    cout << "Total rounds:" << total_rounds << endl;
    cout << "Total hashed:" << history << endl;

    cout << "Guess time:" << time_total << "seconds" << endl;
    cout << "Hash time:" << time_total << "seconds" << endl;
    cout << "Train time:" << time_train << "seconds" << endl;
    cout << "PT batch total time:" << time_total << "seconds" << endl;

    MPI_Finalize();
    return 0;
}