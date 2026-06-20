#include "PCFG.h"
#include "gpu_generator.cuh"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <future>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

namespace {

struct Options {
    string train_path = "Rockyou-singleLined-full.txt";
    int generate_limit = 10000000;
    int cpu_threshold = 4096;
    int target_gpu_batch_guesses = 200000;
    int max_gpu_batch_pts = 128;
    int print_step = 100000;
    bool store_guesses = false;
};

struct Stats {
    long long generated = 0;
    long long cpu_generated = 0;
    long long gpu_generated = 0;
    long long cpu_tasks = 0;
    long long gpu_tasks = 0;
    long long gpu_batches = 0;
    long long generated_bytes = 0;
    double train_seconds = 0.0;
    double total_seconds = 0.0;
    double cpu_generate_seconds = 0.0;
    double gpu_wall_seconds = 0.0;
    double gpu_kernel_seconds = 0.0;
    double gpu_wait_seconds = 0.0;
};

struct InFlightGpuBatch {
    bool active = false;
    std::future<GpuBatchResult> future;
    std::size_t expected = 0;
};

double now_seconds() {
    using clock = chrono::high_resolution_clock;
    static const auto base = clock::now();
    return chrono::duration<double>(clock::now() - base).count();
}

int to_int_arg(const char* value, const char* name) {
    char* end = nullptr;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0) {
        throw runtime_error(string("Invalid value for ") + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

Options parse_args(int argc, char** argv) {
    Options opt;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw runtime_error(string("Missing value after ") + name);
            }
            return argv[++i];
        };

        if (arg == "--train") {
            opt.train_path = need_value("--train");
        } else if (arg == "--limit") {
            opt.generate_limit = to_int_arg(need_value("--limit"), "--limit");
        } else if (arg == "--cpu-threshold") {
            opt.cpu_threshold = to_int_arg(need_value("--cpu-threshold"), "--cpu-threshold");
        } else if (arg == "--batch-guesses") {
            opt.target_gpu_batch_guesses = to_int_arg(need_value("--batch-guesses"), "--batch-guesses");
        } else if (arg == "--batch-pts") {
            opt.max_gpu_batch_pts = to_int_arg(need_value("--batch-pts"), "--batch-pts");
        } else if (arg == "--store-guesses") {
            opt.store_guesses = true;
        } else if (arg == "--help") {
            cout << "Usage: gpu_guess.exe --train <path> [--limit N] [--cpu-threshold N]\n"
                 << "                     [--batch-guesses N] [--batch-pts N] [--store-guesses]\n";
            exit(0);
        } else {
            throw runtime_error("Unknown argument: " + arg);
        }
    }

    return opt;
}

segment* find_segment(PriorityQueue& q, const segment& seg) {
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

GpuPtTask build_task(PriorityQueue& q, const PT& pt) {
    GpuPtTask task;
    int seg_count = static_cast<int>(pt.content.size());

    if (seg_count == 0) {
        return task;
    }

    if (seg_count == 1) {
        segment* last = find_segment(q, pt.content[0]);
        if (last != nullptr) {
            task.suffixes = last->ordered_values;
        }
        return task;
    }

    string prefix;
    int seg_idx = 0;
    for (int idx : pt.curr_indices) {
        if (seg_idx >= seg_count - 1) {
            break;
        }
        segment* cur = find_segment(q, pt.content[seg_idx]);
        if (cur != nullptr && idx >= 0 && idx < static_cast<int>(cur->ordered_values.size())) {
            prefix += cur->ordered_values[idx];
        }
        ++seg_idx;
    }

    segment* last = find_segment(q, pt.content[seg_count - 1]);
    if (last != nullptr) {
        task.prefix = prefix;
        task.suffixes = last->ordered_values;
    }

    return task;
}

void insert_pt_by_prob(PriorityQueue& q, PT pt) {
    q.CalProb(pt);

    auto iter = q.priority.begin();
    while (iter != q.priority.end() && iter->prob >= pt.prob) {
        ++iter;
    }
    q.priority.emplace(iter, pt);
}

GpuPtTask pop_task_without_generate(PriorityQueue& q) {
    PT current = q.priority.front();
    q.priority.erase(q.priority.begin());

    GpuPtTask task = build_task(q, current);

    vector<PT> new_pts = current.NewPTs();
    for (PT& pt : new_pts) {
        insert_pt_by_prob(q, pt);
    }

    return task;
}

void generate_on_cpu(const GpuPtTask& task, vector<string>& out, Stats& stats, bool store) {
    double start = now_seconds();
    stats.cpu_tasks++;
    stats.cpu_generated += static_cast<long long>(task.suffixes.size());
    stats.generated += static_cast<long long>(task.suffixes.size());

    long long local_bytes = 0;
    if (store) {
        out.reserve(out.size() + task.suffixes.size());
        for (const string& suffix : task.suffixes) {
            string guess = task.prefix + suffix;
            local_bytes += static_cast<long long>(guess.size());
            out.emplace_back(std::move(guess));
        }
    } else {
        for (const string& suffix : task.suffixes) {
            string guess = task.prefix + suffix;
            local_bytes += static_cast<long long>(guess.size());
        }
    }

    stats.generated_bytes += local_bytes;
    stats.cpu_generate_seconds += now_seconds() - start;
}

void wait_for_gpu_batch(InFlightGpuBatch& in_flight,
                        vector<string>& out,
                        Stats& stats,
                        bool store) {
    if (!in_flight.active) {
        return;
    }

    double wait_start = now_seconds();
    GpuBatchResult result = in_flight.future.get();
    stats.gpu_wait_seconds += now_seconds() - wait_start;

    stats.gpu_wall_seconds += result.wall_seconds;
    stats.gpu_kernel_seconds += static_cast<double>(result.kernel_milliseconds) / 1000.0;

    if (result.guesses.size() != in_flight.expected) {
        throw runtime_error("GPU generated count mismatch");
    }

    for (const string& guess : result.guesses) {
        stats.generated_bytes += static_cast<long long>(guess.size());
    }

    if (store) {
        out.insert(out.end(),
                   make_move_iterator(result.guesses.begin()),
                   make_move_iterator(result.guesses.end()));
    }

    in_flight.active = false;
    in_flight.expected = 0;
}

void launch_gpu_batch(vector<GpuPtTask>& batch,
                      CudaGuessGenerator& gpu,
                      InFlightGpuBatch& in_flight,
                      vector<string>& out,
                      Stats& stats,
                      bool store) {
    if (batch.empty()) {
        return;
    }

    wait_for_gpu_batch(in_flight, out, stats, store);

    size_t expected = count_task_guesses(batch);
    stats.gpu_batches++;
    stats.gpu_tasks += static_cast<long long>(batch.size());
    stats.gpu_generated += static_cast<long long>(expected);
    stats.generated += static_cast<long long>(expected);

    vector<GpuPtTask> launched;
    launched.swap(batch);

    in_flight.expected = expected;
    in_flight.future = std::async(std::launch::async,
                                  [&gpu, launched = std::move(launched)]() mutable {
                                      return gpu.generate_batch(launched);
                                  });
    in_flight.active = true;
}

void print_config(const Options& opt, const CudaGuessGenerator& gpu) {
    cout << "Local CUDA PCFG password guessing experiment" << endl;
    cout << "CUDA device: " << gpu.device_name() << endl;
    cout << "Training file: " << opt.train_path << endl;
    cout << "Generate limit: " << opt.generate_limit << endl;
    cout << "CPU threshold: " << opt.cpu_threshold << " guesses per PT" << endl;
    cout << "GPU batch target: " << opt.target_gpu_batch_guesses << " guesses" << endl;
    cout << "GPU batch PT cap: " << opt.max_gpu_batch_pts << endl;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        CudaGuessGenerator gpu;

        if (!gpu.available()) {
            cerr << "CUDA is not available: " << gpu.device_name() << endl;
            return 1;
        }

        print_config(opt, gpu);

        PriorityQueue q;
        Stats stats;

        double train_start = now_seconds();
        q.m.train(opt.train_path);
        q.m.order();
        q.init();
        stats.train_seconds = now_seconds() - train_start;

        vector<string> stored_guesses;
        vector<GpuPtTask> pending_gpu_batch;
        size_t pending_guess_count = 0;
        InFlightGpuBatch in_flight;

        double run_start = now_seconds();
        long long next_print = opt.print_step;

        while (!q.priority.empty() &&
               stats.generated + static_cast<long long>(pending_guess_count) < opt.generate_limit) {
            GpuPtTask task = pop_task_without_generate(q);
            size_t task_count = task.suffixes.size();

            if (task_count == 0) {
                continue;
            }

            long long scheduled = stats.generated + static_cast<long long>(pending_guess_count);
            if (scheduled >= opt.generate_limit) {
                break;
            }
            if (scheduled + static_cast<long long>(task_count) > opt.generate_limit) {
                size_t keep = static_cast<size_t>(opt.generate_limit - scheduled);
                task.suffixes.resize(keep);
                task_count = keep;
            }

            if (task_count == 0) {
                continue;
            }

            if (static_cast<int>(task_count) < opt.cpu_threshold) {
                generate_on_cpu(task, stored_guesses, stats, opt.store_guesses);
            } else {
                pending_guess_count += task_count;
                pending_gpu_batch.emplace_back(std::move(task));

                bool batch_full =
                    pending_guess_count >= static_cast<size_t>(opt.target_gpu_batch_guesses) ||
                    static_cast<int>(pending_gpu_batch.size()) >= opt.max_gpu_batch_pts;

                if (batch_full) {
                    launch_gpu_batch(pending_gpu_batch,
                                     gpu,
                                     in_flight,
                                     stored_guesses,
                                     stats,
                                     opt.store_guesses);
                    pending_guess_count = 0;
                }
            }

            while (stats.generated >= next_print) {
                cout << "Guesses generated: " << next_print << endl;
                next_print += opt.print_step;
            }
        }

        launch_gpu_batch(pending_gpu_batch,
                         gpu,
                         in_flight,
                         stored_guesses,
                         stats,
                         opt.store_guesses);
        wait_for_gpu_batch(in_flight, stored_guesses, stats, opt.store_guesses);

        stats.total_seconds = now_seconds() - run_start;

        cout << "Guess time:" << stats.total_seconds << "seconds" << endl;
        cout << "Train time:" << stats.train_seconds << "seconds" << endl;
        cout << "CPU generated:" << stats.cpu_generated << endl;
        cout << "GPU generated:" << stats.gpu_generated << endl;
        cout << "CPU PT tasks:" << stats.cpu_tasks << endl;
        cout << "GPU PT tasks:" << stats.gpu_tasks << endl;
        cout << "GPU batches:" << stats.gpu_batches << endl;
        cout << "CPU generate time:" << stats.cpu_generate_seconds << "seconds" << endl;
        cout << "GPU wall time:" << stats.gpu_wall_seconds << "seconds" << endl;
        cout << "GPU kernel time:" << stats.gpu_kernel_seconds << "seconds" << endl;
        cout << "GPU wait time:" << stats.gpu_wait_seconds << "seconds" << endl;
        cout << "Generated bytes checksum:" << stats.generated_bytes << endl;
        cout << "Stored guesses:" << stored_guesses.size() << endl;

        return 0;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
