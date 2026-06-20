#include "gpu_generator.cuh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

namespace {

constexpr int kThreadsPerBlock = 256;

struct DeviceTask {
    int prefix_offset;
    int prefix_len;
    int suffix_first_index;
    int suffix_count;
    int output_start;
};

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

__global__ void generate_kernel(const DeviceTask* tasks,
                                int task_count,
                                const char* prefixes,
                                const char* suffix_chars,
                                const int* suffix_offsets,
                                const int* suffix_lens,
                                char* output_chars,
                                int* output_lens,
                                int max_guess_len) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int total = 0;
    if (task_count > 0) {
        const DeviceTask& last = tasks[task_count - 1];
        total = last.output_start + last.suffix_count;
    }

    if (gid >= total) {
        return;
    }

    int lo = 0;
    int hi = task_count - 1;
    int task_idx = 0;

    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        const DeviceTask& task = tasks[mid];
        if (gid < task.output_start) {
            hi = mid - 1;
        } else if (gid >= task.output_start + task.suffix_count) {
            lo = mid + 1;
        } else {
            task_idx = mid;
            break;
        }
    }

    const DeviceTask& task = tasks[task_idx];
    int local = gid - task.output_start;
    int suffix_idx = task.suffix_first_index + local;
    int suffix_len = suffix_lens[suffix_idx];
    int guess_len = task.prefix_len + suffix_len;
    int out_base = gid * max_guess_len;

    for (int i = 0; i < task.prefix_len; ++i) {
        output_chars[out_base + i] = prefixes[task.prefix_offset + i];
    }

    int suffix_base = suffix_offsets[suffix_idx];
    for (int i = 0; i < suffix_len; ++i) {
        output_chars[out_base + task.prefix_len + i] = suffix_chars[suffix_base + i];
    }

    output_lens[gid] = guess_len;
}

struct PackedBatch {
    std::vector<DeviceTask> tasks;
    std::vector<char> prefixes;
    std::vector<char> suffix_chars;
    std::vector<int> suffix_offsets;
    std::vector<int> suffix_lens;
    int total_guesses = 0;
    int max_guess_len = 1;
};

PackedBatch pack_batch(const std::vector<GpuPtTask>& input) {
    PackedBatch packed;

    int suffix_first = 0;
    int output_start = 0;

    for (const GpuPtTask& task : input) {
        if (task.suffixes.empty()) {
            continue;
        }

        DeviceTask device_task{};
        device_task.prefix_offset = static_cast<int>(packed.prefixes.size());
        device_task.prefix_len = static_cast<int>(task.prefix.size());
        device_task.suffix_first_index = suffix_first;
        device_task.suffix_count = static_cast<int>(task.suffixes.size());
        device_task.output_start = output_start;

        packed.prefixes.insert(packed.prefixes.end(), task.prefix.begin(), task.prefix.end());

        for (const std::string& suffix : task.suffixes) {
            packed.suffix_offsets.push_back(static_cast<int>(packed.suffix_chars.size()));
            packed.suffix_lens.push_back(static_cast<int>(suffix.size()));
            packed.suffix_chars.insert(packed.suffix_chars.end(), suffix.begin(), suffix.end());

            int guess_len = device_task.prefix_len + static_cast<int>(suffix.size());
            packed.max_guess_len = std::max(packed.max_guess_len, guess_len);
        }

        packed.tasks.push_back(device_task);
        suffix_first += device_task.suffix_count;
        output_start += device_task.suffix_count;
    }

    packed.total_guesses = output_start;
    return packed;
}

template <typename T>
void upload_vector(T** dst, const std::vector<T>& src) {
    if (src.empty()) {
        *dst = nullptr;
        return;
    }

    check_cuda(cudaMalloc(reinterpret_cast<void**>(dst), src.size() * sizeof(T)), "cudaMalloc");
    check_cuda(cudaMemcpy(*dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice),
               "cudaMemcpy host to device");
}

}  // namespace

struct CudaGuessGenerator::Impl {
    bool ok = false;
    std::string name;
};

CudaGuessGenerator::CudaGuessGenerator() : impl_(new Impl()) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        impl_->ok = false;
        if (err != cudaSuccess) {
            impl_->name = std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(err);
        } else {
            impl_->name = "No CUDA device";
        }
        return;
    }

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    impl_->ok = true;
    impl_->name = prop.name;
}

CudaGuessGenerator::~CudaGuessGenerator() {
    delete impl_;
}

bool CudaGuessGenerator::available() const {
    return impl_->ok;
}

std::string CudaGuessGenerator::device_name() const {
    return impl_->name;
}

GpuBatchResult CudaGuessGenerator::generate_batch(const std::vector<GpuPtTask>& tasks) {
    if (!available()) {
        throw std::runtime_error("CUDA device is not available");
    }

    GpuBatchResult result;
    PackedBatch packed = pack_batch(tasks);

    if (packed.total_guesses == 0) {
        return result;
    }

    auto wall_start = std::chrono::high_resolution_clock::now();

    DeviceTask* d_tasks = nullptr;
    char* d_prefixes = nullptr;
    char* d_suffix_chars = nullptr;
    int* d_suffix_offsets = nullptr;
    int* d_suffix_lens = nullptr;
    char* d_output_chars = nullptr;
    int* d_output_lens = nullptr;

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};

    try {
        upload_vector(&d_tasks, packed.tasks);
        upload_vector(&d_prefixes, packed.prefixes);
        upload_vector(&d_suffix_chars, packed.suffix_chars);
        upload_vector(&d_suffix_offsets, packed.suffix_offsets);
        upload_vector(&d_suffix_lens, packed.suffix_lens);

        std::size_t output_chars_size =
            static_cast<std::size_t>(packed.total_guesses) * packed.max_guess_len;

        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_output_chars), output_chars_size),
                   "cudaMalloc output chars");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_output_lens),
                              packed.total_guesses * sizeof(int)),
                   "cudaMalloc output lens");

        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");

        int blocks = (packed.total_guesses + kThreadsPerBlock - 1) / kThreadsPerBlock;

        check_cuda(cudaEventRecord(start_event), "cudaEventRecord start");
        generate_kernel<<<blocks, kThreadsPerBlock>>>(d_tasks,
                                                      static_cast<int>(packed.tasks.size()),
                                                      d_prefixes,
                                                      d_suffix_chars,
                                                      d_suffix_offsets,
                                                      d_suffix_lens,
                                                      d_output_chars,
                                                      d_output_lens,
                                                      packed.max_guess_len);
        check_cuda(cudaGetLastError(), "generate_kernel launch");
        check_cuda(cudaEventRecord(stop_event), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop");
        check_cuda(cudaEventElapsedTime(&result.kernel_milliseconds, start_event, stop_event),
                   "cudaEventElapsedTime");

        std::vector<char> output_chars(output_chars_size);
        std::vector<int> output_lens(packed.total_guesses);

        check_cuda(cudaMemcpy(output_chars.data(),
                              d_output_chars,
                              output_chars.size(),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy output chars");
        check_cuda(cudaMemcpy(output_lens.data(),
                              d_output_lens,
                              output_lens.size() * sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy output lens");

        result.guesses.reserve(packed.total_guesses);
        for (int i = 0; i < packed.total_guesses; ++i) {
            const char* start = output_chars.data() + static_cast<std::size_t>(i) * packed.max_guess_len;
            result.guesses.emplace_back(start, output_lens[i]);
        }
    } catch (...) {
        cudaFree(d_tasks);
        cudaFree(d_prefixes);
        cudaFree(d_suffix_chars);
        cudaFree(d_suffix_offsets);
        cudaFree(d_suffix_lens);
        cudaFree(d_output_chars);
        cudaFree(d_output_lens);
        if (start_event) cudaEventDestroy(start_event);
        if (stop_event) cudaEventDestroy(stop_event);
        throw;
    }

    cudaFree(d_tasks);
    cudaFree(d_prefixes);
    cudaFree(d_suffix_chars);
    cudaFree(d_suffix_offsets);
    cudaFree(d_suffix_lens);
    cudaFree(d_output_chars);
    cudaFree(d_output_lens);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.wall_seconds =
        std::chrono::duration<double>(wall_end - wall_start).count();
    return result;
}

std::size_t count_task_guesses(const std::vector<GpuPtTask>& tasks) {
    std::size_t total = 0;
    for (const GpuPtTask& task : tasks) {
        total += task.suffixes.size();
    }
    return total;
}
