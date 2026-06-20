#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct GpuPtTask {
    std::string prefix;
    std::vector<std::string> suffixes;
    std::size_t output_start = 0;
};

struct GpuBatchResult {
    std::vector<std::string> guesses;
    double wall_seconds = 0.0;
    float kernel_milliseconds = 0.0f;
};

class CudaGuessGenerator {
public:
    CudaGuessGenerator();
    ~CudaGuessGenerator();

    CudaGuessGenerator(const CudaGuessGenerator&) = delete;
    CudaGuessGenerator& operator=(const CudaGuessGenerator&) = delete;

    bool available() const;
    std::string device_name() const;

    GpuBatchResult generate_batch(const std::vector<GpuPtTask>& tasks);

private:
    struct Impl;
    Impl* impl_;
};

std::size_t count_task_guesses(const std::vector<GpuPtTask>& tasks);
