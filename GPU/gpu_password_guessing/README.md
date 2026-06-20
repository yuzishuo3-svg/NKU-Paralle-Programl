# GPU Password Guessing Experiment

This directory is a local NVIDIA/CUDA version of the PCFG password guessing GPU experiment.

It keeps the original PCFG training and priority-queue logic, then replaces the two PT-internal generation loops with a CUDA batch generator.

## What It Implements

- Basic requirement: GPU parallelizes the two loops that fill the last segment of a PT.
- Advanced 1: multiple PTs are packed into one GPU batch instead of launching one PT at a time.
- Advanced 2: CPU and GPU are pipelined. While one GPU batch is running, the CPU continues popping PTs, generating small PTs locally, and preparing the next GPU batch.
- Advanced 3: task-size-aware scheduling. PTs with fewer suffix values than `--cpu-threshold` stay on CPU; larger PTs are sent to GPU. GPU batches are filled by target candidate count, not by a fixed PT count.

## Files

- `main_gpu.cu`: experiment driver and CPU/GPU scheduling policy.
- `gpu_generator.cu`: CUDA multi-PT candidate generator.
- `gpu_generator.cuh`: generator interface.
- `PCFG.h`, `train.cpp`: copied from the original password guessing framework.
- `build_windows.ps1`: local Windows build script.

## Build On Windows

You need NVIDIA CUDA Toolkit, not only the driver. `nvidia-smi` showing a CUDA version means the driver is installed, but `nvcc.exe` still requires the Toolkit.

```powershell
cd C:\Users\Surrender\Desktop\GPUBX\gpu_password_guessing
.\build_windows.ps1
```

If CUDA is installed but not in PATH:

```powershell
.\build_windows.ps1 -CudaPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7"
```

## Run

Pass a local training dictionary path explicitly. This avoids the server-only `/guessdata/...` path.

```powershell
.\gpu_guess.exe --train "C:\path\to\Rockyou-singleLined-full.txt" --limit 10000000
```

Useful parameters:

```powershell
.\gpu_guess.exe --train "C:\path\to\Rockyou-singleLined-full.txt" `
  --limit 10000000 `
  --cpu-threshold 4096 `
  --batch-guesses 200000 `
  --batch-pts 128
```

## Suggested Experiments

Compare these configurations:

```powershell
.\gpu_guess.exe --train "C:\path\to\Rockyou-singleLined-full.txt" --cpu-threshold 1 --batch-guesses 50000
.\gpu_guess.exe --train "C:\path\to\Rockyou-singleLined-full.txt" --cpu-threshold 4096 --batch-guesses 200000
.\gpu_guess.exe --train "C:\path\to\Rockyou-singleLined-full.txt" --cpu-threshold 20000 --batch-guesses 500000
```

Interpretation:

- Too small `--cpu-threshold`: many tiny PTs go to GPU and launch/transfer overhead may dominate.
- Too large `--cpu-threshold`: GPU is underused.
- Larger `--batch-guesses`: fewer launches and better GPU utilization, but less frequent overlap and larger memory pressure.

Key output lines for the report:

- `Guess time`
- `CPU generated`
- `GPU generated`
- `CPU PT tasks`
- `GPU PT tasks`
- `GPU batches`
- `GPU wall time`
- `GPU kernel time`
- `GPU wait time`

`GPU wait time` is the visible blocking time after CPU-side overlap. If it is much smaller than `GPU wall time`, the pipeline successfully hid part of the GPU work behind CPU computation.
