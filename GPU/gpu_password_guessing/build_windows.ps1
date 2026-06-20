param(
    [string]$CudaPath = "",
    [string]$Out = "gpu_guess.exe",
    [string]$VsPath = ""
)

$ErrorActionPreference = "Stop"

if ($CudaPath -eq "") {
    $candidates = @(
        "$env:CUDA_PATH\bin\nvcc.exe",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7\bin\nvcc.exe",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
    )

    $nvcc = $null
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            $nvcc = $candidate
            break
        }
    }

    if ($null -eq $nvcc) {
        $cmd = Get-Command nvcc.exe -ErrorAction SilentlyContinue
        if ($cmd) {
            $nvcc = $cmd.Source
        }
    }
} else {
    $nvcc = Join-Path $CudaPath "bin\nvcc.exe"
}

if ($null -eq $nvcc -or !(Test-Path $nvcc)) {
    Write-Host "nvcc.exe was not found."
    Write-Host "Install NVIDIA CUDA Toolkit, or run:"
    Write-Host "  .\build_windows.ps1 -CudaPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7'"
    exit 1
}

Write-Host "Using nvcc: $nvcc"

if ($VsPath -eq "") {
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $VsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    }
}

if ($VsPath -eq "") {
    $knownVsPaths = @(
        "D:\Program Files\Microsoft Visual Studio\2022\Community",
        "C:\Program Files\Microsoft Visual Studio\2022\Community",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    )

    foreach ($path in $knownVsPaths) {
        if (Test-Path $path) {
            $VsPath = $path
            break
        }
    }
}

if ($VsPath -eq "" -or !(Test-Path $VsPath)) {
    Write-Host "Visual Studio with C++ tools was not found."
    Write-Host "Pass it explicitly, for example:"
    Write-Host "  .\build_windows.ps1 -VsPath 'D:\Program Files\Microsoft Visual Studio\2022\Community'"
    exit 1
}

$msvcRoot = Join-Path $VsPath "VC\Tools\MSVC"
$msvcVersion = Get-ChildItem -Path $msvcRoot -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending |
    Select-Object -First 1

if ($null -eq $msvcVersion) {
    Write-Host "MSVC tools were not found under: $msvcRoot"
    exit 1
}

$clBin = Join-Path $msvcVersion.FullName "bin\Hostx64\x64"
$clExe = Join-Path $clBin "cl.exe"

if (!(Test-Path $clExe)) {
    Write-Host "cl.exe was not found: $clExe"
    exit 1
}

Write-Host "Using MSVC: $clExe"

& $nvcc -std=c++17 -O2 `
    -ccbin "$clBin" `
    main_gpu.cu gpu_generator.cu train.cpp guessing.cpp `
    -o $Out

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "Built $Out"
