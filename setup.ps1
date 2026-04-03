# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Setup script for NHT.
# Creates a .venv with uv, initializes the gsplat submodule, and installs dependencies.

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

function Set-CudaHomeFromToolkit {
    if ($env:CUDA_HOME -and (Test-Path (Join-Path $env:CUDA_HOME "bin\nvcc.exe"))) {
        Write-Host "  Using CUDA_HOME=$($env:CUDA_HOME)" -ForegroundColor DarkGray
        return $true
    }
    $base = "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA"
    if (-not (Test-Path $base)) {
        return $false
    }
    $best = Get-ChildItem $base -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(v)?\d+\.\d+' } |
        ForEach-Object {
            $verStr = $_.Name.TrimStart('v')
            try {
                [PSCustomObject]@{ Dir = $_; Ver = [version]$verStr }
            } catch { $null }
        } |
        Where-Object { $null -ne $_ } |
        Sort-Object -Property Ver -Descending |
        Select-Object -First 1
    if ($best) {
        $env:CUDA_HOME = $best.Dir.FullName
        Write-Host "  Set CUDA_HOME=$($env:CUDA_HOME)" -ForegroundColor Yellow
        return $true
    }
    return $false
}

function Get-PyTorchWheelIndexUrl {
    if ($env:PYTORCH_CUDA_INDEX) {
        $idx = $env:PYTORCH_CUDA_INDEX.Trim()
        if (-not $idx.StartsWith("http")) {
            return "https://download.pytorch.org/whl/$idx"
        }
        return $idx
    }
    $verFile = if ($env:CUDA_HOME) { Join-Path $env:CUDA_HOME "version.txt" } else { $null }
    if ($verFile -and (Test-Path $verFile)) {
        $line = (Get-Content $verFile -TotalCount 1) -join ""
        if ($line -match '(\d+)\.(\d+)') {
            $maj = $matches[1]
            $min = $matches[2]
            return "https://download.pytorch.org/whl/cu$maj$min"
        }
    }
    return "https://download.pytorch.org/whl"
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "NHT Release Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Check that uv is available
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Host "ERROR: 'uv' is not installed." -ForegroundColor Red
    Write-Host "  Install it with:  powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Yellow
    throw "uv is required. Install it and re-run setup."
}

Write-Host "[1/5] Creating virtual environment (.venv, Python 3.11)..." -ForegroundColor Green
uv venv --python 3.11 --prompt nht .venv
& .\.venv\Scripts\Activate.ps1

Write-Host "[2/5] Initializing gsplat submodule..." -ForegroundColor Green
git submodule update --init --recursive

Write-Host "[3/5] CUDA + PyTorch (CUDA wheels)..." -ForegroundColor Green
$cudaOk = Set-CudaHomeFromToolkit
if (-not $cudaOk) {
    Write-Host "  WARNING: CUDA toolkit not found and CUDA_HOME is not set." -ForegroundColor Yellow
    Write-Host "  Install CUDA 12.x (with nvcc) or set CUDA_HOME, then re-run setup." -ForegroundColor Yellow
    Write-Host "  Optional: set PYTORCH_CUDA_INDEX (e.g. cu126, cu128) to match your driver/toolkit." -ForegroundColor Yellow
    throw "Cannot build gsplat without CUDA. Set CUDA_HOME or install the NVIDIA CUDA Toolkit."
}
$wheelUrl = Get-PyTorchWheelIndexUrl
Write-Host "  PyTorch wheel index: $wheelUrl" -ForegroundColor DarkGray
$env:UV_INDEX="pytorch=$wheelUrl"

# Setup TORCH_CUDA_ARCH_LIST
$torchCudaArchList = (uv run python -c "import torch,re; print(';'.join(re.sub(r'sm_(\d+)(\d)([a-z]?)$',lambda m:m[1]+'.'+m[2]+m[3],s) for s in torch.cuda.get_arch_list()))")
if (-not $torchCudaArchList) {
    Write-Host "  WARNING: No CUDA architecture list found for torch. Using default: 9.0" -ForegroundColor Yellow
    $torchCudaArchList = "9.0"
}
$env:TORCH_CUDA_ARCH_LIST = $torchCudaArchList + "+PTX"

# Setup TCNN_CUDA_ARCHITECTURES
$tcnnCudaArchList = (uv run python -c "import torch,re; print(';'.join(re.sub(r'sm_(\d+)(\d)([a-z]?)$',lambda m:m[1]+m[2]+m[3],s) for s in torch.cuda.get_arch_list()))")
if (-not $tcnnCudaArchList) {
    Write-Host "  WARNING: No CUDA architecture list found for tcnn. Using default: 90" -ForegroundColor Yellow
    $tcnnCudaArchList = "90"
}
$env:TCNN_CUDA_ARCHITECTURES = $tcnnCudaArchList
Write-Host "  TCNN_CUDA_ARCHITECTURES: $($env:TCNN_CUDA_ARCHITECTURES)" -ForegroundColor DarkGray

# Install dependencies
Write-Host "[4b/5] Installing 'aov' package (AOV helpers)..." -ForegroundColor Green
uv pip install --no-build-isolation -e .

Write-Host "[4a/5] Installing gsplat..." -ForegroundColor Green
uv pip install --no-build-isolation -e ./gsplat

Write-Host "[5/5] Installing example dependencies..." -ForegroundColor Green
$examplesReq = Join-Path $PSScriptRoot "gsplat\examples\requirements.txt"
$isWindows = ($PSVersionTable.PSVersion.Major -ge 6 -and $IsWindows) -or ($PSVersionTable.PSVersion.Major -lt 6 -and $env:OS -match "Windows")
if ($isWindows) {
    # PyTorch 2.9+ on Windows: nvcc host passes do not define _WIN32, so torch dynamo headers take the
    # wrong branch and MSVC fails with C2872 'std': ambiguous symbol (pytorch#148317). fused-ssim's setup
    # does not add the workaround; install a pinned clone with extra nvcc defines, then the rest of reqs.
    $fusedSsimCommit = "328dc9836f513d00c4b5bc38fe30478b4435cbb5"
    $filteredReq = Join-Path $env:TEMP "nht-examples-req-no-fused-ssim.txt"
    Get-Content $examplesReq | Where-Object { $_ -notmatch "rahul-goel/fused-ssim" } | Set-Content -Path $filteredReq -Encoding utf8

    $fusedDir = Join-Path $env:TEMP "nht-fused-ssim-$($fusedSsimCommit.Substring(0, 7))"
    if (Test-Path $fusedDir) { Remove-Item -Recurse -Force $fusedDir }
    git clone --filter=blob:none "https://github.com/rahul-goel/fused-ssim.git" $fusedDir
    Push-Location $fusedDir
    try {
        git checkout $fusedSsimCommit
        $setupPy = Join-Path $fusedDir "setup.py"
        $setupText = Get-Content $setupPy -Raw
        if ($setupText -notmatch "D_WIN32=1") {
            $patch = @'

# Workaround: nvcc on Windows must see _WIN32 for torch dynamo headers (pytorch#148317 / SageAttention#101).
if os.name == "nt":
    nvcc_args.extend(["-D_WIN32=1", "-DUSE_CUDA=1"])

'@
            $setupText = $setupText -replace '(?m)^setup\(', ($patch + "setup(")
            $utf8NoBom = New-Object System.Text.UTF8Encoding $false
            [System.IO.File]::WriteAllText($setupPy, $setupText, $utf8NoBom)
        }
        uv pip install --no-build-isolation .
    } finally {
        Pop-Location
    }
    uv pip install --no-build-isolation -r $filteredReq
} else {
    uv pip install --no-build-isolation -r $examplesReq
}

Write-Host ""
Write-Host "Setup complete. Activate the environment, then run:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  .\scripts\train.ps1                              # Train a scene"
Write-Host "  .\scripts\view.ps1 -Ckpt <path>                  # View a trained model"
Write-Host "  .\benchmarks\nht\benchmark_XXX.ps1               # Reproduce paper results"

Pop-Location
