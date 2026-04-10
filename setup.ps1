# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Setup script for NHT.
# Creates a .venv with uv, initializes the gsplat submodule, and installs dependencies.

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

$env:DISTUTILS_USE_SDK = "1"

function Set-CudaHomeFromToolkit {
    # Simple, deterministic search for CUDA_HOME

    # 1) respect existing valid CUDA_HOME
    if ($env:CUDA_HOME -and (Test-Path (Join-Path $env:CUDA_HOME "bin\nvcc.exe"))) {
        Write-Host "  [INFO] Using existing CUDA_HOME: $($env:CUDA_HOME)" 'DarkGray'
        return $true
    }

    # 2) nvcc on PATH -> infer parent parent of nvcc.exe
    $nvccCmd = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvccCmd) {
        try {
            $nvccBin = Split-Path $nvccCmd.Path -Parent
            $candidate = Split-Path $nvccBin -Parent
            if (Test-Path (Join-Path $candidate "bin\nvcc.exe")) {
                $env:CUDA_HOME = $candidate
                Write-Host "  Set CUDA_HOME from nvcc: $($env:CUDA_HOME)" -ForegroundColor Yellow
                return $true
            }
        } catch { Write-Host "  Error inferring CUDA_HOME from nvcc: $_" -ForegroundColor Yellow }
    }

    # 3) check common install roots and pick highest version
    $roots = @()
    if ($env:ProgramFiles) { $roots += Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA" }
    if (${env:ProgramFiles(x86)}) { $roots += Join-Path ${env:ProgramFiles(x86)} "NVIDIA GPU Computing Toolkit\CUDA" }
    $roots += "C:\\CUDA"

    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        $vers = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^(v)?\d+\.\d+' }
        if (-not $vers) { continue }
        $best = $vers | ForEach-Object { $_ } | Sort-Object { [version]($_.Name.TrimStart('v')) } -Descending | Select-Object -First 1
        if ($best -and (Test-Path (Join-Path $best.FullName "bin\nvcc.exe"))) {
            $env:CUDA_HOME = $best.FullName
            Write-Host "  Set CUDA_HOME=$($env:CUDA_HOME)" -ForegroundColor Yellow
            return $true
        }
    }

    return $false
}

function Get-PyTorchWheelIndexUrl {
    if ($env:PYTORCH_CUDA_INDEX) {
        $idx = $env:PYTORCH_CUDA_INDEX.Trim()
        if (-not $idx.StartsWith("http")) { return "https://download.pytorch.org/whl/$idx" }
        return $idx
    }

    # helper: parse major.minor from a version-like string
    function Parse-MajorMinor($s) {
        if ($s -and ($s -match '(\d+)\.(\d+)')) { return @{maj=$matches[1]; min=$matches[2]} }
        return $null
    }

    # 1) nvcc --version
    $nvccCmd = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvccCmd) {
        try {
            $out = (& $nvccCmd.Path --version 2>&1) -join ' '
            $pm = Parse-MajorMinor $out
            if ($pm) { return "https://download.pytorch.org/whl/cu$($pm.maj)$($pm.min)" }
        } catch { Write-Host "  [WARNING] nvcc --version failed: $_" -ForegroundColor Yellow }
    }

    # 2) version.json under CUDA_HOME
    if ($env:CUDA_HOME) {
        $jsonFile = Join-Path $env:CUDA_HOME "version.json"
        if (Test-Path $jsonFile) {
            try {
                $j = Get-Content $jsonFile -Raw | ConvertFrom-Json
                $verCandidates = @()
                if ($j.version) { $verCandidates += $j.version }
                if ($j.cuda -and $j.cuda.version) { $verCandidates += $j.cuda.version }
                # scan nested strings
                foreach ($p in $j.PSObject.Properties) {
                    $v = $p.Value
                    if ($v -is [string]) { $verCandidates += $v }
                    elseif ($v -ne $null) {
                        foreach ($sp in $v.PSObject.Properties) { if ($sp.Value -is [string]) { $verCandidates += $sp.Value } }
                    }
                }
                foreach ($c in $verCandidates) { $pm = Parse-MajorMinor $c; if ($pm) { return "https://download.pytorch.org/whl/cu$($pm.maj)$($pm.min)" } }
            } catch { Write-Host "  [WARNING] Failed to parse version.json: $_" -ForegroundColor Yellow }
        }

        # 3) version.txt
        $verFile = Join-Path $env:CUDA_HOME "version.txt"
        if (Test-Path $verFile) {
            try {
                $line = (Get-Content $verFile -TotalCount 1) -join ''
                $pm = Parse-MajorMinor $line
                if ($pm) { return "https://download.pytorch.org/whl/cu$($pm.maj)$($pm.min)" }
            } catch { }
        }
    }

    # 4) fallback: try version.txt adjacent to nvcc (if nvcc found)
    if ($nvccCmd) {
        try {
            $maybeCuda = Split-Path (Split-Path $nvccCmd.Path -Parent) -Parent
            $tryVerFile = Join-Path $maybeCuda "version.txt"
            if (Test-Path $tryVerFile) {
                $line = (Get-Content $tryVerFile -TotalCount 1) -join ''
                $pm = Parse-MajorMinor $line
                if ($pm) { return "https://download.pytorch.org/whl/cu$($pm.maj)$($pm.min)" }
            }
        } catch { }
    }

    # Error out here:
    Write-Host "ERROR: Could not determine CUDA version for PyTorch wheel index. Please set PYTORCH_CUDA_INDEX (e.g. cu126) or provide a valid CUDA_HOME." -ForegroundColor Red
    throw "Cannot determine PyTorch CUDA wheel index. Set PYTORCH_CUDA_INDEX or provide a valid CUDA_HOME."
}

function Ensure-VsBuildEnvironment {
    # Check if cl.exe is truly on the system PATH (not just a PowerShell alias/module).
    # where.exe searches the real PATH that child processes inherit.
    $whereResult = $null
    try { $whereResult = (& where.exe cl.exe 2>&1) | Where-Object { $_ -is [string] -and (Test-Path $_) } | Select-Object -First 1 } catch {}
    if ($whereResult -and $env:INCLUDE) {
        Write-Host "  cl.exe already on PATH: $whereResult" -ForegroundColor DarkGray
        Write-Host "  INCLUDE already set ($($env:INCLUDE.Split(';').Count) entries)" -ForegroundColor DarkGray
        $env:DISTUTILS_USE_SDK = "1"
        return
    }

    Write-Host "  MSVC compiler (cl.exe) not found on system PATH (or INCLUDE not set). Setting up VS build environment..." -ForegroundColor Yellow

    # Locate VS installation via vswhere
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vsPath = $null
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    }
    if (-not $vsPath) {
        foreach ($root in @("$env:ProgramFiles\Microsoft Visual Studio", "${env:ProgramFiles(x86)}\Microsoft Visual Studio")) {
            if (-not (Test-Path $root)) { continue }
            $bat = Get-ChildItem -Path $root -Recurse -Filter "vcvarsall.bat" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($bat) { $vsPath = ($bat.FullName | Split-Path | Split-Path | Split-Path); break }
        }
    }
    if (-not $vsPath) {
        Write-Host "ERROR: No Visual Studio installation with C++ tools found." -ForegroundColor Red
        Write-Host "  Install Visual Studio 2022 (or Build Tools) with the 'Desktop development with C++' workload," -ForegroundColor Yellow
        Write-Host "  or run this script from an 'x64 Native Tools Command Prompt for VS 2022'." -ForegroundColor Yellow
        throw "MSVC (cl.exe) is required to build CUDA extensions. See README for details."
    }
    Write-Host "  Found VS at: $vsPath" -ForegroundColor DarkGray

    # Use vcvarsall.bat to properly set up the full MSVC environment.
    # This is the only reliable way to get PATH, INCLUDE, LIB, and all
    # other variables that cl.exe and nvcc need (vcruntime.h, cassert, etc.).
    $vcvarsall = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"
    if (-not (Test-Path $vcvarsall)) {
        # Fallback: try vcvars64.bat
        $vcvarsall = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    }
    if (-not (Test-Path $vcvarsall)) {
        Write-Host "ERROR: vcvarsall.bat not found under $vsPath" -ForegroundColor Red
        throw "MSVC build environment script not found. Reinstall VS Build Tools with C++ workload."
    }

    Write-Host "  Sourcing: $vcvarsall x64" -ForegroundColor DarkGray
    $vcvarsCmd = if ($vcvarsall -match "vcvarsall") { "`"$vcvarsall`" x64" } else { "`"$vcvarsall`"" }
    $envLines = cmd /c "$vcvarsCmd >nul 2>&1 && set"
    foreach ($line in $envLines) {
        if ($line -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }

    # PyTorch 2.9+ requires this when the VC environment is pre-loaded
    $env:DISTUTILS_USE_SDK = "1"

    # Verify cl.exe actually runs
    $clCmd = Get-Command cl.exe -ErrorAction SilentlyContinue
    if (-not $clCmd) {
        Write-Host "ERROR: cl.exe not found after sourcing vcvarsall.bat." -ForegroundColor Red
        throw "MSVC (cl.exe) is required. Run from an 'x64 Native Tools Command Prompt for VS 2022'."
    }
    Write-Host "  cl.exe on PATH: $($clCmd.Path)" -ForegroundColor DarkGray
    Write-Host "  INCLUDE: $($env:INCLUDE.Split(';').Count) entries" -ForegroundColor DarkGray

    # Smoke test — cl.exe prints its banner to stderr and exits non-zero
    # when called with no arguments, so we capture both streams.
    $clOutput = & $clCmd.Path 2>&1 | Select-Object -First 1
    if ($clOutput -match 'Microsoft.*Compiler') {
        Write-Host "  cl.exe version: $clOutput" -ForegroundColor DarkGray
    } else {
        Write-Host "  WARNING: cl.exe returned unexpected output: $clOutput" -ForegroundColor Yellow
        Write-Host "  Build may still fail. Consider running from an 'x64 Native Tools Command Prompt'." -ForegroundColor Yellow
    }
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

# Ensure the MSVC build environment is available AFTER venv activation.
# Activate.ps1 restores _OLD_VIRTUAL_PATH which would undo any PATH changes made before it.
Ensure-VsBuildEnvironment

Write-Host "[2/5] Initializing gsplat submodule..." -ForegroundColor Green
git submodule update --init --recursive --remote

Write-Host "[3a/5] CUDA + PyTorch (CUDA wheels)..." -ForegroundColor Green
$cudaOk = Set-CudaHomeFromToolkit
if (-not $cudaOk) {
    Write-Host "  WARNING: CUDA toolkit not found and CUDA_HOME is not set." -ForegroundColor Yellow
    Write-Host "  Install CUDA 12.x (with nvcc) or set CUDA_HOME, then re-run setup." -ForegroundColor Yellow
    Write-Host "  Optional: set PYTORCH_CUDA_INDEX (e.g. cu126, cu128) to match your driver/toolkit." -ForegroundColor Yellow
    throw "Cannot build gsplat without CUDA. Set CUDA_HOME or install the NVIDIA CUDA Toolkit."
}
Write-Host "  Found CUDA toolkit at $env:CUDA_HOME" -ForegroundColor DarkGray

$wheelUrl = Get-PyTorchWheelIndexUrl
Write-Host "  PyTorch wheel index: $wheelUrl" -ForegroundColor DarkGray
$env:UV_INDEX="pytorch=$wheelUrl"

Write-Host "[3b/5] Installing pytorch and 'nht' package (AOV helpers)..." -ForegroundColor Green
uv pip install -e .

# Setup TORCH_CUDA_ARCH_LIST and TCNN_CUDA_ARCHITECTURES properly based on the installed PyTorch's supported CUDA architectures, with a 
# minimum of 7.0 (Ampere) to avoid long compile times for older unsupported architectures. Having a minimum capability is to avoid the 
# following compilation issue:
#
# error: namespace "cooperative_groups" has no member "labeled_partition"
# DEBUG       auto warp_group_g = cg::labeled_partition(warp, gid);
# DEBUG                               ^
# DEBUG  
#
# See: https://forums.developer.nvidia.com/t/cuda-11-4-cooperative-groups-no-longer-supported-on-sm-7-0/194001
# See: https://github.com/nerfstudio-project/gsplat/issues/653
#
# Add PTX to TORCH_CUDA_ARCH_LIST to allow JIT compilation for newer architectures not in the list. Tiny-CUDA-NN always generate PTX for 
# each specified architecture, so we don't need to add +PTX to TCNN_CUDA_ARCHITECTURES.
$minCudaArch = 70

$torchCudaArchList = uv run python -c "import torch,re; min_arch = $minCudaArch; archs = set(); [archs.add(m.group(1)+'.'+m.group(2)+m.group(3)) for s in torch.cuda.get_arch_list() for m in [re.match(r'sm_(\d+)(\d)([a-z]?)$', s)] if m and int(m.group(1)+m.group(2)) >= min_arch]; [archs.add(f'{cc//10}.{cc%10}') for i in range(torch.cuda.device_count()) if (cc:=torch.cuda.get_device_capability(i)[0]*10+torch.cuda.get_device_capability(i)[1]) >= min_arch]; print(';'.join(sorted(archs)))"
if (-not $torchCudaArchList) {
    Write-Host "  WARNING: No CUDA architecture list found for torch. Using default: 9.0" -ForegroundColor Yellow
    $torchCudaArchList = "9.0"
}
$env:TORCH_CUDA_ARCH_LIST = $torchCudaArchList + "+PTX"

$tcnnCudaArchList = uv run python -c "import torch,re; min_arch = $minCudaArch; archs = set(); [archs.add(m.group(1)+m.group(2)+m.group(3)) for s in torch.cuda.get_arch_list() for m in [re.match(r'sm_(\d+)(\d)([a-z]?)$', s)] if m and int(m.group(1)+m.group(2)) >= min_arch]; [archs.add(str(cc)) for i in range(torch.cuda.device_count()) if (cc:=torch.cuda.get_device_capability(i)[0]*10+torch.cuda.get_device_capability(i)[1]) >= min_arch]; print(';'.join(sorted(archs)))"
if (-not $tcnnCudaArchList) {
    Write-Host "  WARNING: No CUDA architecture list found for tcnn. Using default: 90" -ForegroundColor Yellow
    $tcnnCudaArchList = "90"
}
$env:TCNN_CUDA_ARCHITECTURES = $tcnnCudaArchList
Write-Host "  TCNN_CUDA_ARCHITECTURES: $($env:TCNN_CUDA_ARCHITECTURES)" -ForegroundColor DarkGray

# Verify cl.exe is visible to child processes before compiling
Write-Host "  [DEBUG] Verifying cl.exe in subprocess..." -ForegroundColor DarkGray
$clCheck = cmd /c "where cl.exe 2>nul"
if ($clCheck) {
    Write-Host "  [DEBUG] cmd sees cl.exe at: $clCheck" -ForegroundColor DarkGray
} else {
    Write-Host "  [DEBUG] cmd CANNOT find cl.exe! PATH length: $($env:PATH.Length) chars" -ForegroundColor Red
    Write-Host "  [DEBUG] First 500 chars of PATH: $($env:PATH.Substring(0, [Math]::Min(500, $env:PATH.Length)))" -ForegroundColor Red
}

# Install dependencies
Write-Host "[4/5] Installing gsplat..." -ForegroundColor Green
uv pip install --no-build-isolation -e ./gsplat

Write-Host "[5/5] Installing example dependencies..." -ForegroundColor Green
$examplesReq = Join-Path $PSScriptRoot "gsplat\examples\requirements.txt"
$isReallyWindows = ($PSVersionTable.PSVersion.Major -ge 6 -and $IsWindows) -or ($PSVersionTable.PSVersion.Major -lt 6 -and $env:OS -match "Windows")
if ($isReallyWindows) {
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
    uv pip install --no-build-isolation --reinstall-package tinycudann -r $filteredReq
} else {
    uv pip install --no-build-isolation --reinstall-package tinycudann -r $examplesReq
}

Write-Host ""
Write-Host "Setup complete. Activate the environment, then run:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  .\scripts\train.ps1                              # Train a scene"
Write-Host "  .\scripts\view.ps1 -Ckpt <path>                  # View a trained model"
Write-Host "  .\benchmarks\nht\benchmark_XXX.ps1               # Reproduce paper results"

Pop-Location
