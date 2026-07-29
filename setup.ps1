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

function Get-PinnedTorchVersion {
    # Parse the pinned torch version from pyproject.toml. Falls back to $null on miss.
    $pp = Join-Path $PSScriptRoot "pyproject.toml"
    if (-not (Test-Path $pp)) { return $null }
    $content = Get-Content $pp -Raw
    # Match e.g. torch==2.9.1 (allow surrounding quotes / spaces).
    $m = [regex]::Match($content, 'torch\s*==\s*(\d+\.\d+(?:\.\d+)?)')
    if ($m.Success) { return $m.Groups[1].Value }
    return $null
}

function Get-CurrentPythonTag {
    # Returns the Python ABI tag (e.g. "cp311") for the currently-active interpreter.
    try {
        $tag = python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')" 2>$null
        if ($tag) { return $tag.Trim() }
    } catch { }
    return "cp311" # Matches the venv we just created with `uv venv --python 3.11`.
}

function Resolve-WindowsCompatibleWheelIndexUrl {
    # On Windows, the auto-detected CUDA toolkit version may not have torch wheels for
    # win_amd64 (e.g. cu129 + torch 2.9.1 ships only Linux wheels at the time of this
    # writing). Probe candidate cu indexes in descending order and pick the first one
    # that actually publishes a Windows wheel for the pinned torch + active Python.
    param([string]$Url)

    if (-not $Url) { return $Url }
    $isWin = ($PSVersionTable.PSVersion.Major -ge 6 -and $IsWindows) -or
             ($PSVersionTable.PSVersion.Major -lt 6 -and $env:OS -match "Windows")
    if (-not $isWin) { return $Url }

    $m = [regex]::Match($Url, "/whl/cu(\d+)/?$")
    if (-not $m.Success) { return $Url }
    $detectedCu = [int]$m.Groups[1].Value

    $torchVer = Get-PinnedTorchVersion
    $pyTag    = Get-CurrentPythonTag
    if (-not $torchVer -or -not $pyTag) { return $Url }

    # Candidate cu versions: detected first, then known Windows-shipped releases.
    $candidates = @($detectedCu, 128, 126, 124, 121, 118) |
        Sort-Object -Unique -Descending |
        Where-Object { $_ -le $detectedCu }

    foreach ($cu in $candidates) {
        $listUrl = "https://download.pytorch.org/whl/cu$cu/torch/"
        try {
            $resp = Invoke-WebRequest -Uri $listUrl -UseBasicParsing -TimeoutSec 20 -ErrorAction Stop
        } catch { continue }
        $needle = "torch-$torchVer+cu$cu-$pyTag-$pyTag-win_amd64.whl"
        if ($resp.Content -match [regex]::Escape($needle)) {
            $picked = "https://download.pytorch.org/whl/cu$cu"
            if ($cu -ne $detectedCu) {
                Write-Host "  Adjusted PyTorch wheel index for Windows: cu$detectedCu -> cu$cu (no Windows wheel for torch $torchVer on cu$detectedCu yet)." -ForegroundColor Yellow
            } else {
                Write-Host "  Verified Windows wheel: torch $torchVer for $pyTag on cu$cu" -ForegroundColor DarkGray
            }
            return $picked
        }
    }

    Write-Host "  WARNING: No Windows wheel found for torch $torchVer ($pyTag) on cu$detectedCu or any known fallback (cu128/126/124/121/118)." -ForegroundColor Yellow
    Write-Host "           Proceeding with the auto-detected index; the install will likely fail." -ForegroundColor Yellow
    Write-Host "           Override manually with `$env:PYTORCH_CUDA_INDEX = 'cuXYZ' and re-run." -ForegroundColor Yellow
    return $Url
}

function Ensure-VsBuildEnvironment {
    # Check if cl.exe is truly on the system PATH (not just a PowerShell alias/module).
    # where.exe searches the real PATH that child processes inherit.
    $whereResult = $null
    try { $whereResult = (& where.exe cl.exe 2>&1) | Where-Object { $_ -is [string] -and (Test-Path $_) } | Select-Object -First 1 } catch {}
    $unsupportedVsPattern = '\\Microsoft Visual Studio\\18\\'
    $hasUnsupportedVsEnv = (($whereResult -and $whereResult -match $unsupportedVsPattern) -or
                            ($env:INCLUDE -and $env:INCLUDE -match $unsupportedVsPattern) -or
                            ($env:LIB -and $env:LIB -match $unsupportedVsPattern) -or
                            ($env:LIBPATH -and $env:LIBPATH -match $unsupportedVsPattern) -or
                            ($env:PATH -and $env:PATH -match $unsupportedVsPattern))
    if ($whereResult -and $env:INCLUDE) {
        if ($hasUnsupportedVsEnv) {
            Write-Host "  WARNING: A Visual Studio 18 build environment is active, which CUDA 12.x nvcc does not support yet." -ForegroundColor Yellow
            Write-Host "           Switching to a Visual Studio 2022 toolchain if available." -ForegroundColor Yellow
        } else {
            Write-Host "  cl.exe already on PATH: $whereResult" -ForegroundColor DarkGray
            Write-Host "  INCLUDE already set ($($env:INCLUDE.Split(';').Count) entries)" -ForegroundColor DarkGray
            $env:DISTUTILS_USE_SDK = "1"
            return
        }
    }

    Write-Host "  MSVC compiler (cl.exe) not found on system PATH (or INCLUDE not set). Setting up VS build environment..." -ForegroundColor Yellow

    # Locate VS installation via vswhere
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vsPath = $null
    if (Test-Path $vswhere) {
        # CUDA 12.x supports the VS 2022 toolset, but VS 18/Build Tools 2026 can
        # make cudafe++ crash with ACCESS_VIOLATION during CUDA extension builds.
        $vsPath = & $vswhere -latest -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if (-not $vsPath) {
            $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        }
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

    if ($hasUnsupportedVsEnv) {
        Write-Host "  Clearing inherited Visual Studio 18 compiler environment..." -ForegroundColor DarkGray
        $varsToClear = @(
            "INCLUDE", "LIB", "LIBPATH",
            "DevEnvDir", "ExtensionSdkDir",
            "Framework40Version", "FrameworkDir", "FrameworkDir64",
            "FrameworkVersion", "FrameworkVersion64",
            "UCRTVersion", "UniversalCRTSdkDir",
            "VCINSTALLDIR", "VCToolsInstallDir", "VCToolsRedistDir", "VCToolsVersion",
            "VisualStudioVersion", "VSINSTALLDIR",
            "VSCMD_ARG_app_plat", "VSCMD_ARG_HOST_ARCH", "VSCMD_ARG_TGT_ARCH", "VSCMD_VER",
            "WindowsLibPath", "WindowsSdkBinPath", "WindowsSdkDir",
            "WindowsSDKLibVersion", "WindowsSdkVerBinPath", "WindowsSDKVersion"
        )
        foreach ($varName in $varsToClear) {
            [System.Environment]::SetEnvironmentVariable($varName, $null, 'Process')
        }
        if ($env:PATH) {
            $env:PATH = (($env:PATH -split ';') |
                Where-Object { $_ -and ($_ -notmatch $unsupportedVsPattern) }) -join ';'
        }
    }

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
    # Temporarily relax error preference since cl.exe writes to stderr.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    $clOutput = & $clCmd.Path 2>&1 | Select-Object -First 1
    $ErrorActionPreference = $prevEAP
    if ($clOutput -match 'Microsoft.*Compiler') {
        Write-Host "  cl.exe version: $clOutput" -ForegroundColor DarkGray
    } elseif ($clOutput -match 'cl|usage') {
        Write-Host "  cl.exe responds (version banner on stderr, build should work)" -ForegroundColor DarkGray
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
# Use a uv-managed (python-build-standalone) interpreter rather than a system
# / Anaconda Python. Recent PyTorch Windows wheels (>=2.9) are built with
# MSVC 19.39+ (VS 2022), and loading their c10.dll / torch_cpu.dll against an
# older CRT (e.g. Anaconda Python's MSC v.1916 from VS 2017) deterministically
# fails with WinError 1114 ("DLL initialization routine failed") because the
# C++ static initialisers can't bind to the older runtime. uv's managed
# interpreters bundle a matching, modern CRT.
#
# --python-preference only-managed forces uv to ignore system Python and
# either reuse a previously-downloaded managed interpreter or download one.
# --allow-existing makes this idempotent (re-running reuses the venv without
# the interactive "replace?" prompt). Delete .venv to force a clean rebuild.
uv python install 3.11

# If a previous run created .venv with a non-managed (Anaconda / system) Python,
# rebuild it from scratch. The interpreter path encodes the source, so we can
# detect this by checking pyvenv.cfg's "home" line.
$venvCfg = Join-Path $PSScriptRoot ".venv\pyvenv.cfg"
if (Test-Path $venvCfg) {
    $homeLine = (Get-Content $venvCfg | Where-Object { $_ -match "^home\s*=" } | Select-Object -First 1)
    $isManaged = $homeLine -and ($homeLine -match "uv[\\/]python|python-build-standalone")
    if (-not $isManaged) {
        Write-Host "  Existing .venv was built from a non-managed Python ($homeLine). Recreating with uv-managed Python to avoid CRT mismatches." -ForegroundColor Yellow
        Remove-Item -Recurse -Force (Join-Path $PSScriptRoot ".venv")
    }
}
uv venv --python 3.11 --python-preference only-managed --prompt nht --allow-existing .venv
& .\.venv\Scripts\Activate.ps1
$pyVerLine = python -c "import sys; print(sys.version)" 2>&1 | Select-Object -First 1
Write-Host "  venv python: $pyVerLine" -ForegroundColor DarkGray

# Ensure the MSVC build environment is available AFTER venv activation.
# Activate.ps1 restores _OLD_VIRTUAL_PATH which would undo any PATH changes made before it.
Ensure-VsBuildEnvironment

Write-Host "[2/5] Initializing gsplat submodule..." -ForegroundColor Green
# NHT builds against the gsplat fork at https://github.com/Arcanous98/gsplat
$gsplatUrl = (git config -f .gitmodules --get submodule.gsplat.url)
$gsplatBranch = (git config -f .gitmodules --get submodule.gsplat.branch)
if (-not $gsplatBranch) { $gsplatBranch = "HEAD" }
Write-Host "  gsplat source: $gsplatUrl (branch $gsplatBranch)" -ForegroundColor DarkGray
git submodule sync --recursive

# Initialize / check out the SHA pinned in the parent repo's index. Do NOT use
# --remote: that would silently fast-forward (or detach) the submodule to
# whatever the configured branch in .gitmodules currently points to on the fork,
# which can clobber local rebase work. The parent repo is the source of truth
# for which gsplat commit goes with which NHT commit.
git submodule update --init --recursive

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
$wheelUrl = Resolve-WindowsCompatibleWheelIndexUrl -Url $wheelUrl
Write-Host "  PyTorch wheel index: $wheelUrl" -ForegroundColor DarkGray
$env:UV_INDEX="pytorch=$wheelUrl"

Write-Host "[3b/5] Installing pytorch and 'nht' package (AOV helpers)..." -ForegroundColor Green
uv pip install -e .

# Setup TORCH_CUDA_ARCH_LIST and TCNN_CUDA_ARCHITECTURES from the locally-installed
# GPU(s) only. Building for every arch torch's wheel supports (sm_70..sm_120) is
# wasteful for an editable / single-machine install and, on Windows + CUDA 12.9,
# fatal: sm_100 / sm_120 pull in cluster-launch headers that hit the known
# `asm operand type size(4) does not match constraint 'l'` bug in
# `cuda/__ptx/instructions/generated/clusterlaunchcontrol.h` (long is 32-bit on
# Windows but the asm constraint expects 64-bit). PTX is appended so the build
# still runs on a slightly newer GPU if the binary is later moved.
#
# A minimum of compute 7.0 (Volta) avoids:
#   error: namespace "cooperative_groups" has no member "labeled_partition"
# on the cg::labeled_partition path. See
# https://github.com/nerfstudio-project/gsplat/issues/653.
#
# Override $env:TORCH_CUDA_ARCH_LIST / $env:TCNN_CUDA_ARCHITECTURES before running
# this script to force a wider build matrix (e.g. for CI that targets multiple
# GPUs).
$minCudaArch = 70

$archDetectScript = @"
import torch
min_arch = $minCudaArch
caps = set()
for i in range(torch.cuda.device_count()):
    maj, minr = torch.cuda.get_device_capability(i)
    cc = maj * 10 + minr
    if cc >= min_arch:
        caps.add((maj, minr))
print(';'.join(f'{m}.{n}' for m, n in sorted(caps)))
"@

# Allow CI / power users to override via a dedicated env var. The script always
# recomputes TORCH_CUDA_ARCH_LIST / TCNN_CUDA_ARCHITECTURES on every invocation,
# so stale wide values left over from earlier runs in the same shell don't keep
# blowing up the build.
if ($env:NHT_TORCH_CUDA_ARCH_LIST) {
    $env:TORCH_CUDA_ARCH_LIST = $env:NHT_TORCH_CUDA_ARCH_LIST
    Write-Host "  TORCH_CUDA_ARCH_LIST: $($env:TORCH_CUDA_ARCH_LIST) (from NHT_TORCH_CUDA_ARCH_LIST)" -ForegroundColor DarkGray
} else {
    $localArchs = uv run python -c $archDetectScript
    if ($localArchs) {
        $env:TORCH_CUDA_ARCH_LIST = $localArchs + "+PTX"
    } else {
        Write-Host "  WARNING: No CUDA-capable GPU detected. Defaulting to TORCH_CUDA_ARCH_LIST=8.9+PTX (RTX 40-series)." -ForegroundColor Yellow
        Write-Host "           Override with `$env:NHT_TORCH_CUDA_ARCH_LIST='X.Y;...' if your target differs." -ForegroundColor Yellow
        $env:TORCH_CUDA_ARCH_LIST = "8.9+PTX"
    }
    Write-Host "  TORCH_CUDA_ARCH_LIST: $($env:TORCH_CUDA_ARCH_LIST)" -ForegroundColor DarkGray
}

# tiny-cuda-nn expects integer-encoded archs (e.g. "89" for sm_89) and always
# emits PTX, so no +PTX suffix here. We also strip any alphabetic arch suffix
# (e.g. "9.0a" -> "90") because tcnn's setup.py runs int() on each entry and
# crashes on Hopper-variant tokens like "90a". gsplat itself still builds for
# the full TORCH_CUDA_ARCH_LIST above (which keeps "9.0a").
$tcnnArchs = ($env:TORCH_CUDA_ARCH_LIST -replace '\+PTX$','').Split(';') |
    ForEach-Object { (($_ -replace '\.','') -replace '[a-zA-Z]','') }
$env:TCNN_CUDA_ARCHITECTURES = ($tcnnArchs -join ';')
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
Write-Host "[4/5] Installing gsplat (with [nht] extra: tinycudann)..." -ForegroundColor Green
# Defensive: --no-build-isolation reuses the active venv's interpreter for the build
# backend, so setuptools / wheel / ninja must already be installed there. They are
# listed in the nht pyproject.toml above, but ensure them here so a partial step [3b]
# doesn't cascade into opaque "ModuleNotFoundError: setuptools" build failures.
uv pip install setuptools wheel ninja

# The [nht] extra pulls tinycudann from git, which recursively initialises
# cutlass. Cutlass's docs tree contains paths longer than Windows' default
# MAX_PATH (260), so without core.longpaths git fails the submodule checkout
# with "Filename too long" and pip aborts metadata generation. We enable it
# globally for the current user before any tcnn-related git submodule init.
git config --global core.longpaths true
uv pip install --no-build-isolation -e "./gsplat[nht]"

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
