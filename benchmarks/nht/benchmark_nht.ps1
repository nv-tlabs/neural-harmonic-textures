# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

<#
.SYNOPSIS
    Run the full NHT MCMC benchmark across Mip-NeRF 360, Tanks & Temples,
    and Deep Blending datasets. Reproduces Table 2 from NHT paper.
    LPIPS is evaluated with VGG (normalize=True). This differs from INRIA 3DGS
    which uses VGG with normalize=False (technically incorrect).

.DESCRIPTION
    Result layout: <OutputRoot>\<scene> (default OutputRoot: <repo>\results\benchmark_nht)

.EXAMPLE
    .\benchmarks\nht\benchmark_nht.ps1
    .\benchmarks\nht\benchmark_nht.ps1 -Scenes garden,truck
    .\benchmarks\nht\benchmark_nht.ps1 -OutputRoot D:\runs\nht_table2
    .\benchmarks\nht\benchmark_nht.ps1 -RuntimeOnly
    .\benchmarks\nht\benchmark_nht.ps1 -MetricsOnly -Step 29999
#>
param(
    [string]$Scenes      = "",
    [string]$DataRoot    = "",
    [string]$OutputRoot  = "",
    [int]$CapMax         = 1000000,
    [int]$GPU            = 0,
    [int]$Step           = -1,
    [int]$NumPasses      = 3,
    [int]$WarmupFrames   = 10,
    [switch]$MetricsOnly,
    [switch]$RuntimeOnly,
    [ValidateSet("vgg","alex")]
    [string]$LpipsNet    = "vgg",
    [switch]$NoLpipsNormalize,
    [switch]$SkipMipNeRF360,
    [switch]$SkipTandT,
    [switch]$SkipDB
)

$ErrorActionPreference = "Stop"
$lpipsArgs = @("--lpips_net", $LpipsNet)
if ($NoLpipsNormalize) { $lpipsArgs += "--no-lpips_normalize" }

$RepoRoot    = (Resolve-Path "$PSScriptRoot\..\..").Path
$Trainer     = "$RepoRoot\gsplat\examples\simple_trainer_nht.py"
$Benchmarker = "$RepoRoot\benchmarks\benchmark_nht.py"

if (-not $DataRoot) { $DataRoot = "$RepoRoot\data" }

$m360Indoor  = @("bonsai", "counter", "kitchen", "room")
$m360Outdoor = @("garden", "bicycle", "stump", "treehill", "flowers")
$tandtScenes = @("train", "truck")
$dbScenes    = @("drjohnson", "playroom")

$jobs = @()
if (-not $SkipMipNeRF360) {
    foreach ($s in ($m360Indoor + $m360Outdoor)) {
        $factor = if ($m360Indoor -contains $s) { 2 } else { 4 }
        $jobs += ,@($s, "$DataRoot/mipnerf360/$s", $factor, "mipnerf360")
    }
}
if (-not $SkipTandT) {
    foreach ($s in $tandtScenes) {
        $jobs += ,@($s, "$DataRoot/tandt_db/tandt/$s", 1, "tandt")
    }
}
if (-not $SkipDB) {
    foreach ($s in $dbScenes) {
        $jobs += ,@($s, "$DataRoot/tandt_db/db/$s", 1, "deepblending")
    }
}

if ($Scenes) {
    # Trim each token; @(...) ensures a single matching job stays an array of one tuple
    # (otherwise foreach iterates the tuple's elements and treats the scene name as char[]).
    $filter = ($Scenes -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $jobs = @($jobs | Where-Object { $filter -contains $_[0] })
}

if (-not $OutputRoot) { $OutputRoot = "$RepoRoot\results\benchmark_nht" }
else { $OutputRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot) }
$resultBase = $OutputRoot
$sceneNames = $jobs | ForEach-Object { $_[0] }
$allScenes  = $jobs | ForEach-Object { $_[0] }

function Get-AggregatedRow {
    param([string]$Label, [string[]]$SceneSubset, [hashtable]$AllMetrics, [string[]]$Keys)
    $valid = $SceneSubset | Where-Object { $AllMetrics.ContainsKey($_) }
    if (-not $valid -or $valid.Count -eq 0) { return $null }
    $row = [ordered]@{ Label = $Label; N = $valid.Count }
    foreach ($k in $Keys) {
        $vals = $valid | ForEach-Object { $AllMetrics[$_][$k] } | Where-Object { $null -ne $_ }
        if ($vals.Count -gt 0) {
            $row[$k] = ($vals | Measure-Object -Average).Average
        } else { $row[$k] = $null }
    }
    return $row
}

$aggColors = @{
    "M360-In" = "Yellow"; "M360-Out" = "Green"; "M360" = "Cyan"
    "T&T" = "Magenta"; "DB" = "DarkYellow"; "Overall" = "White"
}

if (-not $MetricsOnly -and -not $RuntimeOnly) {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "NHT MCMC Full Benchmark" -ForegroundColor Cyan
    Write-Host "  Scenes:  $($sceneNames -join ', ')" -ForegroundColor Green
    Write-Host "  CapMax:  $CapMax" -ForegroundColor Green
    Write-Host "  Results: $resultBase" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Cyan

    foreach ($job in $jobs) {
        $scene = $job[0]; $dataDir = $job[1]; $factor = $job[2]; $dataset = $job[3]
        $resultDir = "$resultBase\$scene"

        Write-Host "`n>>> [$dataset] Training $scene (factor=$factor) <<<" -ForegroundColor Yellow

        $trainArgs = @(
            $Trainer, "default",
            "--disable_viewer",
            "--data_dir", $dataDir,
            "--data_factor", $factor,
            "--result_dir", $resultDir,
            "--strategy.cap-max", $CapMax,
            "--ssim_lambda", "0.1",
            "--render_traj_path", "ellipse"
        ) + $lpipsArgs
        $env:CUDA_VISIBLE_DEVICES = $GPU
        python @trainArgs

        $ckptDir = "$resultDir\ckpts"
        if (Test-Path $ckptDir) {
            foreach ($ckpt in (Get-ChildItem "$ckptDir\*.pt" | Sort-Object Name)) {
                Write-Host "  Evaluating $($ckpt.Name) ..." -ForegroundColor Cyan
                $evalArgs = @(
                    $Trainer, "default",
                    "--disable_viewer",
                    "--data_dir", $dataDir,
                    "--data_factor", $factor,
                    "--result_dir", $resultDir,
                    "--strategy.cap-max", $CapMax,
                    "--ssim_lambda", "0.1",
                    "--render_traj_path", "ellipse",
                    "--ckpt", $ckpt.FullName
                ) + $lpipsArgs
                python @evalArgs
            }
        }
    }
}

if (-not $MetricsOnly) {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "Running NHT Timing Benchmark" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $benchArgs = @(
        $Benchmarker,
        "--results_dir", $resultBase,
        "--scene_dir", $DataRoot,
        "--scenes", ($sceneNames -join ","),
        "--num_passes", $NumPasses,
        "--warmup_frames", $WarmupFrames,
        "--gpu", $GPU
    )
    $env:CUDA_VISIBLE_DEVICES = $GPU
    python @benchArgs
}

if (-not $RuntimeOnly) {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "Results Summary" -ForegroundColor Cyan
    Write-Host "  Source: $resultBase" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Cyan

    $allMetrics = @{}
    $metricKeys = @("psnr", "ssim", "lpips", "num_GS")

    foreach ($job in $jobs) {
        $scene = $job[0]; $dataset = $job[3]
        $statsDir = "$resultBase\$scene\stats"

        Write-Host "`n  [$dataset] $scene" -ForegroundColor Magenta
        if (-not (Test-Path $statsDir)) {
            Write-Host "    ** NO STATS DIR **" -ForegroundColor Red
            continue
        }

        $valFile = Get-ChildItem "$statsDir\val_step*.json" -ErrorAction SilentlyContinue `
            | Where-Object { $_.Name -notmatch "per_image" } `
            | Sort-Object { [int]($_.Name -replace '.*step(\d+).*','$1') } -Descending `
            | Select-Object -First 1

        if ($valFile) {
            $json = Get-Content $valFile.FullName -Raw | ConvertFrom-Json
            Write-Host ("    PSNR={0:F3}  SSIM={1:F4}  LPIPS={2:F3}  #GS={3}" -f `
                $json.psnr, $json.ssim, $json.lpips, [int]$json.num_GS) -ForegroundColor Green
            $allMetrics[$scene] = @{}
            foreach ($k in $metricKeys) {
                if ($null -ne $json.$k) { $allMetrics[$scene][$k] = [double]$json.$k }
            }
        }
    }
}
