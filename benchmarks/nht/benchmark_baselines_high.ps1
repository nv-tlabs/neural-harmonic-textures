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
    Baseline high primitive-count benchmark for 3DGS-MCMC, 3DGUT-MCMC, and 2DGS.
    LPIPS is evaluated with VGG (normalize=True). This differs from INRIA 3DGS
    which uses VGG with normalize=False (technically incorrect).

.DESCRIPTION
    Trains and evaluates three baseline methods sequentially using the same
    per-scene high-quality primitive caps as the NHT benchmark (Table 7).

    Methods:
      3dgs_mcmc   - 3DGS with MCMC strategy  (simple_trainer.py mcmc)
      3dgut_mcmc  - 3DGUT with MCMC strategy  (simple_trainer.py mcmc --with_ut --with_eval3d)
      2dgs        - 2DGS with default strategy (simple_trainer_2dgs.py)

    Per-scene caps (MCMC methods only; 2DGS uses default densification):
      bonsai=1.3M counter=1.2M kitchen=1.8M room=1.5M
      garden=5.2M bicycle=5.9M stump=4.75M treehill=3.5M flowers=3M
      train=1.1M truck=2.6M drjohnson=3.4M playroom=2.5M

    Modes controlled by flags:
      (default)        Train + eval + collect metrics
      -MetricsOnly     Skip training; collect and display metrics only

    -DataRoot defaults to <repo>\data when empty.

    Result layout: <OutputRoot>\<method>\<scene>
      (default OutputRoot: <repo>\results\benchmark_baselines_high)

.EXAMPLE
    .\benchmarks\nht\benchmark_baselines_high.ps1
    .\benchmarks\nht\benchmark_baselines_high.ps1 -Scenes garden,truck
    .\benchmarks\nht\benchmark_baselines_high.ps1 -Methods 3dgs_mcmc,2dgs
    .\benchmarks\nht\benchmark_baselines_high.ps1 -OutputRoot D:\runs\baselines_high
    .\benchmarks\nht\benchmark_baselines_high.ps1 -MetricsOnly -Step 29999
    .\benchmarks\nht\benchmark_baselines_high.ps1 -SkipTandT -SkipDB
#>
param(
    [string]$Scenes      = "",
    [string]$Methods     = "",
    [string]$DataRoot    = "",
    [string]$OutputRoot  = "",
    [int]$GPU            = 0,
    [int]$MaxSteps       = 30000,
    [int]$Step           = -1,
    [ValidateSet("vgg","alex")]
    [string]$LpipsNet    = "vgg",
    [switch]$NoLpipsNormalize,
    [switch]$MetricsOnly,
    [switch]$SkipMipNeRF360,
    [switch]$SkipTandT,
    [switch]$SkipDB
)

$ErrorActionPreference = "Stop"
$lpipsArgs = @("--lpips_net", $LpipsNet)
if ($NoLpipsNormalize) { $lpipsArgs += "--no-lpips_normalize" }

$RepoRoot     = (Resolve-Path "$PSScriptRoot\..\..").Path
$Trainer3DGS  = "$RepoRoot\gsplat\examples\simple_trainer.py"
$Trainer2DGS  = "$RepoRoot\gsplat\examples\simple_trainer_2dgs.py"

if (-not $DataRoot) { $DataRoot = "$RepoRoot\data" }

# Per-scene high-quality primitive caps (matching standard 3DGS-MCMC)
$sceneCaps = @{
    "bonsai"    = 1300000;  "counter"   = 1200000;  "kitchen" = 1800000;  "room"     = 1500000
    "garden"    = 5200000;  "bicycle"   = 5900000;  "stump"   = 4750000;  "treehill" = 3500000
    "flowers"   = 3000000;  "train"     = 1100000;  "truck"   = 2600000
    "drjohnson" = 3400000;  "playroom"  = 2500000
}

# --------------- Dataset definitions ---------------
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
    $filter = ($Scenes -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $jobs = @($jobs | Where-Object { $filter -contains $_[0] })
}

# --------------- Method selection ---------------
$allMethods = @("3dgs_mcmc", "3dgut_mcmc", "2dgs")
if ($Methods) {
    $methodList = ($Methods -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    foreach ($m in $methodList) {
        if ($allMethods -notcontains $m) {
            Write-Error "Unknown method '$m'. Valid methods: $($allMethods -join ', ')"
            return
        }
    }
} else {
    $methodList = $allMethods
}

if (-not $OutputRoot) { $OutputRoot = "$RepoRoot\results\benchmark_baselines_high" }
else { $OutputRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot) }
$resultBase = $OutputRoot
$sceneNames = $jobs | ForEach-Object { $_[0] }
$allScenes  = $jobs | ForEach-Object { $_[0] }

# --------------- Helpers ---------------
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

# ===================================================================
# TRAIN + EVAL  (skipped with -MetricsOnly)
# ===================================================================
if (-not $MetricsOnly) {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Baselines High-Count Benchmark (per-scene caps)" -ForegroundColor Cyan
    Write-Host "  Methods:  $($methodList -join ', ')" -ForegroundColor Green
    Write-Host "  Scenes:   $($sceneNames -join ', ')" -ForegroundColor Green
    Write-Host "  Steps:    $MaxSteps" -ForegroundColor Green
    Write-Host "  Results:  $resultBase" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Cyan

    foreach ($method in $methodList) {
        Write-Host "`n============================================================" -ForegroundColor Cyan
        Write-Host "Method: $method" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan

        foreach ($job in $jobs) {
            $scene = $job[0]; $dataDir = $job[1]; $factor = $job[2]; $dataset = $job[3]
            $capMax = if ($sceneCaps.ContainsKey($scene)) { $sceneCaps[$scene] } else { 1000000 }
            $resultDir = "$resultBase/$method/$scene"

            Write-Host "`n>>> [$dataset] $method - Training $scene (factor=$factor, cap=$capMax) <<<" -ForegroundColor Yellow

            $commonArgs = @(
                "--disable_viewer",
                "--data_dir", $dataDir,
                "--data_factor", $factor,
                "--result_dir", $resultDir,
                "--max_steps", $MaxSteps
            ) + $lpipsArgs

            switch ($method) {
                "3dgs_mcmc" {
                    $trainArgs = @($Trainer3DGS, "mcmc") + $commonArgs + @(
                        "--strategy.cap-max", $capMax
                    )
                }
                "3dgut_mcmc" {
                    $trainArgs = @($Trainer3DGS, "mcmc") + $commonArgs + @(
                        "--strategy.cap-max", $capMax,
                        "--with_ut",
                        "--with_eval3d"
                    )
                }
                "2dgs" {
                    $trainArgs = @($Trainer2DGS) + $commonArgs
                }
            }

            $env:CUDA_VISIBLE_DEVICES = $GPU
            python @trainArgs
        }
    }
}

# ===================================================================
# METRICS
# ===================================================================
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Results Summary" -ForegroundColor Cyan
Write-Host "  Source: $resultBase" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

$metricKeys = @("psnr", "ssim", "lpips", "num_GS")

# Collect metrics per method
$perMethodMetrics = @{}

foreach ($method in $methodList) {
    Write-Host "`n------------------------------------------------------------" -ForegroundColor Cyan
    Write-Host "Method: $method" -ForegroundColor Cyan
    Write-Host "------------------------------------------------------------" -ForegroundColor Cyan

    $allMetrics = @{}
    $missing    = @()

    foreach ($job in $jobs) {
        $scene = $job[0]; $dataset = $job[3]
        $statsDir = "$resultBase/$method/$scene/stats"

        Write-Host "`n  [$dataset] $scene" -ForegroundColor Magenta

        if (-not (Test-Path $statsDir)) {
            Write-Host "    ** NO STATS DIR **" -ForegroundColor Red
            $missing += $scene; continue
        }

        if ($Step -gt 0) {
            $valFile = Get-ChildItem "$statsDir/val_step*.json" -ErrorAction SilentlyContinue `
                | Where-Object { $_.Name -notmatch "per_image" -and $_.Name -match "step0*${Step}\." } `
                | Select-Object -First 1
        } else {
            $valFile = Get-ChildItem "$statsDir/val_step*.json" -ErrorAction SilentlyContinue `
                | Where-Object { $_.Name -notmatch "per_image" } `
                | Sort-Object { [int]($_.Name -replace '.*step(\d+).*','$1') } -Descending `
                | Select-Object -First 1
        }

        if (-not $valFile) {
            $available = Get-ChildItem "$statsDir/val_step*.json" -ErrorAction SilentlyContinue `
                | Where-Object { $_.Name -notmatch "per_image" } | ForEach-Object { $_.Name }
            Write-Host "    ** MISSING step $(if($Step -gt 0){$Step}else{'latest'}) **  (available: $($available -join ', '))" -ForegroundColor Red
            $missing += $scene; continue
        }

        $json = Get-Content $valFile.FullName -Raw | ConvertFrom-Json
        Write-Host ("    $($valFile.Name):  PSNR={0:F3}  SSIM={1:F4}  LPIPS={2:F3}  #GS={3}" -f `
            $json.psnr, $json.ssim, $json.lpips, [int]$json.num_GS) -ForegroundColor Green
        $allMetrics[$scene] = @{}
        foreach ($k in $metricKeys) {
            if ($null -ne $json.$k) { $allMetrics[$scene][$k] = [double]$json.$k }
        }
    }

    $perMethodMetrics[$method] = $allMetrics

    if ($allMetrics.Count -gt 0) {
        $rows = @()
        $rows += Get-AggregatedRow "M360-In"  $m360Indoor   $allMetrics $metricKeys
        $rows += Get-AggregatedRow "M360-Out" $m360Outdoor   $allMetrics $metricKeys
        $rows += Get-AggregatedRow "M360"     ($m360Indoor + $m360Outdoor) $allMetrics $metricKeys
        $rows += Get-AggregatedRow "T&T"      $tandtScenes   $allMetrics $metricKeys
        $rows += Get-AggregatedRow "DB"       $dbScenes      $allMetrics $metricKeys
        $rows += Get-AggregatedRow "Overall"  $allScenes     $allMetrics $metricKeys

        Write-Host "`n  Aggregated ($method):" -ForegroundColor Cyan
        $header = "{0,-10} {1,4}  {2,8}  {3,8}  {4,8}  {5,10}" -f `
            "Split", "N", "PSNR", "SSIM", "LPIPS", "#GS"
        Write-Host $header -ForegroundColor White
        Write-Host ("-" * $header.Length) -ForegroundColor DarkGray

        foreach ($r in $rows) {
            if ($null -eq $r) { continue }
            $numGS = if ($null -ne $r["num_GS"]) { "{0,10:N0}" -f [int]$r["num_GS"] } else { "       N/A" }
            $line = "{0,-10} {1,4}  {2,8:F3}  {3,8:F4}  {4,8:F3}  {5}" -f `
                $r.Label, $r.N, $r["psnr"], $r["ssim"], $r["lpips"], $numGS
            $c = if ($aggColors.ContainsKey($r.Label)) { $aggColors[$r.Label] } else { "White" }
            Write-Host $line -ForegroundColor $c
        }
    }

    if ($missing.Count -gt 0) {
        Write-Host "`n  MISSING ($method): $($missing -join ', ')" -ForegroundColor Red
    }
}

# ===================================================================
# CROSS-METHOD COMPARISON
# ===================================================================
if ($perMethodMetrics.Count -gt 1) {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "Cross-Method Comparison (Overall Averages)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $header = "{0,-14} {1,4}  {2,8}  {3,8}  {4,8}  {5,12}" -f `
        "Method", "N", "PSNR", "SSIM", "LPIPS", "Avg #GS"
    Write-Host $header -ForegroundColor White
    Write-Host ("-" * $header.Length) -ForegroundColor DarkGray

    foreach ($method in $methodList) {
        $mData = $perMethodMetrics[$method]
        if (-not $mData -or $mData.Count -eq 0) { continue }
        $row = Get-AggregatedRow $method $allScenes $mData $metricKeys
        if ($null -eq $row) { continue }
        $numGS = if ($null -ne $row["num_GS"]) { "{0,12:N0}" -f [int]$row["num_GS"] } else { "         N/A" }
        $line = "{0,-14} {1,4}  {2,8:F3}  {3,8:F4}  {4,8:F3}  {5}" -f `
            $row.Label, $row.N, $row["psnr"], $row["ssim"], $row["lpips"], $numGS
        Write-Host $line -ForegroundColor White
    }

    Write-Host "`n--- Markdown table ---" -ForegroundColor DarkGray
    Write-Host "| Method         |  N | PSNR  | SSIM   | LPIPS | Avg #GS     |"
    Write-Host "|----------------|----|-------|--------|-------|-------------|"
    foreach ($method in $methodList) {
        $mData = $perMethodMetrics[$method]
        if (-not $mData -or $mData.Count -eq 0) { continue }
        $row = Get-AggregatedRow $method $allScenes $mData $metricKeys
        if ($null -eq $row) { continue }
        $numGS = if ($null -ne $row["num_GS"]) { "{0:N0}" -f [int]$row["num_GS"] } else { "N/A" }
        Write-Host ("| {0,-14} | {1,2} | {2:F3} | {3:F4} | {4:F3} | {5,-11} |" -f `
            $row.Label, $row.N, $row["psnr"], $row["ssim"], $row["lpips"], $numGS)
    }

    # Per-split comparison across methods
    $splits = @(
        @("M360-In",  $m360Indoor),
        @("M360-Out", $m360Outdoor),
        @("M360",     ($m360Indoor + $m360Outdoor)),
        @("T&T",      $tandtScenes),
        @("DB",       $dbScenes),
        @("Overall",  $allScenes)
    )

    Write-Host "`n--- Per-split comparison ---" -ForegroundColor DarkGray
    foreach ($splitDef in $splits) {
        $splitName = $splitDef[0]
        $splitScenes = $splitDef[1]
        Write-Host "`n  $splitName" -ForegroundColor $(if ($aggColors.ContainsKey($splitName)) { $aggColors[$splitName] } else { "White" })
        $subHeader = "    {0,-14} {1,4}  {2,8}  {3,8}  {4,8}  {5,12}" -f `
            "Method", "N", "PSNR", "SSIM", "LPIPS", "Avg #GS"
        Write-Host $subHeader -ForegroundColor DarkGray
        foreach ($method in $methodList) {
            $mData = $perMethodMetrics[$method]
            if (-not $mData -or $mData.Count -eq 0) { continue }
            $row = Get-AggregatedRow $method $splitScenes $mData $metricKeys
            if ($null -eq $row) { continue }
            $numGS = if ($null -ne $row["num_GS"]) { "{0,12:N0}" -f [int]$row["num_GS"] } else { "         N/A" }
            $line = "    {0,-14} {1,4}  {2,8:F3}  {3,8:F4}  {4,8:F3}  {5}" -f `
                $row.Label, $row.N, $row["psnr"], $row["ssim"], $row["lpips"], $numGS
            Write-Host $line -ForegroundColor White
        }
    }
}
