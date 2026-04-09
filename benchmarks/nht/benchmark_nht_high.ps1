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
    NHT high primitive-count benchmark reproducing Table 7 from the NHT paper.
    LPIPS is evaluated with VGG (normalize=True). This differs from INRIA 3DGS
    which uses VGG with normalize=False (technically incorrect).

.DESCRIPTION
    Trains, evaluates, measures runtime, and collects metrics using per-scene
    high-quality primitive caps matching standard 3DGS-MCMC and 3DGS-trained scenes.

    Per-scene caps:
      bonsai=1.3M counter=1.2M kitchen=1.8M room=1.5M
      garden=5.2M bicycle=5.9M stump=4.75M treehill=3.5M flowers=3M
      train=1.1M truck=2.6M drjohnson=3.4M playroom=2.5M

    Modes controlled by flags:
      (default)        Train + eval + runtime measurement + collect metrics
      -MetricsOnly     Skip training and runtime; collect and display metrics only
      -RuntimeOnly     Skip training; run only runtime measurement + metrics

    -DataRoot defaults to <repo>\data when empty.

    Result layout: <OutputRoot>\<scene> (default OutputRoot: <repo>\results\benchmark_nht_high)

.EXAMPLE
    .\benchmarks\nht\benchmark_nht_high.ps1
    .\benchmarks\nht\benchmark_nht_high.ps1 -Scenes garden,truck
    .\benchmarks\nht\benchmark_nht_high.ps1 -OutputRoot D:\runs\nht_high
    .\benchmarks\nht\benchmark_nht_high.ps1 -RuntimeOnly
    .\benchmarks\nht\benchmark_nht_high.ps1 -MetricsOnly -Step 29999
    .\benchmarks\nht\benchmark_nht_high.ps1 -SkipTandT -SkipDB
#>
param(
    [string]$Scenes      = "",
    [string]$DataRoot    = "",
    [string]$OutputRoot  = "",
    [int]$GPU            = 0,
    [int]$MaxSteps       = 30000,
    [int]$FeatureDim     = 64,
    [int]$Step           = -1,
    [int]$NumPasses      = 3,
    [int]$WarmupFrames   = 10,
    [ValidateSet("vgg","alex")]
    [string]$LpipsNet    = "vgg",
    [switch]$NoLpipsNormalize,
    [switch]$MetricsOnly,
    [switch]$RuntimeOnly,
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

if (-not $OutputRoot) { $OutputRoot = "$RepoRoot\results\benchmark_nht_high" }
else { $OutputRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot) }
$resultBase = $OutputRoot
$sceneNames = $jobs | ForEach-Object { $_[0] }
$allScenes  = $jobs | ForEach-Object { $_[0] }

# NHT hyper-parameters
$nhtArgs = @(
    "--deferred_opt_feature_dim", $FeatureDim,
    "--deferred_opt_view_encoding_type", "sh",
    "--deferred_mlp_ema",
    "--deferred_opt_center_ray_encoding"
)

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
# TRAIN + EVAL  (skipped with -MetricsOnly or -RuntimeOnly)
# ===================================================================
if (-not $MetricsOnly -and -not $RuntimeOnly) {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "NHT High-Count Benchmark (per-scene caps)" -ForegroundColor Cyan
    Write-Host "  Scenes:   $($sceneNames -join ', ')" -ForegroundColor Green
    Write-Host "  Features: $FeatureDim   Steps: $MaxSteps" -ForegroundColor Green
    Write-Host "  Results:  $resultBase" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Cyan

    foreach ($job in $jobs) {
        $scene = $job[0]; $dataDir = $job[1]; $factor = $job[2]; $dataset = $job[3]
        $capMax = if ($sceneCaps.ContainsKey($scene)) { $sceneCaps[$scene] } else { 1000000 }
        $resultDir = "$resultBase/$scene"

        Write-Host "`n>>> [$dataset] Training $scene (factor=$factor, cap=$capMax) <<<" -ForegroundColor Yellow

        $trainArgs = @(
            $Trainer, "default",
            "--eval_steps", "-1",
            "--disable_viewer",
            "--data_dir", $dataDir,
            "--data_factor", $factor,
            "--result_dir", $resultDir,
            "--max_steps", $MaxSteps,
            "--strategy.cap-max", $capMax,
            "--render_traj_path", "ellipse"
        ) + $lpipsArgs + $nhtArgs
        $env:CUDA_VISIBLE_DEVICES = $GPU
        python @trainArgs

        $ckptDir = "$resultDir/ckpts"
        if (Test-Path $ckptDir) {
            foreach ($ckpt in (Get-ChildItem "$ckptDir/*.pt" | Sort-Object Name)) {
                Write-Host "  Evaluating $($ckpt.Name) ..." -ForegroundColor Cyan
                $evalArgs = @(
                    $Trainer, "default",
                    "--disable_viewer",
                    "--data_dir", $dataDir,
                    "--data_factor", $factor,
                    "--result_dir", $resultDir,
                    "--max_steps", $MaxSteps,
                    "--strategy.cap-max", $capMax,
                    "--render_traj_path", "ellipse",
                    "--ckpt", $ckpt.FullName
                ) + $lpipsArgs + $nhtArgs
                python @evalArgs
            }
        }
    }
}

# ===================================================================
# RUNTIME  (skipped with -MetricsOnly)
# ===================================================================
if (-not $MetricsOnly) {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "Running NHT Timing Benchmark" -ForegroundColor Cyan
    Write-Host "  Scenes:  $($sceneNames -join ', ')" -ForegroundColor Green
    Write-Host "  Passes:  $NumPasses (warmup=$WarmupFrames frames)" -ForegroundColor Green
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

    # Collect timing.json
    $timingKeys = @("rasterization_ms", "deferred_mlp_ms", "overhead_ms", "total_ms", "fps", "fps_raster_mlp")
    $timingMetrics = @{}

    foreach ($scene in $sceneNames) {
        $timingFile = "$resultBase/$scene/stats/timing.json"
        $dsLabel = if ($m360Indoor -contains $scene) { "m360-in" }
                   elseif ($m360Outdoor -contains $scene) { "m360-out" }
                   elseif ($tandtScenes -contains $scene) { "tandt" }
                   elseif ($dbScenes -contains $scene) { "db" } else { "?" }

        Write-Host "`n  [$dsLabel] $scene" -ForegroundColor Magenta
        if (-not (Test-Path $timingFile)) {
            Write-Host "    ** NO timing.json **" -ForegroundColor Red
            continue
        }
        $json = Get-Content $timingFile -Raw | ConvertFrom-Json
        $rPct = if ($json.total_ms -gt 0) { $json.rasterization_ms / $json.total_ms * 100 } else { 0 }
        $mPct = if ($json.total_ms -gt 0) { $json.deferred_mlp_ms / $json.total_ms * 100 } else { 0 }
        Write-Host ("    Raster={0:F2}ms ({1:F1}%)  MLP={2:F2}ms ({3:F1}%)  Total={4:F2}ms  FPS={5:F1}  #GS={6}" -f `
            $json.rasterization_ms, $rPct, $json.deferred_mlp_ms, $mPct,
            $json.total_ms, $json.fps, [int]$json.num_gs) -ForegroundColor Green
        $timingMetrics[$scene] = @{}
        foreach ($k in ($timingKeys + @("num_gs"))) {
            if ($null -ne $json.$k) { $timingMetrics[$scene][$k] = [double]$json.$k }
        }
    }

    if ($timingMetrics.Count -gt 0) {
        $tRows = @()
        $tRows += Get-AggregatedRow "M360-In"  $m360Indoor   $timingMetrics ($timingKeys + @("num_gs"))
        $tRows += Get-AggregatedRow "M360-Out" $m360Outdoor   $timingMetrics ($timingKeys + @("num_gs"))
        $tRows += Get-AggregatedRow "M360"     ($m360Indoor + $m360Outdoor) $timingMetrics ($timingKeys + @("num_gs"))
        $tRows += Get-AggregatedRow "T&T"      $tandtScenes   $timingMetrics ($timingKeys + @("num_gs"))
        $tRows += Get-AggregatedRow "DB"       $dbScenes      $timingMetrics ($timingKeys + @("num_gs"))
        $tRows += Get-AggregatedRow "Overall"  $allScenes     $timingMetrics ($timingKeys + @("num_gs"))

        foreach ($r in $tRows) {
            if ($null -eq $r) { continue }
            if ($r["total_ms"] -and $r["total_ms"] -gt 0) {
                $r["fps"] = 1000.0 / $r["total_ms"]
            }
            $rmMs = 0.0
            if ($r["rasterization_ms"]) { $rmMs += $r["rasterization_ms"] }
            if ($r["deferred_mlp_ms"])  { $rmMs += $r["deferred_mlp_ms"] }
            if ($rmMs -gt 0) { $r["fps_raster_mlp"] = 1000.0 / $rmMs }
        }

        Write-Host "`n  Aggregated Timing:" -ForegroundColor Cyan
        $hdr = "{0,-10} {1,3}  {2,11}  {3,9}  {4,10}  {5,7}  {6,10}" -f `
            "Split", "N", "Raster(ms)", "MLP(ms)", "Total(ms)", "FPS", "Avg #GS"
        Write-Host $hdr -ForegroundColor White
        Write-Host ("-" * $hdr.Length) -ForegroundColor DarkGray
        foreach ($r in $tRows) {
            if ($null -eq $r) { continue }
            $numGS = if ($null -ne $r["num_gs"]) { "{0,10:N0}" -f [int]$r["num_gs"] } else { "       N/A" }
            $line = "{0,-10} {1,3}  {2,11:F2}  {3,9:F2}  {4,10:F2}  {5,7:F1}  {6}" -f `
                $r.Label, $r.N, $r["rasterization_ms"], $r["deferred_mlp_ms"],
                $r["total_ms"], $r["fps"], $numGS
            $c = if ($aggColors.ContainsKey($r.Label)) { $aggColors[$r.Label] } else { "White" }
            Write-Host $line -ForegroundColor $c
        }
    }
}

# ===================================================================
# METRICS  (skipped with -RuntimeOnly)
# ===================================================================
if (-not $RuntimeOnly) {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "Results Summary  (LPIPS = VGG, normalized inputs)" -ForegroundColor Cyan
    Write-Host "  Source: $resultBase" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Cyan

    $allMetrics = @{}
    $metricKeys = @("psnr", "ssim", "lpips", "num_GS")
    $missing    = @()

    foreach ($job in $jobs) {
        $scene = $job[0]; $dataset = $job[3]
        $statsDir = "$resultBase/$scene/stats"

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
        Write-Host ("    $($valFile.Name):  PSNR={0:F3}  SSIM={1:F4}  LPIPS(VGG)={2:F3}  #GS={3}" -f `
            $json.psnr, $json.ssim, $json.lpips, [int]$json.num_GS) -ForegroundColor Green
        $allMetrics[$scene] = @{}
        foreach ($k in $metricKeys) {
            if ($null -ne $json.$k) { $allMetrics[$scene][$k] = [double]$json.$k }
        }
    }

    if (-not $MetricsOnly -and $allMetrics.Count -gt 0) {
        $rows = @()
        $rows += Get-AggregatedRow "M360-In"  $m360Indoor   $allMetrics $metricKeys
        $rows += Get-AggregatedRow "M360-Out" $m360Outdoor   $allMetrics $metricKeys
        $rows += Get-AggregatedRow "M360"     ($m360Indoor + $m360Outdoor) $allMetrics $metricKeys
        $rows += Get-AggregatedRow "T&T"      $tandtScenes   $allMetrics $metricKeys
        $rows += Get-AggregatedRow "DB"       $dbScenes      $allMetrics $metricKeys
        $rows += Get-AggregatedRow "Overall"  $allScenes     $allMetrics $metricKeys

        Write-Host "`n============================================================" -ForegroundColor Cyan
        Write-Host "Aggregated Results" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan

        $header = "{0,-10} {1,4}  {2,8}  {3,8}  {4,11}  {5,10}" -f `
            "Split", "N", "PSNR", "SSIM", "LPIPS(VGG)", "#GS"
        Write-Host $header -ForegroundColor White
        Write-Host ("-" * $header.Length) -ForegroundColor DarkGray

        foreach ($r in $rows) {
            if ($null -eq $r) { continue }
            $numGS = if ($null -ne $r["num_GS"]) { "{0,10:N0}" -f [int]$r["num_GS"] } else { "       N/A" }
            $line = "{0,-10} {1,4}  {2,8:F3}  {3,8:F4}  {4,11:F3}  {5}" -f `
                $r.Label, $r.N, $r["psnr"], $r["ssim"], $r["lpips"], $numGS
            $c = if ($aggColors.ContainsKey($r.Label)) { $aggColors[$r.Label] } else { "White" }
            Write-Host $line -ForegroundColor $c
        }

        Write-Host "`n--- Markdown table ---" -ForegroundColor DarkGray
        Write-Host "| Split    |  N | PSNR  | SSIM   | LPIPS(VGG) | #GS     |"
        Write-Host "|----------|----|-------|--------|------------|---------|"
        foreach ($r in $rows) {
            if ($null -eq $r) { continue }
            $numGS = if ($null -ne $r["num_GS"]) { "{0:N0}" -f [int]$r["num_GS"] } else { "N/A" }
            Write-Host ("| {0,-8} | {1,2} | {2:F3} | {3:F4} | {4:F3}      | {5,-7} |" -f `
                $r.Label, $r.N, $r["psnr"], $r["ssim"], $r["lpips"], $numGS)
        }
    }

    if ($missing.Count -gt 0) {
        Write-Host "`nMISSING: $($missing -join ', ')" -ForegroundColor Red
        Write-Host "  Rerun: .\benchmarks\nht\benchmark_nht_high.ps1 -Scenes $($missing -join ',')" -ForegroundColor Yellow
    }
}
