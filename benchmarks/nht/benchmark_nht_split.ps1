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
    NHT split-strategy benchmark (paper configuration). Reproduces Table 1 from NHT paper.
    LPIPS is evaluated with VGG (normalize=True). This differs from INRIA 3DGS
    which uses VGG with normalize=False (technically incorrect).

.DESCRIPTION
    Result layout: <OutputRoot>\<scene> (default OutputRoot: <repo>\results\benchmark_nht_split).
    -ResultBase is an alias for -OutputRoot (backward compatible).

.EXAMPLE
    .\benchmarks\nht\benchmark_nht_split.ps1
    .\benchmarks\nht\benchmark_nht_split.ps1 -Scenes garden,bonsai,truck
    .\benchmarks\nht\benchmark_nht_split.ps1 -OutputRoot D:\runs\nht_split
    .\benchmarks\nht\benchmark_nht_split.ps1 -MetricsOnly
#>
param(
    [string]$Scenes     = "",
    [string]$DataRoot   = "",
    [Alias("ResultBase")]
    [string]$OutputRoot = "",
    [int]$GPU           = 0,
    [int]$Step          = -1,
    [ValidateSet("vgg","alex")]
    [string]$LpipsNet   = "vgg",
    [switch]$NoLpipsNormalize,
    [switch]$MetricsOnly
)

$ErrorActionPreference = "Stop"
$lpipsArgs = @("--lpips_net", $LpipsNet)
if ($NoLpipsNormalize) { $lpipsArgs += "--no-lpips_normalize" }

$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$Trainer  = "$RepoRoot\gsplat\examples\simple_trainer_nht.py"

if (-not $DataRoot) { $DataRoot = "$RepoRoot\data" }
if (-not $OutputRoot) { $OutputRoot = "$RepoRoot\results\benchmark_nht_split" }
else { $OutputRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot) }
$ResultBase = $OutputRoot

$commonArgs = @(
    "--disable_viewer",
    "--render_traj_path", "ellipse",
    "--ssim_lambda", "0.1"
) + $lpipsArgs

$m360Indoor  = @("bonsai", "counter", "kitchen", "room")
$m360Outdoor = @("garden", "bicycle", "stump", "treehill", "flowers")
$tandtScenes = @("train", "truck")
$dbScenes    = @("drjohnson", "playroom")

$jobs = @()
foreach ($s in $m360Outdoor) {
    $jobs += ,@($s, "$DataRoot/mipnerf360/$s", 4, 5000000, 25000, @(), "m360-outdoor")
}
foreach ($s in $m360Indoor) {
    $jobs += ,@($s, "$DataRoot/mipnerf360/$s", 2, 2000000, 45000, @("--deferred_opt_center_ray_encoding"), "m360-indoor")
}
foreach ($s in $tandtScenes) {
    $jobs += ,@($s, "$DataRoot/tandt_db/tandt/$s", 1, 2500000, 40000, @("--deferred_opt_center_ray_encoding"), "tandt")
}
foreach ($s in $dbScenes) {
    $jobs += ,@($s, "$DataRoot/tandt_db/db/$s", 1, 2000000, 30000, @("--deferred_opt_center_ray_encoding"), "deepblend")
}

if ($Scenes) {
    $filter = ($Scenes -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $jobs = @($jobs | Where-Object { $filter -contains $_[0] })
}

$allScenes = $jobs | ForEach-Object { $_[0] }

function Get-AggregatedRow {
    param([string]$Label, [string[]]$SceneSubset, [hashtable]$AllMetrics, [string[]]$Keys)
    $valid = $SceneSubset | Where-Object { $AllMetrics.ContainsKey($_) }
    if (-not $valid -or $valid.Count -eq 0) { return $null }
    $row = [ordered]@{ Label = $Label; N = $valid.Count }
    foreach ($k in $Keys) {
        $vals = $valid | ForEach-Object { $AllMetrics[$_][$k] } | Where-Object { $null -ne $_ }
        if ($vals.Count -gt 0) { $row[$k] = ($vals | Measure-Object -Average).Average }
        else { $row[$k] = $null }
    }
    return $row
}

if (-not $MetricsOnly) {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "NHT Split-Strategy Benchmark (Paper Config)" -ForegroundColor Cyan
    Write-Host "  Scenes: $($allScenes -join ', ')" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Cyan

    foreach ($job in $jobs) {
        $scene = $job[0]; $dataDir = $job[1]; $factor = $job[2]
        $capMax = $job[3]; $maxSteps = $job[4]; $groupArgs = $job[5]; $group = $job[6]
        $resultDir = "$ResultBase\$scene"

        Write-Host "`n>>> [$group] $scene (factor=$factor, cap=$capMax, steps=$maxSteps) <<<" -ForegroundColor Yellow

        $trainArgs = @($Trainer, "default") + $commonArgs + @(
            "--eval_steps", "-1",
            "--data_factor", $factor,
            "--max_steps", $maxSteps,
            "--strategy.cap-max", $capMax,
            "--data_dir", $dataDir,
            "--result_dir", $resultDir
        ) + $groupArgs

        $env:CUDA_VISIBLE_DEVICES = $GPU
        python @trainArgs

        $ckptDir = "$resultDir\ckpts"
        if (Test-Path $ckptDir) {
            foreach ($ckpt in (Get-ChildItem "$ckptDir\*.pt" | Sort-Object Name)) {
                Write-Host "  Evaluating $($ckpt.Name) ..." -ForegroundColor Cyan
                $evalArgs = @($Trainer, "default") + $commonArgs + @(
                    "--data_factor", $factor,
                    "--max_steps", $maxSteps,
                    "--strategy.cap-max", $capMax,
                    "--data_dir", $dataDir,
                    "--result_dir", $resultDir,
                    "--ckpt", $ckpt.FullName
                ) + $groupArgs
                python @evalArgs
            }
        }
    }
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Results Summary" -ForegroundColor Cyan
Write-Host "  Source: $ResultBase" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

$allMetrics = @{}
$metricKeys = @("psnr", "ssim", "lpips", "num_GS")

foreach ($job in $jobs) {
    $scene = $job[0]; $group = $job[6]
    $statsDir = "$ResultBase\$scene\stats"

    Write-Host "`n  [$group] $scene" -ForegroundColor Magenta
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

if ($allMetrics.Count -gt 0) {
    $rows = @()
    $rows += Get-AggregatedRow "M360-In"  $m360Indoor   $allMetrics $metricKeys
    $rows += Get-AggregatedRow "M360-Out" $m360Outdoor   $allMetrics $metricKeys
    $rows += Get-AggregatedRow "M360"     ($m360Indoor + $m360Outdoor) $allMetrics $metricKeys
    $rows += Get-AggregatedRow "T&T"      $tandtScenes   $allMetrics $metricKeys
    $rows += Get-AggregatedRow "DB"       $dbScenes      $allMetrics $metricKeys
    $rows += Get-AggregatedRow "Overall"  $allScenes     $allMetrics $metricKeys

    Write-Host "`nAggregated:" -ForegroundColor Cyan
    foreach ($r in $rows) {
        if ($null -eq $r) { continue }
        $numGS = if ($null -ne $r["num_GS"]) { "{0,10:N0}" -f [int]$r["num_GS"] } else { "       N/A" }
        Write-Host ("{0,-10} N={1}  PSNR={2:F3}  SSIM={3:F4}  LPIPS={4:F3}  #GS={5}" -f `
            $r.Label, $r.N, $r["psnr"], $r["ssim"], $r["lpips"], $numGS) -ForegroundColor Cyan
    }
}
