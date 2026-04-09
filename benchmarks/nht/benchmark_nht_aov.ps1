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
    NHT MCMC AOV benchmark (LSEG, DINOv3, RGB2X) across Mip-NeRF 360 scenes.
    LPIPS is evaluated with VGG (normalize=True). This differs from INRIA 3DGS
    which uses VGG with normalize=False (technically incorrect).

.DESCRIPTION
    Trains and evaluates with auxiliary output targets, then collects quality metrics.

    AOV flag mapping:
      lseg   -> --lseg_data
      dinov3 -> --dinov3_data
      rgb2x  -> --rgb2x_data

    Result layout: <OutputRoot>\<target>\<scene> (default OutputRoot: <repo>\results\benchmark_nht_aov)

    Modes:
      (default)      Train + eval + collect metrics
      -MetricsOnly   Skip training; collect and display metrics from existing results

    -DataRoot defaults to <repo>\data when empty.

.EXAMPLE
    .\benchmarks\nht\benchmark_nht_aov.ps1
    .\benchmarks\nht\benchmark_nht_aov.ps1 -Scenes garden,bonsai -AOVTargets dinov3
    .\benchmarks\nht\benchmark_nht_aov.ps1 -OutputRoot D:\runs\nht_aov
    .\benchmarks\nht\benchmark_nht_aov.ps1 -MetricsOnly
#>
param(
    [string]$Scenes      = "",
    [string]$DataRoot    = "",
    [string]$OutputRoot  = "",
    [string]$AOVTargets  = "lseg",
    [int]$CapMax         = 1000000,
    [int]$GPU            = 0,
    [int]$MaxSteps       = 30000,
    [int]$NumWorkers     = 0,
    [ValidateSet("vgg","alex")]
    [string]$LpipsNet    = "vgg",
    [switch]$NoLpipsNormalize,
    [switch]$MetricsOnly
)

$ErrorActionPreference = "Stop"
$lpipsArgs = @("--lpips_net", $LpipsNet)
if ($NoLpipsNormalize) { $lpipsArgs += "--no-lpips_normalize" }

$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$Trainer  = "$RepoRoot\aov\examples\simple_trainer_nht_aov.py"

if (-not $DataRoot) { $DataRoot = "$RepoRoot\data" }

$m360Indoor   = @("bonsai", "counter", "kitchen", "room")
$m360Outdoor  = @("garden", "bicycle", "stump", "treehill", "flowers")

if ($Scenes) {
    $sceneList = @($Scenes -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
} else {
    $sceneList = $m360Indoor + $m360Outdoor
}

$targetList = @($AOVTargets -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if (-not $OutputRoot) { $OutputRoot = "$RepoRoot\results\benchmark_nht_aov" }
else { $OutputRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot) }
$resultBase = $OutputRoot

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

# ===================================================================
# TRAIN + EVAL  (skipped with -MetricsOnly)
# ===================================================================
if (-not $MetricsOnly) {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "NHT MCMC AOV Benchmark" -ForegroundColor Cyan
    Write-Host "  Scenes:      $($sceneList -join ', ')" -ForegroundColor Green
    Write-Host "  AOV targets: $($targetList -join ', ')" -ForegroundColor Green
    Write-Host "  CapMax:      $CapMax   MaxSteps: $MaxSteps" -ForegroundColor Green
    Write-Host "  Results:     $resultBase" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Cyan

    foreach ($target in $targetList) {
        Write-Host "`n**** AOV TARGET: $target ****" -ForegroundColor Magenta

        foreach ($scene in $sceneList) {
            $dataFactor = if ($m360Indoor -contains $scene) { 2 } else { 4 }
            $dataDir = "$DataRoot/mipnerf360/$scene"
            $resultDir = "$resultBase/${target}/$scene"

            Write-Host "`n>>> Training $scene (factor=$dataFactor, target=$target) <<<" -ForegroundColor Yellow

            $aovFlags = @()
            switch ($target) {
                "lseg"   { $aovFlags += @("--lseg_data") }
                "dinov3" { $aovFlags += @("--dinov3_data") }
                "rgb2x"  { $aovFlags += @("--rgb2x_data") }
            }
            $trainArgs = @(
                $Trainer, "default",
                "--disable_viewer",
                "--data_dir", $dataDir,
                "--data_factor", $dataFactor,
                "--result_dir", $resultDir,
                "--max_steps", $MaxSteps,
                "--strategy.cap-max", $CapMax,
                "--render_traj_path", "ellipse",
                "--num_workers", $NumWorkers
            ) + $lpipsArgs + $aovFlags

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
                        "--data_factor", $dataFactor,
                        "--result_dir", $resultDir,
                        "--max_steps", $MaxSteps,
                        "--strategy.cap-max", $CapMax,
                        "--render_traj_path", "ellipse",
                        "--num_workers", $NumWorkers
                    ) + $lpipsArgs + $aovFlags + @("--ckpt", $ckpt.FullName)
                    python @evalArgs
                }
            }
        }
    }
}

# ===================================================================
# COLLECT METRICS
# ===================================================================
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Results Summary" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$metricKeys = @("psnr", "ssim", "lpips", "num_GS")

foreach ($target in $targetList) {
    Write-Host "`n--- AOV target: $target ---" -ForegroundColor Magenta
    $allMetrics = @{}

    foreach ($scene in $sceneList) {
        $statsDir = "$resultBase/${target}/$scene/stats"
        if (-not (Test-Path $statsDir)) { continue }

        Write-Host "`n  $scene" -ForegroundColor Yellow
        # Pick latest checkpoint by step in filename (e.g. val_step29999.json). String sort
        # would wrongly prefer val_step6999 over val_step29999 because '6' > '2'.
        $valFile = $null
        $bestStep = -1
        Get-ChildItem "$statsDir/val*.json" -ErrorAction SilentlyContinue `
            | Where-Object { $_.Name -notmatch "per_image" } `
            | ForEach-Object {
                if ($_.Name -match '_step(\d+)\.json$') {
                    $s = [int]$Matches[1]
                    if ($s -gt $bestStep) { $bestStep = $s; $valFile = $_ }
                }
            }

        if ($valFile) {
            $json = Get-Content $valFile.FullName -Raw | ConvertFrom-Json
            Write-Host ("    PSNR={0:F3}  SSIM={1:F4}  LPIPS={2:F3}  #GS={3}" -f `
                $json.psnr, $json.ssim, $json.lpips, [int]$json.num_GS) -ForegroundColor Green

            $allMetrics[$scene] = @{}
            foreach ($k in $metricKeys) {
                if ($null -ne $json.$k) { $allMetrics[$scene][$k] = [double]$json.$k }
            }

            $aovKeys = $json.PSObject.Properties.Name | Where-Object {
                $_ -match "^(lseg|dinov3|rgb2x)" -and $_ -notmatch "per_image"
            }
            if ($aovKeys) {
                $parts = $aovKeys | ForEach-Object { "{0}={1:F4}" -f $_, $json.$_ }
                Write-Host "    AOV: $($parts -join '  ')" -ForegroundColor Green
            }
        }
    }

    if ($allMetrics.Count -gt 0) {
        $outdoorScenes = $sceneList | Where-Object { $m360Indoor -notcontains $_ }
        $rows = @()
        $rows += Get-AggregatedRow "Indoor"  $m360Indoor   $allMetrics $metricKeys
        $rows += Get-AggregatedRow "Outdoor" $outdoorScenes $allMetrics $metricKeys
        $rows += Get-AggregatedRow "Overall" $sceneList      $allMetrics $metricKeys

        Write-Host "`n  Aggregated ($target):" -ForegroundColor Cyan
        $header = "    {0,-10} {1,4}  {2,8}  {3,8}  {4,11}  {5,10}" -f `
            "Split", "N", "PSNR", "SSIM", "LPIPS(VGG)", "#GS"
        Write-Host $header -ForegroundColor White
        foreach ($r in $rows) {
            if ($null -eq $r) { continue }
            $numGS = if ($null -ne $r["num_GS"]) { "{0,10:N0}" -f [int]$r["num_GS"] } else { "       N/A" }
            Write-Host ("    {0,-10} {1,4}  {2,8:F3}  {3,8:F4}  {4,11:F3}  {5}" -f `
                $r.Label, $r.N, $r["psnr"], $r["ssim"], $r["lpips"], $numGS) -ForegroundColor Cyan
        }
    }
}
