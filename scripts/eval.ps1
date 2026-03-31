<#
.SYNOPSIS
    Evaluate a trained NHT checkpoint: quality metrics, trajectory video,
    and runtime timing via benchmark_nht.py.

.EXAMPLE
    .\scripts\eval.ps1 -Ckpt results\benchmark_nht\garden\ckpts\ckpt_29999_rank0.pt
    .\scripts\eval.ps1 -CkptDir results\benchmark_nht\garden\ckpts
    .\scripts\eval.ps1 -Ckpt results\benchmark_nht\garden\ckpts\ckpt_29999_rank0.pt -SkipRuntime
#>
param(
    [string]$Ckpt        = "",
    [string]$CkptDir     = "",
    [string]$Scene       = "garden",
    [string]$SceneDir    = "",
    [int]$DataFactor     = 4,
    [string]$ResultDir   = "",
    [int]$CapMax         = 1000000,
    [int]$GPU            = 0,
    [int]$NumPasses      = 3,
    [int]$WarmupFrames   = 10,
    [switch]$SkipRuntime
)

$ErrorActionPreference = "Stop"

$RepoRoot    = (Resolve-Path "$PSScriptRoot\..").Path
$Trainer     = "$RepoRoot\gsplat\examples\simple_trainer_nht.py"
$Benchmarker = "$RepoRoot\benchmarks\benchmark_nht.py"

if (-not $SceneDir) { $SceneDir = "$RepoRoot\data\mipnerf360" }

if ($Ckpt -and (Test-Path $Ckpt)) {
    $Ckpt = (Resolve-Path $Ckpt).Path
}
if ($CkptDir -and (Test-Path $CkptDir)) {
    $CkptDir = (Resolve-Path $CkptDir).Path
}

if (-not $ResultDir) {
    $ResultDir = "$RepoRoot\results\nht_mcmc_${CapMax}\${Scene}"
}

$ckpts = @()
if ($Ckpt) {
    $ckpts += $Ckpt
} elseif ($CkptDir) {
    $ckpts = Get-ChildItem -Path $CkptDir -Filter "*.pt" | Sort-Object Name | ForEach-Object { $_.FullName }
} else {
    $defaultCkptDir = "$ResultDir\ckpts"
    if (Test-Path $defaultCkptDir) {
        $ckpts = Get-ChildItem -Path $defaultCkptDir -Filter "*.pt" | Sort-Object Name | ForEach-Object { $_.FullName }
    }
}

if ($ckpts.Count -eq 0) {
    Write-Error "No checkpoints found. Specify -Ckpt or -CkptDir."
    return
}

foreach ($ckptFile in $ckpts) {
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "Evaluating: $ckptFile" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Cyan

    $args_list = @(
        $Trainer, "default",
        "--disable_viewer",
        "--data_dir", "$SceneDir/$Scene",
        "--data_factor", $DataFactor,
        "--result_dir", $ResultDir,
        "--strategy.cap-max", $CapMax,
        "--render_traj_path", "ellipse",
        "--ckpt", $ckptFile
    )

    $env:CUDA_VISIBLE_DEVICES = $GPU
    python @args_list
}

$statsDir = "$ResultDir\stats"
if (Test-Path $statsDir) {
    Write-Host ""
    Write-Host "=== Quality Results ===" -ForegroundColor Yellow
    Get-ChildItem -Path $statsDir -Filter "val_step*.json" `
        | Where-Object { $_.Name -notmatch "per_image" -and $_.Name -match "step(\d+)" } `
        | Sort-Object { [int]($_.Name -replace '.*step(\d+).*','$1') } -Descending `
        | Select-Object -First 1 | ForEach-Object {
        $json = Get-Content $_.FullName -Raw | ConvertFrom-Json
        Write-Host ("  $($_.Name):  PSNR={0:F3}  SSIM={1:F4}  LPIPS={2:F3}  #GS={3}" -f `
            $json.psnr, $json.ssim, $json.lpips, [int]$json.num_GS) -ForegroundColor Green
    }
}

if (-not $SkipRuntime) {
    $latestCkpt = $ckpts[-1]
    $timingJson = "$ResultDir\stats\timing.json"

    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "Runtime Benchmark" -ForegroundColor Cyan
    Write-Host "  Checkpoint: $(Split-Path $latestCkpt -Leaf)" -ForegroundColor Green
    Write-Host "  Passes:     $NumPasses (warmup=$WarmupFrames frames)" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Cyan

    $benchArgs = @(
        $Benchmarker,
        "--ckpt", $latestCkpt,
        "--data_dir", "$SceneDir/$Scene",
        "--data_factor", $DataFactor,
        "--num_passes", $NumPasses,
        "--warmup_frames", $WarmupFrames,
        "--save_json", $timingJson,
        "--scene_name", $Scene
    )

    $env:CUDA_VISIBLE_DEVICES = $GPU
    python @benchArgs

    if (Test-Path $timingJson) {
        $t = Get-Content $timingJson -Raw | ConvertFrom-Json
        $rPct = if ($t.total_ms -gt 0) { $t.rasterization_ms / $t.total_ms * 100 } else { 0 }
        $mPct = if ($t.total_ms -gt 0) { $t.deferred_mlp_ms / $t.total_ms * 100 } else { 0 }
        Write-Host ""
        Write-Host "=== Timing Summary ===" -ForegroundColor Yellow
        Write-Host ("  Raster={0:F2}ms ({1:F1}%)  MLP={2:F2}ms ({3:F1}%)  Total={4:F2}ms  FPS={5:F1}  #GS={6}" -f `
            $t.rasterization_ms, $rPct, $t.deferred_mlp_ms, $mPct,
            $t.total_ms, $t.fps, [int]$t.num_gs) -ForegroundColor Green
    }
}
