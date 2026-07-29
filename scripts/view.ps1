<#
.SYNOPSIS
    Launch the interactive NHT viewer for a trained checkpoint.

.DESCRIPTION
    The viewer renders with the fully-fused rasterize+MLP kernel by default,
    which supports RGB and alpha through a pinhole camera only. Pass -NoFused
    for depth, normals, non-pinhole cameras, antialiasing or radius clipping.

.EXAMPLE
    .\scripts\view.ps1 -Ckpt results\benchmark_nht\garden\ckpts\ckpt_29999_rank0.pt
    .\scripts\view.ps1 -Ckpt results\benchmark_nht\garden\ckpts\ckpt_29999_rank0.pt -Port 8082
    .\scripts\view.ps1 -Ckpt results\benchmark_nht\garden\ckpts\ckpt_29999_rank0.pt -NoFused
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$Ckpt,
    [string]$OutputDir   = "",
    [int]$Port           = 8080,
    [int]$GPU            = 0,
    [switch]$NoFused
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$Viewer   = "$RepoRoot\gsplat\examples\simple_viewer_nht.py"

if ($Ckpt -and (Test-Path $Ckpt)) {
    $Ckpt = (Resolve-Path $Ckpt).Path
}

if (-not $OutputDir) {
    $OutputDir = Split-Path (Split-Path $Ckpt -Parent) -Parent
}

$args_list = @(
    $Viewer,
    "--ckpt", $Ckpt,
    "--output_dir", $OutputDir,
    "--port", $Port
)
if ($NoFused) { $args_list += "--no_fused" }

if ($NoFused) { $KernelLabel = "two-stage (rasterize + tcnn)" }
else { $KernelLabel = "fused (RGB/alpha, pinhole)" }

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "NHT Viewer" -ForegroundColor Cyan
Write-Host "  Checkpoint: $Ckpt" -ForegroundColor Green
Write-Host "  Port:       $Port" -ForegroundColor Green
Write-Host "  Output:     $OutputDir" -ForegroundColor Green
Write-Host "  Kernel:     $KernelLabel" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Open http://localhost:$Port in your browser" -ForegroundColor Yellow

$env:CUDA_VISIBLE_DEVICES = $GPU
python @args_list
