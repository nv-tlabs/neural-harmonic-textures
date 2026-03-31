<#
.SYNOPSIS
    Launch the interactive NHT viewer for a trained checkpoint.

.EXAMPLE
    .\scripts\view.ps1 -Ckpt results\benchmark_nht\garden\ckpts\ckpt_29999_rank0.pt
    .\scripts\view.ps1 -Ckpt results\benchmark_nht\garden\ckpts\ckpt_29999_rank0.pt -Port 8082
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$Ckpt,
    [string]$OutputDir   = "",
    [int]$Port           = 8080,
    [int]$GPU            = 0
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

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "NHT Viewer" -ForegroundColor Cyan
Write-Host "  Checkpoint: $Ckpt" -ForegroundColor Green
Write-Host "  Port:       $Port" -ForegroundColor Green
Write-Host "  Output:     $OutputDir" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Open http://localhost:$Port in your browser" -ForegroundColor Yellow

$env:CUDA_VISIBLE_DEVICES = $GPU
python @args_list
