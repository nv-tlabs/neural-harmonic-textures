# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Setup script for NHT paper release.
# Initializes the gsplat submodule and installs all dependencies.

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "NHT Release Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

Write-Host "[1/3] Initializing gsplat submodule..." -ForegroundColor Green
git submodule update --init --recursive

Write-Host "[2/3] Installing gsplat (editable)..." -ForegroundColor Green
pip install -e ./gsplat

Write-Host "[3/3] Installing additional dependencies..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete. You can now run:" -ForegroundColor Green
Write-Host "  .\scripts\train.ps1                              # Train a scene"
Write-Host "  .\scripts\view.ps1 -Ckpt <path>                  # View a trained model"
Write-Host "  .\benchmarks\nht\benchmark_nht.ps1                # Reproduce Table 2"

Pop-Location
