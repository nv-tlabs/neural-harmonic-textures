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
    Train an NHT model on a single scene.

.EXAMPLE
    .\scripts\train.ps1
    .\scripts\train.ps1 -Scene kitchen -DataFactor 2
#>
param(
    [string]$Scene       = "garden",
    [string]$SceneDir    = "",
    [int]$DataFactor     = 4,
    [int]$CapMax         = 1000000,
    [string]$ResultDir   = "",
    [int]$GPU            = 0,
    [int]$Port           = 8080,
    [switch]$DisableViewer
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$Trainer  = "$RepoRoot\gsplat\examples\simple_trainer_nht.py"

if (-not $SceneDir) { $SceneDir = "$RepoRoot\data\mipnerf360" }
if (-not $ResultDir) { $ResultDir = "$RepoRoot\results\nht_mcmc_${CapMax}\${Scene}" }

$args_list = @(
    $Trainer, "default",
    "--data_dir", "$SceneDir/$Scene",
    "--data_factor", $DataFactor,
    "--result_dir", $ResultDir,
    "--strategy.cap-max", $CapMax,
    "--port", $Port,
    "--render_traj_path", "ellipse"
)

if ($DisableViewer) {
    $args_list += "--disable_viewer"
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "NHT Training" -ForegroundColor Cyan
Write-Host "  Scene:      $Scene" -ForegroundColor Green
Write-Host "  CapMax:     $CapMax" -ForegroundColor Green
Write-Host "  Result dir: $ResultDir" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan

$env:CUDA_VISIBLE_DEVICES = $GPU
python @args_list
