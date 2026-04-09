# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download and extract the MipNeRF 360 dataset into data\mipnerf360\.

$ErrorActionPreference = "Stop"
Push-Location (Split-Path $PSScriptRoot -Parent)

$DataDir = if ($args.Count -ge 1) { $args[0] } else { "data\mipnerf360" }

$Urls = @(
    "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
    "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"
)

if (Test-Path $DataDir) {
    Write-Host "Dataset directory '$DataDir' already exists - skipping download." -ForegroundColor DarkGray
    Write-Host "  Delete it and re-run to force a fresh download."
    Pop-Location
    exit 0
}

New-Item -ItemType Directory -Path $DataDir -Force | Out-Null

foreach ($url in $Urls) {
    $fileName = $url.Split("/")[-1]
    $zipFile = Join-Path $env:TEMP "nht-$fileName"

    Write-Host "Downloading $fileName..." -ForegroundColor Green
    Write-Host "  URL: $url"
    Invoke-WebRequest -Uri $url -OutFile $zipFile -UseBasicParsing

    Write-Host "Extracting..."
    Expand-Archive -Path $zipFile -DestinationPath $DataDir -Force
    Remove-Item $zipFile
}

Write-Host ""
Write-Host "Done. Dataset extracted to $DataDir\" -ForegroundColor Green

Pop-Location
