#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download and extract the MipNeRF 360 dataset into data/mipnerf360/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_DIR="${1:-data/mipnerf360}"

URLS=(
  "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
  "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"
)

download_and_extract() {
  local url="$1"
  local zip_file
  zip_file="$(mktemp --suffix=.zip)"

  echo "Downloading ${url##*/}..."
  echo "  URL: ${url}"
  if command -v wget &>/dev/null; then
    wget -q --show-progress -O "$zip_file" "$url"
  elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "$zip_file" "$url"
  else
    echo "ERROR: Neither wget nor curl found." >&2
    exit 1
  fi

  echo "Extracting..."
  unzip -q -o "$zip_file" -d "$DATA_DIR"
  rm "$zip_file"
}

if [[ -d "$DATA_DIR" ]]; then
  echo "Dataset directory '$DATA_DIR' already exists — skipping download."
  echo "  Delete it and re-run to force a fresh download."
  exit 0
fi

mkdir -p "$DATA_DIR"

for url in "${URLS[@]}"; do
  download_and_extract "$url"
done

echo "Done. Dataset extracted to ${DATA_DIR}/"
