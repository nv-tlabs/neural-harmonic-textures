#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Setup script for NHT paper release.
# Creates conda env "nht", initializes the gsplat submodule, and installs dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ensure_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" ]] && [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "  Using CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  if [[ -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
    echo "  Set CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  local best=""
  for d in /usr/local/cuda-*; do
    if [[ -d "$d" && -x "$d/bin/nvcc" ]]; then
      best="$d"
    fi
  done
  if [[ -n "$best" ]]; then
    export CUDA_HOME="$best"
    echo "  Set CUDA_HOME=${CUDA_HOME}"
    return 0
  fi
  return 1
}

get_pytorch_wheel_index() {
  if [[ -n "${PYTORCH_CUDA_INDEX:-}" ]]; then
    if [[ "${PYTORCH_CUDA_INDEX}" == http* ]]; then
      echo "${PYTORCH_CUDA_INDEX}"
    else
      echo "https://download.pytorch.org/whl/${PYTORCH_CUDA_INDEX}"
    fi
    return
  fi
  if [[ -n "${CUDA_HOME:-}" && -f "${CUDA_HOME}/version.txt" ]]; then
    local line
    line="$(head -1 "${CUDA_HOME}/version.txt")"
    if [[ "$line" =~ ([0-9]+)\.([0-9]+) ]]; then
      echo "https://download.pytorch.org/whl/cu${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
      return
    fi
  fi
  echo "https://download.pytorch.org/whl/cu126"
}

echo "============================================"
echo "NHT Release Setup"
echo "============================================"

echo "[1/5] Ensuring conda environment 'nht' exists..."
if ! conda env list | grep -qE '^\s*nht\s'; then
  conda create -n nht python=3.11 -y
else
  echo "  Conda env 'nht' already exists; skipping create."
fi

echo "[2/5] Initializing gsplat submodule..."
git submodule update --init --recursive --remote

echo "[3/5] CUDA + PyTorch (CUDA wheels)..."
if ! ensure_cuda_home; then
  echo "  ERROR: CUDA toolkit not found and CUDA_HOME is not set." >&2
  echo "  Install CUDA 12.x (with nvcc) or set CUDA_HOME, then re-run setup." >&2
  echo "  Optional: export PYTORCH_CUDA_INDEX (e.g. cu126, cu128) to match your stack." >&2
  exit 1
fi
WHEEL_URL="$(get_pytorch_wheel_index)"
echo "  PyTorch wheel index: ${WHEEL_URL}"
conda run -n nht pip install -U pip
conda run -n nht pip install "setuptools>=42" wheel ninja numpy rich
conda run -n nht pip install torch==2.9.1 torchvision==0.24.1 --index-url "${WHEEL_URL}"

echo "[4/5] Installing gsplat..."
conda run -n nht pip install --no-build-isolation -e ./gsplat

echo "[4b/5] Installing 'aov' package (AOV helpers)..."
conda run -n nht pip install --no-build-isolation -e .

echo "[5/5] Installing example dependencies..."
conda run -n nht pip install --no-build-isolation -r gsplat/examples/requirements.txt

echo ""
echo "Setup complete. Activate the environment, then run:"
echo "  conda activate nht"
echo "  bash scripts/train.sh                         # Train a scene"
echo "  bash scripts/view.sh --ckpt <path>            # View a trained model"
echo "  bash benchmarks/nht/benchmark_XXX.sh          # Reproduce paper results"
