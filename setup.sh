#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Setup script for NHT paper release.
# Creates a .venv with uv, initializes the gsplat submodule, and installs dependencies.

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
  local cuda_ver=""
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    cuda_ver="$("${CUDA_HOME}/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')" || true
  elif command -v nvcc &>/dev/null; then
    cuda_ver="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')" || true
  fi
  if [[ -z "$cuda_ver" && -n "${CUDA_HOME:-}" ]]; then
    if [[ -f "${CUDA_HOME}/version.json" ]]; then
      cuda_ver="$(python -c "import json,pathlib; d=json.loads(pathlib.Path('${CUDA_HOME}/version.json').read_text()); print(d['cuda']['version'])" 2>/dev/null)" || true
    elif [[ -f "${CUDA_HOME}/version.txt" ]]; then
      local line; line="$(head -1 "${CUDA_HOME}/version.txt")"
      [[ "$line" =~ ([0-9]+\.[0-9]+) ]] && cuda_ver="${BASH_REMATCH[1]}"
    fi
  fi
  if [[ "$cuda_ver" =~ ^([0-9]+)\.([0-9]+) ]]; then
    echo "https://download.pytorch.org/whl/cu${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    return
  fi
  echo "https://download.pytorch.org/whl/cu126"
}

echo "============================================"
echo "NHT Release Setup"
echo "============================================"

# Check that uv is available
if ! command -v uv &>/dev/null; then
  echo "ERROR: 'uv' is not installed." >&2
  echo "  Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "[1/5] Creating virtual environment (.venv, Python 3.11)..."
uv venv --python 3.11 .venv
source .venv/bin/activate

export CC="$(which gcc)"
export CXX="$(which g++)"

echo "[2/5] Initializing gsplat submodule..."
git submodule update --init --recursive

echo "[3/5] CUDA + PyTorch (CUDA wheels)..."
if ! ensure_cuda_home; then
  echo "  ERROR: CUDA toolkit not found and CUDA_HOME is not set." >&2
  echo "  Install CUDA 12.x (with nvcc) or set CUDA_HOME, then re-run setup." >&2
  echo "  Optional: export PYTORCH_CUDA_INDEX (e.g. cu126, cu128) to match your stack." >&2
  exit 1
fi
WHEEL_URL="$(get_pytorch_wheel_index)"
echo "  PyTorch wheel index: ${WHEEL_URL}"
uv pip install "setuptools==78.1.1" wheel ninja numpy rich
uv pip install torch==2.9.1 torchvision==0.24.1 --index-url "${WHEEL_URL}"

export TORCH_CUDA_ARCH_LIST=$(python -c "import torch,re;print(';'.join(re.sub(r'sm_(\d+)(\d)([a-z]?)$',lambda m:m[1]+'.'+m[2]+m[3],s) for s in torch.cuda.get_arch_list()))")+PTX

echo "[4/5] Installing gsplat..."
uv pip install --no-build-isolation -e ./gsplat

echo "[4b/5] Installing 'aov' package (AOV helpers)..."
uv pip install --no-build-isolation -e .

echo "[5/5] Installing example dependencies..."
uv pip install --no-build-isolation -r gsplat/examples/requirements.txt

# Persist environment variables into the venv activation script so they are
# restored automatically on `source .venv/bin/activate`.
ACTIVATE=".venv/bin/activate"
if ! grep -q "# --- NHT env vars ---" "$ACTIVATE" 2>/dev/null; then
  cat >> "$ACTIVATE" <<ENVEOF

# --- NHT env vars ---
export CC="${CC}"
export CXX="${CXX}"
export CUDA_HOME="${CUDA_HOME}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
ENVEOF
  echo "  Persisted CC, CXX, CUDA_HOME, TORCH_CUDA_ARCH_LIST in ${ACTIVATE}"
fi

echo ""
echo "Setup complete. Activate the environment, then run:"
echo "  source .venv/bin/activate"
echo "  bash scripts/train.sh                         # Train a scene"
echo "  bash scripts/view.sh --ckpt <path>            # View a trained model"
echo "  bash benchmarks/nht/benchmark_XXX.sh          # Reproduce paper results"
