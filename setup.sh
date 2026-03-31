#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Setup script for NHT paper release.
# Initializes the gsplat submodule and installs all dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "NHT Release Setup"
echo "============================================"

echo "[1/3] Initializing gsplat submodule..."
git submodule update --init --recursive

echo "[2/3] Installing gsplat (editable)..."
pip install -e ./gsplat

echo "[3/3] Installing additional dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete. You can now run:"
echo "  bash scripts/train.sh                         # Train a scene"
echo "  bash scripts/view.sh --ckpt <path>            # View a trained model"
echo "  bash benchmarks/nht/benchmark_nht.sh          # Reproduce Table 2"
