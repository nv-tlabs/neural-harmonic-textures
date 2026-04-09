#!/usr/bin/env bash
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

# Launch the interactive NHT viewer for a trained checkpoint.
#
# Usage:
#   ./scripts/view.sh --ckpt results/benchmark_nht/garden/ckpts/ckpt_29999_rank0.pt
#   ./scripts/view.sh --ckpt results/benchmark_nht/garden/ckpts/ckpt_29999_rank0.pt --port 8082

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VIEWER="$REPO_ROOT/gsplat/examples/simple_viewer_nht.py"

CKPT=""
OUTPUT_DIR=""
PORT=8080
GPU=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)       CKPT="$2";       shift 2 ;;
        --output_dir) OUTPUT_DIR="$2";  shift 2 ;;
        --port)       PORT="$2";        shift 2 ;;
        --gpu)        GPU="$2";         shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CKPT" ]]; then
    echo "Error: --ckpt is required." >&2
    exit 1
fi

if [[ -f "$CKPT" ]]; then
    CKPT="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$(dirname "$(dirname "$CKPT")")"
fi

args=(
    "$VIEWER"
    --ckpt "$CKPT"
    --output_dir "$OUTPUT_DIR"
    --port "$PORT"
)

echo "============================================"
echo "NHT Viewer"
echo "  Checkpoint: $CKPT"
echo "  Port:       $PORT"
echo "  Output:     $OUTPUT_DIR"
echo "============================================"
echo "Open http://localhost:$PORT in your browser"

CUDA_VISIBLE_DEVICES="$GPU" python "${args[@]}"
