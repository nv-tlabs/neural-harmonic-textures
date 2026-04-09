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

# Evaluate a trained NHT checkpoint: quality metrics, trajectory video,
# and runtime timing via benchmark_nht.py.
#
# Usage:
#   ./scripts/eval.sh --ckpt results/benchmark_nht/garden/ckpts/ckpt_29999_rank0.pt
#   ./scripts/eval.sh --ckpt_dir results/benchmark_nht/garden/ckpts
#   ./scripts/eval.sh --ckpt results/benchmark_nht/garden/ckpts/ckpt_29999_rank0.pt --skip_runtime

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAINER="$REPO_ROOT/gsplat/examples/simple_trainer_nht.py"
BENCHMARKER="$REPO_ROOT/benchmarks/benchmark_nht.py"

CKPT=""
CKPT_DIR=""
SCENE="garden"
SCENE_DIR="$REPO_ROOT/data/mipnerf360"
DATA_FACTOR=4
RESULT_DIR=""
CAP_MAX=1000000
GPU=0
NUM_PASSES=3
WARMUP_FRAMES=10
SKIP_RUNTIME=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)           CKPT="$2";           shift 2 ;;
        --ckpt_dir)       CKPT_DIR="$2";       shift 2 ;;
        --scene)          SCENE="$2";           shift 2 ;;
        --scene_dir)      SCENE_DIR="$2";       shift 2 ;;
        --data_factor)    DATA_FACTOR="$2";     shift 2 ;;
        --result_dir)     RESULT_DIR="$2";      shift 2 ;;
        --cap_max)        CAP_MAX="$2";         shift 2 ;;
        --gpu)            GPU="$2";             shift 2 ;;
        --num_passes)     NUM_PASSES="$2";      shift 2 ;;
        --warmup_frames)  WARMUP_FRAMES="$2";   shift 2 ;;
        --skip_runtime)   SKIP_RUNTIME=true;    shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "$CKPT" && -f "$CKPT" ]]; then
    CKPT="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"
fi
if [[ -n "$CKPT_DIR" && -d "$CKPT_DIR" ]]; then
    CKPT_DIR="$(cd "$CKPT_DIR" && pwd)"
fi

if [[ -z "$RESULT_DIR" ]]; then
    RESULT_DIR="$REPO_ROOT/results/nht_mcmc_${CAP_MAX}/${SCENE}"
fi

ckpts=()
if [[ -n "$CKPT" ]]; then
    ckpts+=("$CKPT")
elif [[ -n "$CKPT_DIR" ]]; then
    while IFS= read -r f; do
        ckpts+=("$f")
    done < <(find "$CKPT_DIR" -maxdepth 1 -name '*.pt' | sort)
else
    default_ckpt_dir="$RESULT_DIR/ckpts"
    if [[ -d "$default_ckpt_dir" ]]; then
        while IFS= read -r f; do
            ckpts+=("$f")
        done < <(find "$default_ckpt_dir" -maxdepth 1 -name '*.pt' | sort)
    fi
fi

if [[ ${#ckpts[@]} -eq 0 ]]; then
    echo "Error: No checkpoints found. Specify --ckpt or --ckpt_dir." >&2
    exit 1
fi

for ckpt_file in "${ckpts[@]}"; do
    echo "============================================"
    echo "Evaluating: $ckpt_file"
    echo "============================================"

    CUDA_VISIBLE_DEVICES="$GPU" python \
        "$TRAINER" default \
        --disable_viewer \
        --data_dir "${SCENE_DIR}/${SCENE}" \
        --data_factor "$DATA_FACTOR" \
        --result_dir "$RESULT_DIR" \
        --strategy.cap-max "$CAP_MAX" \
        --render_traj_path ellipse \
        --ckpt "$ckpt_file"
done

stats_dir="$RESULT_DIR/stats"
if [[ -d "$stats_dir" ]]; then
    echo ""
    echo "=== Quality Results ==="

    latest_val=""
    latest_step=-1
    for f in "$stats_dir"/val_step*.json; do
        [[ -f "$f" ]] || continue
        base="$(basename "$f")"
        [[ "$base" == *per_image* ]] && continue
        step=$(echo "$base" | sed -n 's/.*step\([0-9]*\).*/\1/p')
        if [[ -n "$step" && "$step" -gt "$latest_step" ]]; then
            latest_step="$step"
            latest_val="$f"
        fi
    done

    if [[ -n "$latest_val" ]]; then
        python -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
print('  %s:  PSNR=%.3f  SSIM=%.4f  LPIPS=%.3f  #GS=%d' % (
    sys.argv[2], d['psnr'], d['ssim'], d['lpips'], int(d['num_GS'])))
" "$latest_val" "$(basename "$latest_val")"
    fi
fi

if ! $SKIP_RUNTIME; then
    latest_ckpt="${ckpts[-1]}"
    timing_json="$RESULT_DIR/stats/timing.json"

    echo ""
    echo "============================================"
    echo "Runtime Benchmark"
    echo "  Checkpoint: $(basename "$latest_ckpt")"
    echo "  Passes:     $NUM_PASSES (warmup=$WARMUP_FRAMES frames)"
    echo "============================================"

    CUDA_VISIBLE_DEVICES="$GPU" python \
        "$BENCHMARKER" \
        --ckpt "$latest_ckpt" \
        --data_dir "${SCENE_DIR}/${SCENE}" \
        --data_factor "$DATA_FACTOR" \
        --num_passes "$NUM_PASSES" \
        --warmup_frames "$WARMUP_FRAMES" \
        --save_json "$timing_json" \
        --scene_name "$SCENE"

    if [[ -f "$timing_json" ]]; then
        python -c "
import json, sys
with open(sys.argv[1]) as f:
    t = json.load(f)
total = t['total_ms']
r_pct = (t['rasterization_ms'] / total * 100) if total > 0 else 0
m_pct = (t['deferred_mlp_ms'] / total * 100) if total > 0 else 0
print()
print('=== Timing Summary ===')
print('  Raster=%.2fms (%.1f%%)  MLP=%.2fms (%.1f%%)  Total=%.2fms  FPS=%.1f  #GS=%d' % (
    t['rasterization_ms'], r_pct, t['deferred_mlp_ms'], m_pct,
    total, t['fps'], int(t['num_gs'])))
" "$timing_json"
    fi
fi
