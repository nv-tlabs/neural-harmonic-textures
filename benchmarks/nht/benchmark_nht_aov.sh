#!/bin/bash
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
#
# NHT AOV benchmark: trains deferred AOV heads (LSEG, DINOv3, RGB2X).
# Uses simple_trainer_nht_aov.py with MCMC strategy.
#
# Flags:
#   --metrics_only   Skip training; collect results from existing outputs
#
# Usage:
#   bash benchmarks/nht/benchmark_nht_aov.sh
#   GPU=1 AOV_TARGET=dinov3 bash benchmarks/nht/benchmark_nht_aov.sh
#   SCENE_LIST="garden bonsai" AOV_TARGET=lseg bash benchmarks/nht/benchmark_nht_aov.sh
#   OUTPUT_ROOT=/path/to/results bash benchmarks/nht/benchmark_nht_aov.sh
#   bash benchmarks/nht/benchmark_nht_aov.sh --metrics_only
#   LPIPS_NET=alex bash benchmarks/nht/benchmark_nht_aov.sh   # INRIA-style VGG unnormalized

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINER="$REPO_ROOT/gsplat/examples/simple_trainer_nht_aov.py"

GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT/data"}
CAP_MAX=${CAP_MAX:-1000000}
MAX_STEPS=${MAX_STEPS:-30000}
AOV_TARGET=${AOV_TARGET:-"lseg"}
LPIPS_NET=${LPIPS_NET:-"vgg"}
LPIPS_NORMALIZE=${LPIPS_NORMALIZE:-1}
LPIPS_ARGS="--lpips_net $LPIPS_NET"
[ "$LPIPS_NORMALIZE" -eq 0 ] && LPIPS_ARGS+=" --no-lpips_normalize"
if [ -n "${OUTPUT_ROOT:-}" ]; then
  RESULT_BASE="$OUTPUT_ROOT"
else
  RESULT_BASE=${RESULT_BASE:-"$REPO_ROOT/results/benchmark_nht_aov"}
fi
RENDER_TRAJ_PATH="ellipse"
METRICS_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --metrics_only) METRICS_ONLY=1 ;;
    esac
done

M360_INDOOR=("bonsai" "counter" "kitchen" "room")
M360_OUTDOOR=("garden" "bicycle" "stump" "treehill" "flowers")

if [ -n "${SCENE_LIST:-}" ]; then
    IFS=' ' read -ra ALL_SCENES <<< "$SCENE_LIST"
else
    ALL_SCENES=("${M360_INDOOR[@]}" "${M360_OUTDOOR[@]}")
fi

get_data_dir() {
    local scene=$1
    for p in "$DATA_ROOT/mipnerf360/$scene" "$DATA_ROOT/360_v2/$scene" "$DATA_ROOT/$scene"; do
        [ -d "$p" ] && echo "$p" && return
    done
    echo ""
}

get_factor() {
    local scene=$1
    case "$scene" in
        bonsai|counter|kitchen|room) echo 2 ;;
        *) echo 4 ;;
    esac
}

get_aov_flag() {
    case "$AOV_TARGET" in
        lseg)   echo "--lseg_data" ;;
        dinov3)  echo "--dinov3_data" ;;
        rgb2x)   echo "--rgb2x_data" ;;
        *)       echo "ERROR: unknown AOV target: $AOV_TARGET" >&2; exit 1 ;;
    esac
}

AOV_FLAG=$(get_aov_flag)

echo "============================================================"
echo "NHT AOV Benchmark"
echo "  Target:  $AOV_TARGET   Cap: $CAP_MAX   Steps: $MAX_STEPS"
echo "  Scenes:  ${ALL_SCENES[*]}"
echo "  Results: $RESULT_BASE"
[ "$METRICS_ONLY" -eq 1 ] && echo "  Mode:    METRICS ONLY"
echo "============================================================"

if [ "$METRICS_ONLY" -eq 0 ]; then
for SCENE in "${ALL_SCENES[@]}"; do
    DATA_DIR=$(get_data_dir "$SCENE")
    DATA_FACTOR=$(get_factor "$SCENE")
    RESULT_DIR="$RESULT_BASE/$SCENE"

    if [ -z "$DATA_DIR" ] || [ ! -d "$DATA_DIR" ]; then
        echo "WARNING: data not found for $SCENE, skipping"
        continue
    fi

    echo
    echo ">>> Training $SCENE (factor=$DATA_FACTOR, target=$AOV_TARGET) <<<"

    CUDA_VISIBLE_DEVICES=$GPU python "$TRAINER" default \
        --eval_steps -1 --disable_viewer \
        --data_factor $DATA_FACTOR \
        --max_steps $MAX_STEPS \
        --strategy.cap-max $CAP_MAX \
        --render_traj_path $RENDER_TRAJ_PATH \
        --num_workers ${NUM_WORKERS:-4} \
        $LPIPS_ARGS \
        $AOV_FLAG \
        --data_dir "$DATA_DIR/" \
        --result_dir "$RESULT_DIR/"

    for CKPT in $(ls -rt "$RESULT_DIR/ckpts/"*.pt 2>/dev/null); do
        echo "  Evaluating $(basename $CKPT) ..."
        CUDA_VISIBLE_DEVICES=$GPU python "$TRAINER" default \
            --disable_viewer \
            --data_factor $DATA_FACTOR \
            --max_steps $MAX_STEPS \
            --strategy.cap-max $CAP_MAX \
            --render_traj_path $RENDER_TRAJ_PATH \
            --num_workers ${NUM_WORKERS:-4} \
            $LPIPS_ARGS \
            $AOV_FLAG \
            --data_dir "$DATA_DIR/" \
            --result_dir "$RESULT_DIR/" \
            --ckpt "$CKPT"
    done
done
fi

echo
echo "============================================================"
echo "Results Summary ($AOV_TARGET)"
echo "============================================================"
for scene_dir in "$RESULT_BASE"/*/stats; do
    scene=$(basename $(dirname "$scene_dir"))
    # Highest step in filename (mtime order can pick the wrong ckpt after multi-eval).
    LATEST=""
    best=-1
    for f in "$scene_dir"/val_step*.json; do
        [ -f "$f" ] || continue
        case "$f" in *per_image*) continue ;; esac
        n=$(basename "$f" | sed -n 's/^val_step\([0-9]*\)\.json$/\1/p')
        [ -n "$n" ] && [ "$n" -gt "$best" ] && best=$n && LATEST=$f
    done
    if [ -n "$LATEST" ]; then
        echo "  $scene: $(cat "$LATEST")"
    fi
done
echo "Done."
