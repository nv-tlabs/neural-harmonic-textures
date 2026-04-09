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
# NHT split-strategy benchmark (paper configuration).
# Different cap-max, steps. Reproduces Table 1 from NHT paper.
#
# M360 outdoor: 5M, 25k iters, per-ray dir encoding
# M360 indoor:  2M,   45k iters, center-ray dir encoding
# T&T:          2.5M, 40k iters, center-ray dir encoding
# Deep Blending: 2M,  30k iters, center-ray dir encoding
#
# Flags:
#   --metrics_only   Skip training; collect results from existing outputs
#
# Usage:
#   bash benchmarks/nht/benchmark_nht_split.sh
#   GPU=1 bash benchmarks/nht/benchmark_nht_split.sh
#   SCENE_LIST="bonsai garden truck" bash benchmarks/nht/benchmark_nht_split.sh
#   OUTPUT_ROOT=/path/to/results bash benchmarks/nht/benchmark_nht_split.sh
#   bash benchmarks/nht/benchmark_nht_split.sh --metrics_only
#   LPIPS_NET=alex bash benchmarks/nht/benchmark_nht_split.sh   # INRIA-style VGG unnormalized

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINER="$REPO_ROOT/gsplat/examples/simple_trainer_nht.py"

GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT/data"}
if [ -n "${OUTPUT_ROOT:-}" ]; then
  RESULT_BASE="$OUTPUT_ROOT"
else
  RESULT_BASE=${RESULT_BASE:-"$REPO_ROOT/results/benchmark_nht_split"}
fi
RENDER_TRAJ_PATH="ellipse"
LPIPS_NET=${LPIPS_NET:-"vgg"}
LPIPS_NORMALIZE=${LPIPS_NORMALIZE:-1}
LPIPS_ARGS="--lpips_net $LPIPS_NET"
[ "$LPIPS_NORMALIZE" -eq 0 ] && LPIPS_ARGS+=" --no-lpips_normalize"
METRICS_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --metrics_only) METRICS_ONLY=1 ;;
    esac
done

COMMON_ARGS="--disable_viewer --render_traj_path $RENDER_TRAJ_PATH --ssim_lambda 0.1 $LPIPS_ARGS"

M360_INDOOR=("bonsai" "counter" "kitchen" "room")
M360_OUTDOOR=("garden" "bicycle" "stump" "treehill" "flowers")
TANDT=("train" "truck")
DB=("drjohnson" "playroom")

if [ -n "${SCENE_LIST:-}" ]; then
    IFS=' ' read -ra FILTER <<< "$SCENE_LIST"
else
    FILTER=()
fi

should_run() {
    local scene=$1
    if [ ${#FILTER[@]} -eq 0 ]; then return 0; fi
    for s in "${FILTER[@]}"; do [ "$s" = "$scene" ] && return 0; done
    return 1
}

get_data_dir() {
    local scene=$1
    case "$scene" in
        train|truck)
            for p in "$DATA_ROOT/tandt_db/tandt/$scene" "$DATA_ROOT/$scene"; do
                [ -d "$p" ] && echo "$p" && return; done ;;
        drjohnson|playroom)
            for p in "$DATA_ROOT/tandt_db/db/$scene" "$DATA_ROOT/$scene"; do
                [ -d "$p" ] && echo "$p" && return; done ;;
        *)
            for p in "$DATA_ROOT/mipnerf360/$scene" "$DATA_ROOT/360_v2/$scene" "$DATA_ROOT/$scene"; do
                [ -d "$p" ] && echo "$p" && return; done ;;
    esac
    echo ""
}

run_scene() {
    local scene=$1 data_dir=$2 factor=$3 cap_max=$4 max_steps=$5 group_args=$6 group=$7
    local result_dir="$RESULT_BASE/$scene"

    should_run "$scene" || return 0

    if [ -z "$data_dir" ] || [ ! -d "$data_dir" ]; then
        echo "WARNING: data not found for $scene, skipping"
        return 0
    fi

    echo ">>> [$group] Training $scene (factor=$factor, cap=$cap_max, steps=$max_steps) <<<"

    CUDA_VISIBLE_DEVICES=$GPU python "$TRAINER" default \
        $COMMON_ARGS --eval_steps -1 \
        --data_factor $factor \
        --max_steps $max_steps \
        --strategy.cap-max $cap_max \
        $group_args \
        --data_dir "$data_dir/" \
        --result_dir "$result_dir/"

    for CKPT in $(ls -rt "$result_dir/ckpts/"*.pt 2>/dev/null); do
        echo "  Evaluating $(basename $CKPT) ..."
        CUDA_VISIBLE_DEVICES=$GPU python "$TRAINER" default \
            $COMMON_ARGS \
            --data_factor $factor \
            --max_steps $max_steps \
            --strategy.cap-max $cap_max \
            $group_args \
            --data_dir "$data_dir/" \
            --result_dir "$result_dir/" \
            --ckpt "$CKPT"
    done
}

echo "============================================================"
echo "NHT Split-Strategy Benchmark (Paper Config)"
echo "  M360 outdoor: 4.5M, 20k steps, per-ray encoding"
echo "  M360 indoor:  2M, 45k steps, center-ray encoding"
echo "  T&T:          2.5M, 40k steps, center-ray encoding"
echo "  Deep Blending: 2M, 30k steps, center-ray encoding"
[ "$METRICS_ONLY" -eq 1 ] && echo "  Mode: METRICS ONLY"
echo "============================================================"

if [ "$METRICS_ONLY" -eq 0 ]; then
    for s in "${M360_OUTDOOR[@]}"; do
        run_scene "$s" "$(get_data_dir $s)" 4 5000000 25000 "" "m360-outdoor"
    done
    for s in "${M360_INDOOR[@]}"; do
        run_scene "$s" "$(get_data_dir $s)" 2 2000000 45000 "--deferred_opt_center_ray_encoding" "m360-indoor"
    done
    for s in "${TANDT[@]}"; do
        run_scene "$s" "$(get_data_dir $s)" 1 2500000 40000 "--deferred_opt_center_ray_encoding" "tandt"
    done
    for s in "${DB[@]}"; do
        run_scene "$s" "$(get_data_dir $s)" 1 2000000 30000 "--deferred_opt_center_ray_encoding" "deepblend"
    done
fi

echo
echo "============================================================"
echo "Results Summary"
echo "============================================================"
for scene_dir in "$RESULT_BASE"/*/stats; do
    scene=$(basename $(dirname "$scene_dir"))
    LATEST=$(ls -t "$scene_dir"/val_step*.json 2>/dev/null | grep -v per_image | head -1)
    if [ -n "$LATEST" ]; then
        echo "  $scene: $(cat "$LATEST")"
    fi
done
echo "Done."
