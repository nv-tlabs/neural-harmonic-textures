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
# NHT with standard 3DGS-MCMC high primitive counts (per-scene).
# Each scene uses the default high-quality cap from standard 3DGS-MCMC and 3DGS-trained scenes. Reproduces Table 7 from NHT paper.
#
# Flags:
#   --metrics_only   Skip training and runtime; collect and display metrics only
#   --runtime_only   Skip training; run only runtime measurement + metrics
#
# Usage:
#   bash benchmarks/nht/benchmark_nht_high.sh
#   GPU=1 bash benchmarks/nht/benchmark_nht_high.sh
#   SCENE_LIST="garden bonsai truck" bash benchmarks/nht/benchmark_nht_high.sh
#   OUTPUT_ROOT=/path/to/results bash benchmarks/nht/benchmark_nht_high.sh
#   bash benchmarks/nht/benchmark_nht_high.sh --metrics_only
#   bash benchmarks/nht/benchmark_nht_high.sh --runtime_only
#   LPIPS_NET=alex bash benchmarks/nht/benchmark_nht_high.sh   # INRIA-style VGG unnormalized

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINER="$REPO_ROOT/gsplat/examples/simple_trainer_nht.py"
BENCHMARKER="$REPO_ROOT/benchmarks/benchmark_nht.py"

GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT/data"}
MAX_STEPS=${MAX_STEPS:-30000}
FEATURE_DIM=${FEATURE_DIM:-64}
LPIPS_NET=${LPIPS_NET:-"vgg"}
LPIPS_NORMALIZE=${LPIPS_NORMALIZE:-1}
LPIPS_ARGS="--lpips_net $LPIPS_NET"
[ "$LPIPS_NORMALIZE" -eq 0 ] && LPIPS_ARGS+=" --no-lpips_normalize"
METRICS_ONLY=0
RUNTIME_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --metrics_only)   METRICS_ONLY=1 ;;
        --runtime_only)   RUNTIME_ONLY=1 ;;
    esac
done
if [ -n "${OUTPUT_ROOT:-}" ]; then
  RESULT_BASE="$OUTPUT_ROOT"
else
  RESULT_BASE=${RESULT_BASE:-"$REPO_ROOT/results/benchmark_nht_high"}
fi
RENDER_TRAJ_PATH="ellipse"

declare -A SCENE_CAPS=(
    [bonsai]=1300000   [counter]=1200000  [kitchen]=1800000  [room]=1500000
    [garden]=5200000   [bicycle]=5900000  [stump]=4750000    [treehill]=3500000
    [flowers]=3000000  [train]=1100000    [truck]=2600000
    [drjohnson]=3400000 [playroom]=2500000
)

M360_INDOOR=("bonsai" "counter" "kitchen" "room")
M360_OUTDOOR=("garden" "bicycle" "stump" "treehill" "flowers")
TANDT=("train" "truck")
DB=("drjohnson" "playroom")

if [ -n "${SCENE_LIST:-}" ]; then
    IFS=' ' read -ra ALL_SCENES <<< "$SCENE_LIST"
else
    ALL_SCENES=("${M360_INDOOR[@]}" "${M360_OUTDOOR[@]}" "${TANDT[@]}" "${DB[@]}")
fi

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

get_factor() {
    local scene=$1
    case "$scene" in
        bonsai|counter|kitchen|room) echo 2 ;;
        train|truck|drjohnson|playroom) echo 1 ;;
        *) echo 4 ;;
    esac
}

NHT_ARGS="--deferred_opt_feature_dim $FEATURE_DIM"
NHT_ARGS+=" --deferred_opt_view_encoding_type sh"
NHT_ARGS+=" --deferred_mlp_hidden_dim 128"
NHT_ARGS+=" --deferred_mlp_num_layers 3"
NHT_ARGS+=" --deferred_mlp_ema"
NHT_ARGS+=" --deferred_features_lr 0.015"
NHT_ARGS+=" --deferred_mlp_lr 0.0072"
NHT_ARGS+=" --opacity_reg 0.02"
NHT_ARGS+=" --scale_reg 0.01"
NHT_ARGS+=" --color_refine_steps 3000"
NHT_ARGS+=" --deferred_features_lr_decay_final 0.1"
NHT_ARGS+=" --deferred_mlp_lr_decay_final 0.1"
NHT_ARGS+=" --deferred_opt_center_ray_encoding"

echo "============================================================"
echo "NHT High-Count Benchmark (per-scene 3DGS-MCMC caps)"
echo "  Features: $FEATURE_DIM   Steps: $MAX_STEPS"
echo "  Scenes:   ${ALL_SCENES[*]}"
echo "  Results:  $RESULT_BASE"
[ "$METRICS_ONLY" -eq 1 ] && echo "  Mode:     METRICS ONLY"
[ "$RUNTIME_ONLY" -eq 1 ] && echo "  Mode:     RUNTIME ONLY"
echo "============================================================"

if [ "$METRICS_ONLY" -eq 0 ] && [ "$RUNTIME_ONLY" -eq 0 ]; then
    for SCENE in "${ALL_SCENES[@]}"; do
        DATA_DIR=$(get_data_dir "$SCENE")
        DATA_FACTOR=$(get_factor "$SCENE")
        CAP_MAX=${SCENE_CAPS[$SCENE]:-1000000}
        RESULT_DIR="$RESULT_BASE/$SCENE"

        if [ -z "$DATA_DIR" ] || [ ! -d "$DATA_DIR" ]; then
            echo "WARNING: data not found for $SCENE, skipping"
            continue
        fi

        echo
        echo ">>> Training $SCENE (factor=$DATA_FACTOR, cap=$CAP_MAX, steps=$MAX_STEPS, fdim=$FEATURE_DIM) <<<"

        CUDA_VISIBLE_DEVICES=$GPU python "$TRAINER" default \
            --eval_steps -1 --disable_viewer \
            --data_factor $DATA_FACTOR \
            --max_steps $MAX_STEPS \
            --strategy.cap-max $CAP_MAX \
            --render_traj_path $RENDER_TRAJ_PATH \
            $LPIPS_ARGS \
            $NHT_ARGS \
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
                $LPIPS_ARGS \
                $NHT_ARGS \
                --data_dir "$DATA_DIR/" \
                --result_dir "$RESULT_DIR/" \
                --ckpt "$CKPT"
        done
    done
fi

if [ "$METRICS_ONLY" -eq 0 ]; then
    echo
    echo "============================================================"
    echo "Running NHT Timing Benchmark"
    echo "============================================================"
    SCENE_CSV=$(IFS=,; echo "${ALL_SCENES[*]}")
    CUDA_VISIBLE_DEVICES=$GPU python "$BENCHMARKER" \
        --results_dir "$RESULT_BASE" \
        --scene_dir "$DATA_ROOT" \
        --scenes "$SCENE_CSV"
fi

if [ "$RUNTIME_ONLY" -eq 0 ]; then
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
fi
