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
# NHT unified MCMC benchmark (48 features, 1M primitives, 30k steps).
# Standard configuration for Mip-NeRF 360 + Tanks & Temples + Deep Blending. Reproduces Table 2 from NHT paper.
#
# Usage:
#   bash benchmarks/nht/benchmark_nht.sh
#   GPU=1 CAP_MAX=2000000 bash benchmarks/nht/benchmark_nht.sh
#   SCENE_LIST="bonsai garden truck" bash benchmarks/nht/benchmark_nht.sh
#   LPIPS_NET=alex bash benchmarks/nht/benchmark_nht.sh     # INRIA-style VGG unnormalized
#   OUTPUT_ROOT=/path/to/results bash benchmarks/nht/benchmark_nht.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINER="$REPO_ROOT/gsplat/examples/simple_trainer_nht.py"

GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT/data"}
CAP_MAX=${CAP_MAX:-1000000}
MAX_STEPS=${MAX_STEPS:-30000}
FEATURE_DIM=${FEATURE_DIM:-64}
# LPIPS backbone and normalization. Our default: VGG with normalize=True.
# INRIA 3DGS uses VGG with normalize=False (technically incorrect).
# To reproduce INRIA-style numbers: LPIPS_NET=vgg LPIPS_NORMALIZE=0
LPIPS_NET=${LPIPS_NET:-"vgg"}
LPIPS_NORMALIZE=${LPIPS_NORMALIZE:-1}
LPIPS_ARGS="--lpips_net $LPIPS_NET"
[ "$LPIPS_NORMALIZE" -eq 0 ] && LPIPS_ARGS+=" --no-lpips_normalize"
if [ -n "${OUTPUT_ROOT:-}" ]; then
  RESULT_BASE="$OUTPUT_ROOT"
else
  RESULT_BASE=${RESULT_BASE:-"$REPO_ROOT/results/benchmark_nht"}
fi
RENDER_TRAJ_PATH="ellipse"

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

echo "============================================================"
echo "NHT MCMC Benchmark (unified)"
echo "  Features: $FEATURE_DIM   Cap: $CAP_MAX   Steps: $MAX_STEPS"
echo "  Scenes:   ${ALL_SCENES[*]}"
echo "  Results:  $RESULT_BASE"
echo "============================================================"

for SCENE in "${ALL_SCENES[@]}"; do
    DATA_DIR=$(get_data_dir "$SCENE")
    DATA_FACTOR=$(get_factor "$SCENE")
    RESULT_DIR="$RESULT_BASE/$SCENE"

    if [ -z "$DATA_DIR" ] || [ ! -d "$DATA_DIR" ]; then
        echo "WARNING: data not found for $SCENE, skipping"
        continue
    fi

    echo
    echo ">>> Training $SCENE (factor=$DATA_FACTOR, cap=$CAP_MAX, steps=$MAX_STEPS) <<<"

    CUDA_VISIBLE_DEVICES=$GPU python "$TRAINER" mcmc \
        --eval_steps -1 --disable_viewer \
        --data_factor $DATA_FACTOR \
        --max_steps $MAX_STEPS \
        --strategy.cap-max $CAP_MAX \
        --deferred_opt_feature_dim $FEATURE_DIM \
        --ssim_lambda 0.1 \
        --render_traj_path $RENDER_TRAJ_PATH \
        $LPIPS_ARGS \
        --data_dir "$DATA_DIR/" \
        --result_dir "$RESULT_DIR/"

    for CKPT in $(ls -rt "$RESULT_DIR/ckpts/"*.pt 2>/dev/null); do
        echo "  Evaluating $(basename $CKPT) ..."
        CUDA_VISIBLE_DEVICES=$GPU python "$TRAINER" mcmc \
            --disable_viewer \
            --data_factor $DATA_FACTOR \
            --max_steps $MAX_STEPS \
            --strategy.cap-max $CAP_MAX \
            --deferred_opt_feature_dim $FEATURE_DIM \
            --ssim_lambda 0.1 \
            --render_traj_path $RENDER_TRAJ_PATH \
            $LPIPS_ARGS \
            --data_dir "$DATA_DIR/" \
            --result_dir "$RESULT_DIR/" \
            --ckpt "$CKPT"
    done
done

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
