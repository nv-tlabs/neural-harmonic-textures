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

# Train an NHT model on a single scene.
#
# Usage:
#   ./scripts/train.sh
#   ./scripts/train.sh --scene kitchen --data_factor 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAINER="$REPO_ROOT/gsplat/examples/simple_trainer_nht.py"

SCENE="garden"
SCENE_DIR="$REPO_ROOT/data/mipnerf360"
DATA_FACTOR=4
CAP_MAX=1000000
RESULT_DIR=""
GPU=0
PORT=8080
DISABLE_VIEWER=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)        SCENE="$2";       shift 2 ;;
        --scene_dir)    SCENE_DIR="$2";   shift 2 ;;
        --data_factor)  DATA_FACTOR="$2"; shift 2 ;;
        --cap_max)      CAP_MAX="$2";     shift 2 ;;
        --result_dir)   RESULT_DIR="$2";  shift 2 ;;
        --gpu)          GPU="$2";         shift 2 ;;
        --port)         PORT="$2";        shift 2 ;;
        --disable_viewer) DISABLE_VIEWER=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$RESULT_DIR" ]]; then
    RESULT_DIR="$REPO_ROOT/results/nht_mcmc_${CAP_MAX}/${SCENE}"
fi

args=(
    "$TRAINER" default
    --data_dir "${SCENE_DIR}/${SCENE}"
    --data_factor "$DATA_FACTOR"
    --result_dir "$RESULT_DIR"
    --strategy.cap-max "$CAP_MAX"
    --port "$PORT"
    --render_traj_path ellipse
)

if $DISABLE_VIEWER; then
    args+=(--disable_viewer)
fi

echo "============================================"
echo "NHT Training"
echo "  Scene:      $SCENE"
echo "  CapMax:     $CAP_MAX"
echo "  Result dir: $RESULT_DIR"
echo "============================================"

CUDA_VISIBLE_DEVICES="$GPU" python "${args[@]}"
