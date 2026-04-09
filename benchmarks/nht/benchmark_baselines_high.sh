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
# Baseline high primitive-count benchmark for 3DGS-MCMC, 3DGUT-MCMC, and 2DGS.
# LPIPS is evaluated with VGG (normalize=True). This differs from INRIA 3DGS
# which uses VGG with normalize=False (technically incorrect).
#
# Methods:
#   3dgs_mcmc   - 3DGS with MCMC strategy  (simple_trainer.py mcmc)
#   3dgut_mcmc  - 3DGUT with MCMC strategy  (simple_trainer.py mcmc --with_ut --with_eval3d)
#   2dgs        - 2DGS with default strategy (simple_trainer_2dgs.py)
#
# Per-scene caps (MCMC methods only; 2DGS uses default densification):
#   bonsai=1.3M counter=1.2M kitchen=1.8M room=1.5M
#   garden=5.2M bicycle=5.9M stump=4.75M treehill=3.5M flowers=3M
#   train=1.1M truck=2.6M drjohnson=3.4M playroom=2.5M
#
# Flags:
#   --metrics_only   Skip training; collect and display metrics only
#
# Usage:
#   bash benchmarks/nht/benchmark_baselines_high.sh
#   GPU=1 bash benchmarks/nht/benchmark_baselines_high.sh
#   SCENE_LIST="garden truck" bash benchmarks/nht/benchmark_baselines_high.sh
#   METHOD_LIST="3dgs_mcmc 2dgs" bash benchmarks/nht/benchmark_baselines_high.sh
#   OUTPUT_ROOT=/path/to/results bash benchmarks/nht/benchmark_baselines_high.sh
#   bash benchmarks/nht/benchmark_baselines_high.sh --metrics_only
#   STEP=29999 bash benchmarks/nht/benchmark_baselines_high.sh --metrics_only
#   LPIPS_NET=alex bash benchmarks/nht/benchmark_baselines_high.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINER_3DGS="$REPO_ROOT/gsplat/examples/simple_trainer.py"
TRAINER_2DGS="$REPO_ROOT/gsplat/examples/simple_trainer_2dgs.py"

GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT/data"}
MAX_STEPS=${MAX_STEPS:-30000}
STEP=${STEP:--1}
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

if [ -n "${OUTPUT_ROOT:-}" ]; then
    RESULT_BASE="$OUTPUT_ROOT"
else
    RESULT_BASE=${RESULT_BASE:-"$REPO_ROOT/results/benchmark_baselines_high"}
fi

# --------------- Per-scene caps ---------------
declare -A SCENE_CAPS=(
    [bonsai]=1300000   [counter]=1200000  [kitchen]=1800000  [room]=1500000
    [garden]=5200000   [bicycle]=5900000  [stump]=4750000    [treehill]=3500000
    [flowers]=3000000  [train]=1100000    [truck]=2600000
    [drjohnson]=3400000 [playroom]=2500000
)

# --------------- Dataset definitions ---------------
M360_INDOOR=("bonsai" "counter" "kitchen" "room")
M360_OUTDOOR=("garden" "bicycle" "stump" "treehill" "flowers")
TANDT=("train" "truck")
DB=("drjohnson" "playroom")

if [ -n "${SCENE_LIST:-}" ]; then
    IFS=' ' read -ra ALL_SCENES <<< "$SCENE_LIST"
else
    ALL_SCENES=("${M360_INDOOR[@]}" "${M360_OUTDOOR[@]}" "${TANDT[@]}" "${DB[@]}")
fi

ALL_METHODS=("3dgs_mcmc" "3dgut_mcmc" "2dgs")
if [ -n "${METHOD_LIST:-}" ]; then
    IFS=' ' read -ra METHODS <<< "$METHOD_LIST"
    for m in "${METHODS[@]}"; do
        valid=0
        for am in "${ALL_METHODS[@]}"; do
            [ "$m" = "$am" ] && valid=1 && break
        done
        if [ "$valid" -eq 0 ]; then
            echo "ERROR: Unknown method '$m'. Valid methods: ${ALL_METHODS[*]}" >&2
            exit 1
        fi
    done
else
    METHODS=("${ALL_METHODS[@]}")
fi

# --------------- Helpers ---------------
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

get_dataset_label() {
    local scene=$1
    case "$scene" in
        bonsai|counter|kitchen|room) echo "mipnerf360" ;;
        garden|bicycle|stump|treehill|flowers) echo "mipnerf360" ;;
        train|truck) echo "tandt" ;;
        drjohnson|playroom) echo "deepblending" ;;
        *) echo "unknown" ;;
    esac
}

is_in_array() {
    local needle=$1; shift
    for item in "$@"; do [ "$item" = "$needle" ] && return 0; done
    return 1
}

# Compute average of values passed as arguments; prints "N/A" when no values
avg() {
    if [ $# -eq 0 ]; then echo "N/A"; return; fi
    local sum=0 n=0
    for v in "$@"; do
        sum=$(echo "$sum + $v" | bc -l)
        n=$((n + 1))
    done
    echo "scale=6; $sum / $n" | bc -l
}

# ===================================================================
# TRAIN + EVAL  (skipped with --metrics_only)
# ===================================================================
if [ "$METRICS_ONLY" -eq 0 ]; then
    echo "============================================================"
    echo "Baselines High-Count Benchmark (per-scene caps)"
    echo "  Methods:  ${METHODS[*]}"
    echo "  Scenes:   ${ALL_SCENES[*]}"
    echo "  Steps:    $MAX_STEPS"
    echo "  Results:  $RESULT_BASE"
    echo "============================================================"

    for METHOD in "${METHODS[@]}"; do
        echo
        echo "============================================================"
        echo "Method: $METHOD"
        echo "============================================================"

        for SCENE in "${ALL_SCENES[@]}"; do
            DATA_DIR=$(get_data_dir "$SCENE")
            DATA_FACTOR=$(get_factor "$SCENE")
            DATASET=$(get_dataset_label "$SCENE")
            CAP_MAX=${SCENE_CAPS[$SCENE]:-1000000}
            RESULT_DIR="$RESULT_BASE/$METHOD/$SCENE"

            if [ -z "$DATA_DIR" ] || [ ! -d "$DATA_DIR" ]; then
                echo "WARNING: data not found for $SCENE, skipping"
                continue
            fi

            echo
            echo ">>> [$DATASET] $METHOD - Training $SCENE (factor=$DATA_FACTOR, cap=$CAP_MAX) <<<"

            COMMON_ARGS="--disable_viewer \
                --data_dir $DATA_DIR \
                --data_factor $DATA_FACTOR \
                --result_dir $RESULT_DIR \
                --max_steps $MAX_STEPS \
                $LPIPS_ARGS"

            case "$METHOD" in
                3dgs_mcmc)
                    CUDA_VISIBLE_DEVICES=$GPU python "$TRAINER_3DGS" mcmc \
                        $COMMON_ARGS \
                        --strategy.cap-max $CAP_MAX
                    ;;
                3dgut_mcmc)
                    CUDA_VISIBLE_DEVICES=$GPU python "$TRAINER_3DGS" mcmc \
                        $COMMON_ARGS \
                        --strategy.cap-max $CAP_MAX \
                        --with_ut \
                        --with_eval3d
                    ;;
                2dgs)
                    CUDA_VISIBLE_DEVICES=$GPU python "$TRAINER_2DGS" \
                        $COMMON_ARGS
                    ;;
            esac
        done
    done
fi

# ===================================================================
# METRICS
# ===================================================================
echo
echo "============================================================"
echo "Results Summary"
echo "  Source: $RESULT_BASE"
echo "============================================================"

# Per-method metrics collection using temp files keyed by method+scene
TMPDIR_METRICS=$(mktemp -d)
trap "rm -rf $TMPDIR_METRICS" EXIT

for METHOD in "${METHODS[@]}"; do
    echo
    echo "------------------------------------------------------------"
    echo "Method: $METHOD"
    echo "------------------------------------------------------------"

    MISSING_SCENES=""

    for SCENE in "${ALL_SCENES[@]}"; do
        DATASET=$(get_dataset_label "$SCENE")
        STATS_DIR="$RESULT_BASE/$METHOD/$SCENE/stats"

        echo
        echo "  [$DATASET] $SCENE"

        if [ ! -d "$STATS_DIR" ]; then
            echo "    ** NO STATS DIR **"
            MISSING_SCENES="$MISSING_SCENES $SCENE"
            continue
        fi

        if [ "$STEP" -gt 0 ] 2>/dev/null; then
            VAL_FILE=$(ls "$STATS_DIR"/val_step*.json 2>/dev/null \
                | grep -v per_image \
                | grep "step0*${STEP}\." \
                | head -1)
        else
            VAL_FILE=$(ls -t "$STATS_DIR"/val_step*.json 2>/dev/null \
                | grep -v per_image \
                | head -1)
        fi

        if [ -z "$VAL_FILE" ]; then
            AVAILABLE=$(ls "$STATS_DIR"/val_step*.json 2>/dev/null | grep -v per_image | xargs -I{} basename {} 2>/dev/null | tr '\n' ' ')
            STEP_LABEL=$( [ "$STEP" -gt 0 ] 2>/dev/null && echo "$STEP" || echo "latest" )
            echo "    ** MISSING step $STEP_LABEL **  (available: $AVAILABLE)"
            MISSING_SCENES="$MISSING_SCENES $SCENE"
            continue
        fi

        PSNR=$(python -c "import json; d=json.load(open('$VAL_FILE')); print(f\"{d['psnr']:.3f}\")")
        SSIM=$(python -c "import json; d=json.load(open('$VAL_FILE')); print(f\"{d['ssim']:.4f}\")")
        LPIPS=$(python -c "import json; d=json.load(open('$VAL_FILE')); print(f\"{d['lpips']:.3f}\")")
        NUM_GS=$(python -c "import json; d=json.load(open('$VAL_FILE')); print(int(d['num_GS']))")

        echo "    $(basename "$VAL_FILE"):  PSNR=$PSNR  SSIM=$SSIM  LPIPS=$LPIPS  #GS=$NUM_GS"

        mkdir -p "$TMPDIR_METRICS/$METHOD"
        echo "$PSNR $SSIM $LPIPS $NUM_GS" > "$TMPDIR_METRICS/$METHOD/$SCENE"
    done

    # Aggregated results per method
    if [ -d "$TMPDIR_METRICS/$METHOD" ] && [ "$(ls "$TMPDIR_METRICS/$METHOD" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo
        echo "  Aggregated ($METHOD):"
        printf "  %-10s %4s  %8s  %8s  %8s  %10s\n" "Split" "N" "PSNR" "SSIM" "LPIPS" "#GS"
        printf "  %s\n" "--------------------------------------------------------------"

        print_agg_row() {
            local label=$1; shift
            local -a scenes=("$@")
            local psnr_vals=() ssim_vals=() lpips_vals=() gs_vals=()
            for s in "${scenes[@]}"; do
                if [ -f "$TMPDIR_METRICS/$METHOD/$s" ]; then
                    read -r p ss lp gs < "$TMPDIR_METRICS/$METHOD/$s"
                    psnr_vals+=("$p"); ssim_vals+=("$ss"); lpips_vals+=("$lp"); gs_vals+=("$gs")
                fi
            done
            local n=${#psnr_vals[@]}
            if [ "$n" -eq 0 ]; then return; fi
            local avg_psnr avg_ssim avg_lpips avg_gs
            avg_psnr=$(avg "${psnr_vals[@]}")
            avg_ssim=$(avg "${ssim_vals[@]}")
            avg_lpips=$(avg "${lpips_vals[@]}")
            avg_gs=$(avg "${gs_vals[@]}")
            printf "  %-10s %4d  %8.3f  %8.4f  %8.3f  %10.0f\n" "$label" "$n" "$avg_psnr" "$avg_ssim" "$avg_lpips" "$avg_gs"
        }

        print_agg_row "M360-In"  "${M360_INDOOR[@]}"
        print_agg_row "M360-Out" "${M360_OUTDOOR[@]}"
        print_agg_row "M360"     "${M360_INDOOR[@]}" "${M360_OUTDOOR[@]}"
        print_agg_row "T&T"      "${TANDT[@]}"
        print_agg_row "DB"       "${DB[@]}"
        print_agg_row "Overall"  "${ALL_SCENES[@]}"
    fi

    if [ -n "$MISSING_SCENES" ]; then
        echo
        echo "  MISSING ($METHOD):$MISSING_SCENES"
    fi
done

# ===================================================================
# CROSS-METHOD COMPARISON
# ===================================================================
if [ "${#METHODS[@]}" -gt 1 ]; then
    echo
    echo "============================================================"
    echo "Cross-Method Comparison (Overall Averages)"
    echo "============================================================"

    printf "  %-14s %4s  %8s  %8s  %8s  %12s\n" "Method" "N" "PSNR" "SSIM" "LPIPS" "Avg #GS"
    printf "  %s\n" "--------------------------------------------------------------------"

    print_method_row() {
        local method=$1; shift
        local -a scenes=("$@")
        local psnr_vals=() ssim_vals=() lpips_vals=() gs_vals=()
        for s in "${scenes[@]}"; do
            if [ -f "$TMPDIR_METRICS/$method/$s" ]; then
                read -r p ss lp gs < "$TMPDIR_METRICS/$method/$s"
                psnr_vals+=("$p"); ssim_vals+=("$ss"); lpips_vals+=("$lp"); gs_vals+=("$gs")
            fi
        done
        local n=${#psnr_vals[@]}
        if [ "$n" -eq 0 ]; then return; fi
        local avg_psnr avg_ssim avg_lpips avg_gs
        avg_psnr=$(avg "${psnr_vals[@]}")
        avg_ssim=$(avg "${ssim_vals[@]}")
        avg_lpips=$(avg "${lpips_vals[@]}")
        avg_gs=$(avg "${gs_vals[@]}")
        printf "  %-14s %4d  %8.3f  %8.4f  %8.3f  %12.0f\n" "$method" "$n" "$avg_psnr" "$avg_ssim" "$avg_lpips" "$avg_gs"
    }

    for METHOD in "${METHODS[@]}"; do
        print_method_row "$METHOD" "${ALL_SCENES[@]}"
    done

    echo
    echo "--- Markdown table ---"
    echo "| Method         |  N | PSNR  | SSIM   | LPIPS | Avg #GS     |"
    echo "|----------------|----|-------|--------|-------|-------------|"
    for METHOD in "${METHODS[@]}"; do
        local_psnr_vals=() local_ssim_vals=() local_lpips_vals=() local_gs_vals=()
        for s in "${ALL_SCENES[@]}"; do
            if [ -f "$TMPDIR_METRICS/$METHOD/$s" ]; then
                read -r p ss lp gs < "$TMPDIR_METRICS/$METHOD/$s"
                local_psnr_vals+=("$p"); local_ssim_vals+=("$ss"); local_lpips_vals+=("$lp"); local_gs_vals+=("$gs")
            fi
        done
        n=${#local_psnr_vals[@]}
        if [ "$n" -eq 0 ]; then continue; fi
        avg_psnr=$(avg "${local_psnr_vals[@]}")
        avg_ssim=$(avg "${local_ssim_vals[@]}")
        avg_lpips=$(avg "${local_lpips_vals[@]}")
        avg_gs=$(avg "${local_gs_vals[@]}")
        printf "| %-14s | %2d | %.3f | %.4f | %.3f | %11.0f |\n" "$METHOD" "$n" "$avg_psnr" "$avg_ssim" "$avg_lpips" "$avg_gs"
    done

    # Per-split comparison across methods
    echo
    echo "--- Per-split comparison ---"

    SPLIT_NAMES=("M360-In" "M360-Out" "M360" "T&T" "DB" "Overall")

    get_split_scenes() {
        local split=$1
        case "$split" in
            M360-In)  echo "${M360_INDOOR[*]}" ;;
            M360-Out) echo "${M360_OUTDOOR[*]}" ;;
            M360)     echo "${M360_INDOOR[*]} ${M360_OUTDOOR[*]}" ;;
            "T&T")    echo "${TANDT[*]}" ;;
            DB)       echo "${DB[*]}" ;;
            Overall)  echo "${ALL_SCENES[*]}" ;;
        esac
    }

    for SPLIT in "${SPLIT_NAMES[@]}"; do
        echo
        echo "  $SPLIT"
        printf "    %-14s %4s  %8s  %8s  %8s  %12s\n" "Method" "N" "PSNR" "SSIM" "LPIPS" "Avg #GS"

        IFS=' ' read -ra SPLIT_SCENES <<< "$(get_split_scenes "$SPLIT")"
        for METHOD in "${METHODS[@]}"; do
            local_psnr_vals=() local_ssim_vals=() local_lpips_vals=() local_gs_vals=()
            for s in "${SPLIT_SCENES[@]}"; do
                if [ -f "$TMPDIR_METRICS/$METHOD/$s" ]; then
                    read -r p ss lp gs < "$TMPDIR_METRICS/$METHOD/$s"
                    local_psnr_vals+=("$p"); local_ssim_vals+=("$ss"); local_lpips_vals+=("$lp"); local_gs_vals+=("$gs")
                fi
            done
            n=${#local_psnr_vals[@]}
            if [ "$n" -eq 0 ]; then continue; fi
            avg_psnr=$(avg "${local_psnr_vals[@]}")
            avg_ssim=$(avg "${local_ssim_vals[@]}")
            avg_lpips=$(avg "${local_lpips_vals[@]}")
            avg_gs=$(avg "${local_gs_vals[@]}")
            printf "    %-14s %4d  %8.3f  %8.4f  %8.3f  %12.0f\n" "$METHOD" "$n" "$avg_psnr" "$avg_ssim" "$avg_lpips" "$avg_gs"
        done
    done
fi

echo
echo "Done."
