RESULT_DIR="$1"
if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 <result_dir>"
    exit 1
fi

SCENE_LIST="bicycle garden stump treehill flowers room counter kitchen bonsai drjohnson playroom train truck"

CR="\033[31m" # Red
CG="\033[32m" # Green
CY="\033[33m" # Yellow
CB="\033[34m" # Blue
CM="\033[35m" # Magenta
CC="\033[36m" # Cyan
RESET="\033[0m" # Reset
BOLD="\033[1m"

PRINT_HORIZONTAL=${PRINT_HORIZONTAL:-0}

function find_last_stats() {
    local scene=$1
    local last_step=0
    for stats in $(ls -t $scene/stats/val_step*.json); do
        # extract the step number
        step=$(echo $stats | grep -oE 'step[0-9]+' | grep -oE '[0-9]+')
        if [ $step -gt $last_step ]; then
            last_step=$step
        fi
    done
    echo $last_step
}

function find_last_train_stats() {
    local scene=$1
    local last_step=0
    for stats in $(ls -t $scene/stats/train_step*_rank0.json 2>/dev/null); do
        # extract the step number
        step=$(echo $stats | grep -oE 'step[0-9]+' | grep -oE '[0-9]+')
        if [ $step -gt $last_step ]; then
            last_step=$step
        fi
    done
    echo $last_step
}

function json_value() {
    local stats=$1
    local key=$2
    python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], 'N/A'))" "$stats" "$key"
}

function print_stats() {
    local key=$1
    local -a results=()
    local -a scenes=()
    for SCENE in $SCENE_LIST;
    do
        # skip if folder does not exist
        if [ ! -d "$RESULT_DIR/$SCENE" ]; then
            continue
        fi

        LAST=$(find_last_stats $RESULT_DIR/$SCENE)

        STATS=$RESULT_DIR/$SCENE/stats/val_step$LAST.json
        DATA=$(json_value "$STATS" "$key" | awk '{if ($1 == "N/A") printf "N/A"; else printf "%.3f", $1}')
        scenes+=("$SCENE")
        results+=("$DATA")

        if [ $PRINT_HORIZONTAL -eq 0 ]; then
            printf "${BOLD}${CC}%-10s${CM}%s${RESET}\n" "$SCENE" "$DATA"
        fi
    done

    if [ $PRINT_HORIZONTAL -eq 1 ]; then
        printf "${BOLD}${CC}%-12s${RESET}" "${scenes[@]}"
        printf "\n"
        printf "${CM}%-12s${RESET}" "${results[@]}"
        printf "\n"
    fi
}

function print_train_stats() {
    local key=$1
    local -a results=()
    local -a scenes=()
    for SCENE in $SCENE_LIST;
    do
        # skip if folder does not exist
        if [ ! -d "$RESULT_DIR/$SCENE" ]; then
            continue
        fi

        LAST=$(find_last_train_stats $RESULT_DIR/$SCENE)
        if [ "$LAST" -eq 0 ]; then
            continue
        fi

        STATS=$RESULT_DIR/$SCENE/stats/train_step${LAST}_rank0.json
        DATA=$(json_value "$STATS" "$key" | awk '{printf "%.3f", $1}')
        scenes+=("$SCENE")
        results+=("$DATA")

        if [ $PRINT_HORIZONTAL -eq 0 ]; then
            printf "${BOLD}${CC}%-10s${CM}%s${RESET}\n" "$SCENE" "$DATA"
        fi
    done

    if [ $PRINT_HORIZONTAL -eq 1 ]; then
        printf "${BOLD}${CC}%-12s${RESET}" "${scenes[@]}"
        printf "\n"
        printf "${CM}%-12s${RESET}" "${results[@]}"
        printf "\n"
    fi
}

echo "=== LAST STEP ==="
for SCENE in $SCENE_LIST;
do
    # if folder does not exist, skip
    if [ ! -d "$RESULT_DIR/$SCENE" ]; then
        printf "${BOLD}${CC}%-10s${CR}%s${RESET}\n" "$SCENE" "empty"
    else
        printf "${BOLD}${CC}%-10s${CM}%s${RESET}\n" "$SCENE" "$(find_last_stats $RESULT_DIR/$SCENE)"
    fi
done

echo "=== PSNR ==="
print_stats psnr
echo "=== SSIM ==="
print_stats ssim
echo "=== LPIPS ==="
print_stats lpips_alex
echo "=== LPIPS VGG ==="
print_stats lpips
echo "=== COUNT ==="
print_stats num_GS
echo "=== TRAIN TIME ==="
print_train_stats ellipse_time
echo "=== MEMORY ==="
print_train_stats mem

exit 0