#!/bin/bash

set -euo pipefail

bash scripts/train.sh --scene train --scene_dir data/tandt --data_factor 1 --cap_max 1000000 --disable_viewer
bash scripts/train.sh --scene truck --scene_dir data/tandt --data_factor 1 --cap_max 1000000 --disable_viewer
bash scripts/train.sh --scene drjohnson --scene_dir data/deepblend --data_factor 1 --cap_max 1000000 --disable_viewer
bash scripts/train.sh --scene playroom  --scene_dir data/deepblend --data_factor 1 --cap_max 1000000 --disable_viewer
bash scripts/train.sh --scene bicycle  --data_factor 4 --cap_max 1000000 --disable_viewer
bash scripts/train.sh --scene garden   --data_factor 4 --cap_max 1000000 --disable_viewer
bash scripts/train.sh --scene stump    --data_factor 4 --cap_max 1000000 --disable_viewer
bash scripts/train.sh --scene treehill --data_factor 4 --cap_max 1000000 --disable_viewer
bash scripts/train.sh --scene flowers  --data_factor 4 --cap_max 1000000 --disable_viewer
bash scripts/train.sh --scene room     --data_factor 2 --cap_max 1000000 --disable_viewer
bash scripts/train.sh --scene counter  --data_factor 2 --cap_max 1000000 --disable_viewer
bash scripts/train.sh --scene kitchen  --data_factor 2 --cap_max 1000000 --disable_viewer
bash scripts/train.sh --scene bonsai   --data_factor 2 --cap_max 1000000 --disable_viewer
