<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<div align="center">

# Neural Harmonic Textures for High-Quality Primitive Based Neural Reconstruction

[Jorge Condor](mailto:jorge.condor@usi.ch)<sup>1,2</sup>,
[Nicolas Moenne-Loccoz](mailto:nicolasm@nvidia.com)<sup>2</sup>,
[Merlin Nimier-David](mailto:mnimierdavid@nvidia.com)<sup>2</sup>,
[Piotr Didyk](mailto:piotr.didyk@usi.ch)<sup>1</sup>,
[Zan Gojcic](mailto:zgojcic@nvidia.com)<sup>2</sup>,
[Qi Wu](mailto:qiwu@nvidia.com)<sup>2</sup>

<sup>1</sup>Universit&agrave; della Svizzera italiana, Lugano, Switzerland &nbsp;&nbsp;
<sup>2</sup>NVIDIA

[[Paper]](#) &nbsp;
[[Project Page]](#) &nbsp;
[[Video]](#)

<!-- TODO: add teaser image -->

</div>

---

## Abstract

Primitive-based methods such as 3D Gaussian Splatting have recently become the state-of-the-art for novel-view synthesis and related reconstruction tasks. Compared to neural fields, these representations are more flexible, adaptive, and scale better to large scenes. However, the limited expressivity of individual primitives makes modeling high-frequency detail challenging.

We introduce *Neural Harmonic Textures*, a neural representation approach that anchors latent feature vectors on a virtual scaffold surrounding each primitive. These features are interpolated within the primitive at ray intersection points. Inspired by Fourier analysis, we apply periodic activations to the interpolated features, turning alpha blending into a weighted sum of harmonic components. The resulting signal is then decoded in a single deferred pass using a small neural network, significantly reducing computational cost.

Neural Harmonic Textures yield state-of-the-art results in real-time novel view synthesis while bridging the gap between primitive- and neural-field-based reconstruction. It can be interpreted as a Lagrangian alternative to positional encoding in neural fields.

---

## Installation

### Prerequisites

| Component | Requirement |
|---|---|
| **Python** | >= 3.8 |
| **PyTorch** | >= 2.0 with CUDA support |
| **CUDA** | >= 12.0 (required for tiny-cuda-nn cooperative vectors) |
| **GPU** | Ada Lovelace or newer recommended (RTX 4090, A6000 Ada, L40, etc.); Ampere GPUs (A100, RTX 3090) also work |

### Quick setup

```bash
# Clone with submodule
git clone --recurse-submodules <repo-url>

# Run the setup script (Linux)
bash setup.sh
```

```powershell
# Windows (PowerShell)
git clone --recurse-submodules <repo-url>
.\setup.ps1
```

### Manual setup

```bash
git clone --recurse-submodules <repo-url>

# Install gsplat from submodule
pip install -e ./gsplat

# Install additional dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Training

```bash
# Train on the garden scene (MipNeRF 360, outdoor, factor 4, 1M primitives)
bash scripts/train.sh

# Train on kitchen (indoor, factor 2)
bash scripts/train.sh --scene kitchen --data_factor 2

# Train with 2M primitives
bash scripts/train.sh --scene bonsai --data_factor 2 --cap_max 2000000
```

```powershell
# Windows
.\scripts\train.ps1
.\scripts\train.ps1 -Scene kitchen -DataFactor 2
```

### Viewing

```bash
# Launch the interactive viewer
bash scripts/view.sh --ckpt results/nht_mcmc_1000000/garden/ckpts/ckpt_29999_rank0.pt
```

```powershell
# Windows
.\scripts\view.ps1 -Ckpt results\nht_mcmc_1000000\garden\ckpts\ckpt_29999_rank0.pt
```

The viewer starts a [viser](https://viser.studio/) server. Open `http://localhost:8080` in your browser.

**Viewer render modes** (selectable in the UI dropdown):

| Mode | Description |
|---|---|
| `rgb` | Final decoded RGB color (features -> MLP -> color) |
| `depth(accumulated)` | Accumulated z-depth (alpha-weighted sum of depths) |
| `depth(expected)` | Expected depth (accumulated depth normalized by alpha) |
| `alpha` | Accumulated opacity / transmittance map |

### Evaluation

```bash
# Evaluate quality metrics + runtime benchmark
bash scripts/eval.sh --ckpt results/nht_mcmc_1000000/garden/ckpts/ckpt_29999_rank0.pt \
    --scene garden --scene_dir data/mipnerf360 --data_factor 4

# Skip runtime benchmark
bash scripts/eval.sh --ckpt results/nht_mcmc_1000000/garden/ckpts/ckpt_29999_rank0.pt --skip_runtime
```

---

## Benchmarks and dataset paths

### Running benchmarks

From the **repository root** (with the environment from [Installation](#installation) and a CUDA-visible GPU):

| What | Command |
|---|---|
| Paper Table 2 (unified MCMC) | `bash benchmarks/nht/benchmark_nht.sh` |
| Paper Table 1 (split strategy) | `bash benchmarks/nht/benchmark_nht_split.sh` |
| Paper Table 7 (high primitive count) | `bash benchmarks/nht/benchmark_nht_high.sh` |
| AOV (LSEG / DINOv3/ RGB2X) | `bash benchmarks/nht/benchmark_nht_aov.sh` |
| Quick MipNeRF-360-style sweep | `bash benchmarks/basic_nht.sh` |
| Standalone **runtime** timing (raster + deferred MLP) | `python benchmarks/benchmark_nht.py --ckpt <ckpt.pt> --data_dir <scene_dir> --data_factor <N>` |

On Windows, use the matching scripts under `benchmarks/nht/` (for example `.\benchmarks\nht\benchmark_nht.ps1`).

**Useful environment variables** (bash benchmarks under `benchmarks/nht/`):

| Variable | Default | Role |
|---|---|---|
| `GPU` | `0` | `CUDA_VISIBLE_DEVICES` for training and timing |
| `DATA_ROOT` | `<repo>/data` | Root folder used to resolve scene paths (see below) |
| `SCENE_LIST` | (all paper scenes) | Space-separated subset, e.g. `SCENE_LIST="garden bonsai"` |
| `RESULT_BASE` | varies per script | Where checkpoints and stats are written |
| `CAP_MAX`, `MAX_STEPS`, `FEATURE_DIM` | script defaults | Training budget overrides for Table 2-style runs |

Flags such as `--metrics_only` (split / high / AOV) and `--runtime_only` (high) skip training or metric collection when you already have outputs. For eval plus timing on one checkpoint, use `scripts/eval.sh` / `scripts/eval.ps1` (see [Evaluation](#evaluation)).

**`benchmark_nht.py` batch mode** (one timing run per scene under a results tree):

```bash
python benchmarks/benchmark_nht.py --results_dir results/benchmark_nht --scene_dir data
```

Scene names are taken from subdirectories of `--results_dir`. With `--scene_dir`, each scene path is resolved by trying `<scene_dir>/<scene>`, then `<scene_dir>/mipnerf360/<scene>`, `tandt_db/tandt`, `tandt_db/db`, and a few other dataset layouts. Use `--collect_only` to aggregate existing `stats/timing.json` files without re-running GPU timing.

### Pointing the code at your data

Datasets are **not** shipped with the repo. By convention they live under `<repo>/data/` (gitignored). The trainer expects a **single scene directory** in COLMAP / MipNeRF-360 style (images, poses, sparse reconstruction), passed as `--data_dir`.

**Repo helper scripts** (`scripts/train.sh`, `scripts/eval.sh`, `scripts/view.sh`):

- `--scene_dir` — parent directory containing one folder per scene name.
- `--scene` — scene folder name; the full path is `scene_dir/scene`.

Defaults use `data/mipnerf360` and `garden`. PowerShell equivalents use `-SceneDir` and `-Scene`.

**Paper benchmark shell scripts** (`benchmarks/nht/*.sh`) set `DATA_ROOT` to the directory that **contains** the dataset trees. For each scene they search in order, for example:

- **MipNeRF 360:** `DATA_ROOT/mipnerf360/<scene>`, then `DATA_ROOT/360_v2/<scene>`, then `DATA_ROOT/<scene>`.
- **Tanks & Temples:** `DATA_ROOT/tandt_db/tandt/<scene>` or `DATA_ROOT/<scene>`.
- **Deep Blending:** `DATA_ROOT/tandt_db/db/<scene>` or `DATA_ROOT/<scene>`.

To use a different disk location, either symlink that layout under `data/` or set `DATA_ROOT` to the parent of `mipnerf360/` / `tandt_db/` (or to a flat folder of scene directories).

**Direct Python** (see `gsplat/examples/simple_trainer_nht.py`): pass `--data_dir /path/to/one/scene` and `--data_factor` explicitly; no separate `scene_dir` argument in the trainer itself.

---

## Reproducing Paper Results

The paper evaluates on three standard benchmarks: **MipNeRF 360**, **Tanks & Temples**, and **Deep Blending**. Place datasets under `data/`:

```
data/
  mipnerf360/{garden,bicycle,stump,...}
  tandt_db/tandt/{train,truck}
  tandt_db/db/{drjohnson,playroom}
```

### Table 2 -- Controlled Comparison (1M primitives, 30k steps)


```bash
bash benchmarks/nht/benchmark_nht.sh

# If you want to override defaults
GPU=1 CAP_MAX=2000000 bash benchmarks/nht/benchmark_nht.sh
SCENE_LIST="bonsai garden truck" bash benchmarks/nht/benchmark_nht.sh
```

**Measured results (RTX A6000 Ada):**

| Method (w/ MCMC) | M360 PSNR | M360 SSIM | M360 LPIPS | T&T PSNR | T&T SSIM | T&T LPIPS | DB PSNR | DB SSIM | DB LPIPS |
|---|---|---|---|---|---|---|---|---|---|
| 3DGS + SH | 27.94 | 0.829 | 0.246 | 24.25 | 0.861 | 0.188 | 29.98 | 0.912 | 0.317 |
| 3DGUT + SH | 27.93 | 0.828 | 0.247 | 23.99 | 0.859 | 0.192 | 30.21 | 0.913 | 0.318 |
| 3DGUT + NHT (Ours) | **28.46** | **0.830** | **0.232** | **24.79** | **0.875** | **0.169** | **30.88** | **0.918** | **0.311** |

### Table 1 -- Split-Strategy Benchmark (Best Quality, Per-Dataset Config)

| Dataset Group | Primitives | Steps | Ray Encoding | Data Factor |
|---|---|---|---|---|
| M360 Outdoor | 4.5M | 20k | per-pixel ray | 4 |
| M360 Indoor | 2M | 45k | center ray | 2 |
| Tanks & Temples | 2.5M | 40k | center ray | 1 |
| Deep Blending | 2M | 30k | center ray | 1 |

```bash
bash benchmarks/nht/benchmark_nht_split.sh

# Collect results only (skip training)
bash benchmarks/nht/benchmark_nht_split.sh --metrics_only
```

**Measured results:**

| Dataset | PSNR | SSIM | LPIPS |
|---|---|---|---|
| M360 Outdoor Avg | 25.34 | 0.755 | 0.236 |
| M360 Indoor Avg | 33.59 | 0.947 | 0.183 |
| **M360 Total** | **29.01** | **0.840** | **0.212** |
| **T&T Avg** | **25.68** | **0.882** | **0.141** |
| **DB Avg** | **30.94** | **0.919** | **0.302** |

Per-scene breakdown:

| Scene | PSNR | SSIM | LPIPS |
|---|---|---|---|
| garden | 28.33 | 0.881 | 0.106 |
| bicycle | 25.88 | 0.788 | 0.216 |
| stump | 27.06 | 0.795 | 0.223 |
| treehill | 23.36 | 0.670 | 0.306 |
| flowers | 22.08 | 0.640 | 0.330 |
| bonsai | 35.73 | 0.964 | 0.192 |
| counter | 31.00 | 0.934 | 0.193 |
| kitchen | 33.59 | 0.945 | 0.123 |
| room | 34.03 | 0.947 | 0.221 |
| truck | 26.91 | 0.900 | 0.112 |
| train | 24.45 | 0.865 | 0.169 |
| drjohnson | 30.43 | 0.918 | 0.309 |
| playroom | 31.45 | 0.921 | 0.296 |

### Table 7 -- High Primitive Count (Per-Scene 3DGS Caps)

```bash
bash benchmarks/nht/benchmark_nht_high.sh
SCENE_LIST="garden bonsai truck" bash benchmarks/nht/benchmark_nht_high.sh
bash benchmarks/nht/benchmark_nht_high.sh --runtime_only
bash benchmarks/nht/benchmark_nht_high.sh --metrics_only
```

### AOV Benchmark (Semantic / LSEG / DINOv3)

> **Experimental:** AOV (arbitrary output variables / semantic heads) is an **experimental** feature and still **work in progress**. 

```bash
# LSEG features
bash benchmarks/nht/benchmark_nht_aov.sh

# DINOv3 features
AOV_TARGET=dinov3 bash benchmarks/nht/benchmark_nht_aov.sh

# Specific scenes
SCENE_LIST="garden bonsai" AOV_TARGET=lseg bash benchmarks/nht/benchmark_nht_aov.sh
```

Training reads **precomputed** maps from disk: LSEG features, DINOv3 features, and optional RGB2X channels (albedo, roughness, etc.). This repository does **not** ship those models or preprocessing pipelines as dependencies—you must **generate (or otherwise obtain) the AOV dataset yourself** before running `benchmark_nht_aov.sh` or `aov/examples/simple_trainer_nht_aov.py`, and lay it out next to your RGB captures as documented in `aov/aov_dataset.py` (expected directory names, file formats, and pointers to external projects you can adapt).

---

## Key NHT-Specific Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--deferred_opt_feature_dim` | `48` | Total feature dimensionality per primitive (divided among 4 tetrahedron vertices) |
| `--deferred_features_lr` | `0.015` | Learning rate for per-primitive features |
| `--deferred_mlp_lr` | `0.00072` | Learning rate for the deferred MLP |
| `--deferred_mlp_hidden_dim` | `128` | Width of each hidden layer in the deferred MLP |
| `--deferred_mlp_num_layers` | `3` | Number of hidden layers |
| `--deferred_mlp_ema` | `True` | Enable EMA on MLP weights (decay=0.95) |
| `--deferred_opt_center_ray_encoding` | `False` | Use per-tile center ray instead of per-pixel ray for view encoding |
| `--deferred_opt_view_encoding_type` | `"sh"` | View encoding: `"sh"` or `"fourier"` |
| `--deferred_opt_sh_degree` | `3` | SH degree for view direction encoding |
| `--deferred_opt_sh_scale` | `3.0` | Scale applied to normalized directions before SH evaluation |
| `--deferred_lr_scheduler` | `"cosine"` | LR schedule: `"cosine"` or `"exponential"` |
| `--color_refine_steps` | `3000` | Steps at end of training where geometry is frozen |
| `--opacity_reg` | `0.02` | Opacity regularization weight |
| `--scale_reg` | `0.01` | Scale regularization weight |
| `--ssim_lambda` | `0.2` | D-SSIM weight in the loss (vs. L1) |
| `--tile_size` | `16` | Rasterization tile size (lower to 8 for large feature_dim) |

---

## Using gsplat's NHT API Directly

```python
from gsplat.nht import DeferredShaderModule, HarmonicFeatures
from gsplat.rendering import rasterization

# Rasterize features + ray directions
renders, alphas, meta = rasterization(
    means, quats, scales, opacities, features,
    viewmats, Ks, width, height,
    nht=True, with_eval3d=True, with_ut=True,
    sh_degree=None,
)
# renders[..., :-3] = encoded features, renders[..., -3:] = ray dirs

# Decode to RGB with the deferred shader
rgb = deferred_shader(renders)
```

### Loading a Checkpoint and Rendering

```python
import torch
from gsplat.rendering import rasterization
from gsplat.nht.deferred_shader import DeferredShaderModule

device = torch.device("cuda:0")
ckpt = torch.load("results/garden/ckpts/ckpt_29999_rank0.pt", map_location=device)
splats = {k: v.to(device) for k, v in ckpt["splats"].items()}

# Restore deferred module
dm_state = ckpt["deferred_module"]
dm = DeferredShaderModule(**dm_state["config"]).to(device)
dm.load_state_dict(dm_state["state_dict"])
if "ema" in dm_state:
    for n, p in dm.named_parameters():
        if n in dm_state["ema"]:
            p.data.copy_(dm_state["ema"][n])
dm.eval()

# Prepare splats
means = splats["means"]
quats = torch.nn.functional.normalize(splats["quats"], p=2, dim=-1)
scales = torch.exp(splats["scales"])
opacities = torch.sigmoid(splats["opacities"])
features = splats["features"].half()

# Rasterize
with torch.no_grad():
    render_colors, render_alphas, info = rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opacities, colors=features,
        viewmats=viewmat[None], Ks=K[None],
        width=W, height=H,
        nht=True, with_eval3d=True, with_ut=True,
        sh_degree=None,
        center_ray_mode=dm.center_ray_encoding,
        ray_dir_scale=dm.ray_dir_scale,
    )
    rgb, extras = dm(render_colors)
    rgb = rgb[0].clamp(0, 1)
```

---

## Paper Setup

All paper results were measured on an **NVIDIA RTX A6000 Ada** (48 GB, Ada Lovelace architecture).

> **Note on LPIPS**: Uses VGG backbone with inputs normalized to [-1, 1]. This differs from INRIA 3DGS, which does not normalize inputs.

---

## Citation

```bibtex
@article{condor2026nht,
  title={Neural Harmonic Textures for High-Quality Primitive Based Neural Reconstruction},
  author={Condor, Jorge and Moenne-Loccoz, Nicolas and Nimier-David, Merlin and Didyk, Piotr and Gojcic, Zan and Wu, Qi},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0. See the gsplat submodule for its own license terms.
