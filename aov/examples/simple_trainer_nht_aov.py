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

# Superproject: this script lives under nht-release/aov/examples.
import sys
from pathlib import Path

_NHT_REPO_ROOT = Path(__file__).resolve().parents[2]
_GSPLAT_EXAMPLES = _NHT_REPO_ROOT / "gsplat" / "examples"
for _p in (_GSPLAT_EXAMPLES, _NHT_REPO_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from aov.aov_dataset import AOVDataset, AOVDatasetConfig, RGB2X_DEFAULT_CHANNEL_DIMS
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal
from utils import CameraOptModule, knn, set_random_seed

from gsplat.nht.exporter import export_splats_nht
from gsplat.color_correct import color_correct_affine, color_correct_quadratic
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.nht.strategy import NHTMCMCStrategy as MCMCStrategy
from gsplat_viewer_nht import GsplatNHTViewer, GsplatNHTRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap
## NHT ##
from gsplat.nht.deferred_shader import HarmonicFeatures
from aov.deferred_shader import DeferredShaderAOVModule

## TODO: this is expensive. We should think of creating a custom fused CUDA kernel for this.
def _semantic_loss(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """L1 + cosine distance on semantic features."""
    feat_l1 = F.l1_loss(pred, gt)
    feat_cos = 1.0 - F.cosine_similarity(pred, gt, dim=-1).mean()
    return feat_l1, feat_cos


def total_variation_loss_features(feat: Tensor) -> Tensor:
    """Spatial total variation on feature maps [B, H, W, C]."""
    diff_h = (feat[:, 1:, :, :] - feat[:, :-1, :, :]).abs().mean()
    diff_w = (feat[:, :, 1:, :] - feat[:, :, :-1, :]).abs().mean()
    return diff_h + diff_w


class PCAVisualizer:
    """PCA projection of high-dim features to RGB for visualization."""

    def __init__(self):
        self._fitted = False
        self._components = None
        self._mean = None
        self._min = None
        self._max = None

    def fit(self, features: Tensor):
        """Fit PCA on features [N, D]. Call once on first frame."""
        features = features.float()
        self._mean = features.mean(dim=0)
        centered = features - self._mean
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        self._components = Vt[:3]  # [3, D]
        projected = centered @ self._components.T  # [N, 3]
        self._min = projected.min(dim=0).values
        self._max = projected.max(dim=0).values
        self._fitted = True

    def transform(self, features: Tensor) -> Tensor:
        """Project features [H, W, D] to [H, W, 3] RGB via top-3 PCs."""
        H, W, D = features.shape
        flat = features.reshape(-1, D).float()
        centered = flat - self._mean
        projected = centered @ self._components.T  # [N, 3]
        rng = (self._max - self._min).clamp(min=1e-6)
        projected = (projected - self._min) / rng
        return projected.clamp(0, 1).reshape(H, W, 3)

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    # Load EXIF exposure metadata from images (if available)
    load_exposure: bool = True

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # Number of DataLoader workers
    num_workers: int = 4
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 0
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.5
    # Initial scale of GS
    init_scale: float = 0.1
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification (NHT only supports MCMC)
    strategy: MCMCStrategy = field(
        default_factory=MCMCStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 4e-3
    # LR for alpha blending weights
    opacities_lr: float = 4e-2
    # LR for orientation (quaternions)
    quats_lr: float = 8e-4
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.02
    # Scale regularization
    scale_reg: float = 0.01

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Post-processing method for appearance correction (experimental)
    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = None
    # Use fused implementation for bilateral grid (only applies when post_processing="bilateral_grid")
    bilateral_grid_fused: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # Enable PPISP controller
    ppisp_use_controller: bool = True
    # Use controller distillation in PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_distillation: bool = True
    # Controller activation ratio for PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_activation_num_steps: int = 25_000
    # Color correction method for cc_* metrics (only applies when post_processing is set)
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    # Compute color-corrected metrics (cc_psnr, cc_ssim, cc_lpips) during evaluation
    use_color_correction_metric: bool = False

    # Enable camera pose optimization (experimental)
    pose_opt: bool = False
    # Learning rate for camera pose optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera pose optimization
    pose_opt_reg: float = 1e-6
    # Camera pose noise for evaluation (experimental)
    pose_noise: float = 0.0

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Number of evaluation timing passes for benchmarking
    eval_timing_passes: int = 1

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    # LPIPS backbone: "vgg" or "alex".
    lpips_net: Literal["vgg", "alex"] = "vgg"
    # Normalize LPIPS inputs to [-1, 1]. True is the correct usage; INRIA 3DGS
    # uses normalize=False with VGG which is technically incorrect.
    lpips_normalize: bool = True

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = True
    with_eval3d: bool = True

    ##### NHT OPTIONS #####
    # Tile size for rasterization. Use 8 for large feature_dim (>48).
    tile_size: int = 8
    primitive_type: str = "3dgs"
    # Color refinement steps at the end of training. Freezes geometry and opacities.
    color_refine_steps: int = 3000
    # Enable deferred shading.
    deferred_opt: bool = True
    # Weight decay regularization for deferred MLP optimizer
    deferred_opt_reg: float = 0.0
    # Enable view encoding for deferred shading
    deferred_opt_enable_view_encoding: bool = True
    # View encoding type: "sh" for spherical harmonics, "fourier" for Fourier features
    deferred_opt_view_encoding_type: Literal["sh", "fourier"] = "sh"
    # SH degree for view direction encoding (only used when view_encoding_type="sh")
    deferred_opt_sh_degree: int = 3
    # Scale applied to normalized dirs before SH evaluation (only for "sh")
    deferred_opt_sh_scale: float = 3.0
    # Number of Fourier frequency levels L (only used when view_encoding_type="fourier")
    deferred_opt_fourier_num_freqs: int = 4
    # Use a single center ray for view encoding instead of per-pixel rays.
    deferred_opt_center_ray_encoding: bool = False
    # Deferred shading feature embedding dimension
    deferred_opt_feature_dim: int = 64
    # Learning rate for deferred features (per-Gaussian features in splats)
    deferred_features_lr: float = 15e-3
    # Learning rate for deferred MLP
    deferred_mlp_lr: float = 72e-5
    # Enable exponential decay for deferred features LR (ends at deferred_features_lr_decay_final of initial value)
    deferred_features_lr_decay: bool = True
    # Final multiplier for deferred features LR decay (e.g., 0.01 means LR ends at 1% of initial)
    deferred_features_lr_decay_final: float = 0.1
    # Enable exponential decay for deferred MLP LR (ends at deferred_mlp_lr_decay_final of initial value)
    deferred_mlp_lr_decay: bool = True
    # Final multiplier for deferred MLP LR decay (e.g., 0.01 means LR ends at 1% of initial)
    deferred_mlp_lr_decay_final: float = 0.1
    # LR scheduler type for deferred features and MLP: "exponential" or "cosine"
    deferred_lr_scheduler: Literal["exponential", "cosine"] = "cosine"
    # Use the same LR decay as vertex positions (0.01 factor) for both deferred features and MLP
    # When enabled, this overrides deferred_features_lr_decay_final and deferred_mlp_lr_decay_final
    deferred_use_means_lr_decay: bool = False
    # Initial feature values range (random uniform between min and max)
    deferred_features_init_min: float = -np.pi / 2.0
    deferred_features_init_max: float = np.pi / 2.0
    # Hidden dimension for deferred MLP
    deferred_mlp_hidden_dim: int = 128
    # Number of hidden layers for deferred MLP
    deferred_mlp_num_layers: int = 3
    # Activation for non-MLP decode modes: "sigmoid" or "relu_clamp"
    deferred_decode_activation: Literal["sigmoid", "relu_clamp"] = "sigmoid"
    # EMA (Exponential Moving Average) for deferred MLP weights
    deferred_mlp_ema: bool = True
    # EMA decay rate (higher = smoother, typical values: 0.99 to 0.9999)
    deferred_mlp_ema_decay: float = 0.95
    # Step to start EMA updates (allow MLP to warm up first)
    deferred_mlp_ema_start_step: int = 0

    ##### AOV OPTIONS #####
    # AOV maps (LSEG / DINOv3 / RGB2X) must be precomputed on disk; see
    # aov.aov_dataset for expected layout and external tool links.

    # Enable LSEG feature training
    lseg_data: bool = False
    # Override LSEG data directory (auto-detected from image_dir + "_bin_lseg" if None)
    lseg_data_dir: Optional[str] = None
    # Weight for LSEG L1 loss
    lseg_loss_lambda: float = 1.0
    # Weight for LSEG cosine similarity loss
    lseg_cosine_lambda: float = 0.1
    # Weight for LSEG total variation loss
    lseg_tv_lambda: float = 0.0

    # Enable DINOv3 feature training
    dinov3_data: bool = False
    # Override DINOv3 data directory (auto-detected from image_dir + "_dinov3" if None)
    dinov3_data_dir: Optional[str] = None
    # Weight for DINOv3 L1 loss
    dinov3_loss_lambda: float = 1.0
    # Weight for DINOv3 cosine similarity loss
    dinov3_cosine_lambda: float = 0.1
    # Weight for DINOv3 total variation loss
    dinov3_tv_lambda: float = 0.0

    # Enable RGB2X material map training
    rgb2x_data: bool = False
    # Override RGB2X data directory base (auto-detected from image_dir if None)
    rgb2x_data_dir: Optional[str] = None
    # Global weight for all RGB2X losses
    rgb2x_loss_lambda: float = 1.0
    # Per-channel loss weights
    rgb2x_albedo_lambda: float = 1.0
    rgb2x_roughness_lambda: float = 1.0
    rgb2x_metallic_lambda: float = 1.0
    rgb2x_irradiance_lambda: float = 1.0
    rgb2x_normal_lambda: float = 1.0
    # Weight for normal-specific cosine loss
    rgb2x_normal_cosine_lambda: float = 0.1
    # Weight for per-channel total variation loss
    rgb2x_tv_lambda: float = 0.0

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.color_refine_steps = int(self.color_refine_steps * factor)

        strategy = self.strategy
        strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
        strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
        strategy.refine_every = int(strategy.refine_every * factor)

def create_splats_with_optimizers(
    parser: Parser,
    feature_module: HarmonicFeatures,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    scene_scale: float = 1.0,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    cap_max: Optional[int] = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        if cap_max is not None and points.shape[0] > cap_max:
            keep = torch.randperm(points.shape[0])[:cap_max]
            points = points[keep]
            rgbs = rgbs[keep]
    elif init_type == "random":
        n_pts = min(init_num_pts, cap_max) if cap_max is not None else init_num_pts
        points = init_extent * scene_scale * (torch.rand((n_pts, 3)) * 2 - 1)
        rgbs = torch.rand((n_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # For sfm/random: initialize scales from KNN distances, random quats
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    assert N == rgbs.shape[0]

    quats = torch.rand((N, 4))  # [N, 4]
    quats = quats[world_rank::world_size]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]
    feature_module.init_features(params, rgbs)

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
        for name, _, lr in params
    }
    return splats, optimizers

class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            load_exposure=cfg.load_exposure,
        )
        base_trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        base_valset = Dataset(self.parser, split="val")

        # AOV dataset wrapping
        self.has_aov = cfg.lseg_data or cfg.dinov3_data or cfg.rgb2x_data
        if self.has_aov:
            aov_config = AOVDatasetConfig(
                load_lseg=cfg.lseg_data,
                lseg_data_dir=cfg.lseg_data_dir,
                load_dinov3=cfg.dinov3_data,
                dinov3_data_dir=cfg.dinov3_data_dir,
                load_rgb2x=cfg.rgb2x_data,
                rgb2x_data_dir=cfg.rgb2x_data_dir,
            )
            self.trainset = AOVDataset(base_trainset, aov_config, self.parser)
            self.valset = AOVDataset(base_valset, aov_config, self.parser)
            self.aov_detected_config = self.trainset.get_aov_config()
            print(f"AOV config detected: {self.aov_detected_config}")
        else:
            self.trainset = base_trainset
            self.valset = base_valset
            self.aov_detected_config = {
                "lseg_feature_dim": 0,
                "dinov3_feature_dim": 0,
                "rgb2x_channels": None,
            }

        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        if self.parser.num_cameras > 1 and cfg.batch_size != 1:
            raise ValueError(
                f"When using multiple cameras ({self.parser.num_cameras} found), batch_size must be 1, "
                f"but got batch_size={cfg.batch_size}."
            )
        if cfg.post_processing == "ppisp" and cfg.batch_size != 1:
            raise ValueError(
                f"PPISP post-processing requires batch_size=1, got batch_size={cfg.batch_size}"
            )
        if cfg.post_processing is not None and world_size > 1:
            raise ValueError(
                f"Post-processing ({cfg.post_processing}) requires single-GPU training, "
                f"but world_size={world_size}."
            )
        if not isinstance(cfg.strategy, MCMCStrategy):
            raise ValueError(
                "NHT training requires MCMCStrategy. DefaultStrategy is not supported "
            )

        deferred_feature_dim = cfg.deferred_opt_feature_dim
        self.feature_module = HarmonicFeatures(
            feature_dim=deferred_feature_dim,
            feature_lr=cfg.deferred_features_lr,
            features_init_min=cfg.deferred_features_init_min,
            features_init_max=cfg.deferred_features_init_max,
        )

        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            self.feature_module,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            scene_scale=self.scene_scale,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            cap_max=cfg.strategy.cap_max,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.cfg.strategy.initialize_state()

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        # Pose Optimization
        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        # Deferred Shading
        self.deferred_optimizers = []
        self._hashgrid_grad_scale = None
        deferred_feature_dim = cfg.deferred_opt_feature_dim
        self.deferred_module = DeferredShaderAOVModule(
            feature_dim=deferred_feature_dim,
            enable_view_encoding=cfg.deferred_opt_enable_view_encoding,
            view_encoding_type=cfg.deferred_opt_view_encoding_type,
            mlp_hidden_dim=cfg.deferred_mlp_hidden_dim,
            mlp_num_layers=cfg.deferred_mlp_num_layers,
            sh_degree=cfg.deferred_opt_sh_degree,
            sh_scale=cfg.deferred_opt_sh_scale,
            fourier_num_freqs=cfg.deferred_opt_fourier_num_freqs,
            primitive_type=cfg.primitive_type,
            center_ray_encoding=cfg.deferred_opt_center_ray_encoding,
            lseg_feature_dim=self.aov_detected_config["lseg_feature_dim"],
            dinov3_feature_dim=self.aov_detected_config["dinov3_feature_dim"],
            rgb2x_channels=self.aov_detected_config["rgb2x_channels"],
        ).to(self.device)
        self.deferred_optimizers = [
            torch.optim.Adam(
                self.deferred_module.parameters(),
                lr=cfg.deferred_mlp_lr * math.sqrt(cfg.batch_size),
                weight_decay=cfg.deferred_opt_reg,
            ),
        ]
        if world_size > 1:
            self.deferred_module = DDP(self.deferred_module)

        # EMA shadow weights for deferred MLP (TCNN loss_scale already defaults to 128)
        self._ema_shadow = None
        if cfg.deferred_mlp_ema:
            dm = self.deferred_module.module if world_size > 1 else self.deferred_module
            self._ema_shadow = {n: p.data.clone() for n, p in dm.named_parameters()}

        self.post_processing_module = None
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_module = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
        elif cfg.post_processing == "ppisp":
            ppisp_config = PPISPConfig(
                use_controller=cfg.ppisp_use_controller,
                controller_distillation=cfg.ppisp_controller_distillation,
                controller_activation_ratio=cfg.ppisp_controller_activation_num_steps
                / cfg.max_steps,
            )
            self.post_processing_module = PPISP(
                num_cameras=self.parser.num_cameras,
                num_frames=len(self.trainset),
                config=ppisp_config,
            ).to(self.device)

        self.post_processing_optimizers = []
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_optimizers = [
                torch.optim.Adam(
                    self.post_processing_module.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]
        elif cfg.post_processing == "ppisp":
            self.post_processing_optimizers = (
                self.post_processing_module.create_optimizers()
            )

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=cfg.lpips_net, normalize=cfg.lpips_normalize
        ).to(self.device)

        self.lpips_alex = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatNHTViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        # PCA visualizers for high-dim feature AOVs
        self.lseg_pca = PCAVisualizer() if cfg.lseg_data else None
        self.dinov3_pca = PCAVisualizer() if cfg.dinov3_data else None

        # Track if Gaussians are frozen (for controller distillation)
        self._gaussians_frozen = False

    def freeze_gaussians(self):
        """Freeze all Gaussian parameters for controller distillation.

        This prevents Gaussians from being updated by any loss (including regularization)
        while the controller learns to predict per-frame corrections.
        """
        if self._gaussians_frozen:
            return

        for name, param in self.splats.items():
            param.requires_grad = False

        self._gaussians_frozen = True
        print("[Distillation] Gaussian parameters frozen")

    def _deferred_mod(self):
        return self.deferred_module.module if self.world_size > 1 else self.deferred_module

    @torch.no_grad()
    def _update_ema(self, step: int):
        if self._ema_shadow is None or step < self.cfg.deferred_mlp_ema_start_step:
            return
        decay = self.cfg.deferred_mlp_ema_decay
        for n, p in self._deferred_mod().named_parameters():
            self._ema_shadow[n].lerp_(p.data, 1.0 - decay)

    @torch.no_grad()
    def _apply_ema(self):
        if self._ema_shadow is None:
            return
        dm = self._deferred_mod()
        self._ema_saved = {n: p.data.clone() for n, p in dm.named_parameters()}
        for n, p in dm.named_parameters():
            p.data.copy_(self._ema_shadow[n])

    @torch.no_grad()
    def _restore_from_ema(self):
        if self._ema_shadow is None or not hasattr(self, "_ema_saved"):
            return
        for n, p in self._deferred_mod().named_parameters():
            p.data.copy_(self._ema_saved[n])
        self._ema_saved = None

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        frame_idcs: Optional[Tensor] = None,
        camera_idcs: Optional[Tensor] = None,
        exposure: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:

        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])

        image_ids = kwargs.pop("image_ids", None)
        kwargs.pop("sh_degree", None)
        render_mode = kwargs.pop("render_mode", "RGB")
        if render_mode in ["D", "ED"]:
            render_mode = "RGB+" + render_mode
        kwargs.pop("backgrounds", None)

        colors = self.splats["features"]
        sh_degree = None

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        if camera_model is None:
            camera_model = self.cfg.camera_model

        use_eval3d = self.cfg.with_eval3d
        use_ut = self.cfg.with_ut
        render_colors, render_alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                sh_degree=sh_degree,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                tile_size=self.cfg.tile_size,
                packed=self.cfg.packed,
                absgrad=False,
                sparse_grad=self.cfg.sparse_grad,
                rasterize_mode=rasterize_mode,
                render_mode=render_mode,
                distributed=self.world_size > 1,
                camera_model=self.cfg.camera_model,
                with_ut=use_ut,
                with_eval3d=use_eval3d,
                nht=True,
                center_ray_mode=self.cfg.deferred_opt_center_ray_encoding,
                ray_dir_scale=self._deferred_mod().ray_dir_scale,
                **kwargs,
            )

        render_colors, aov_outputs, extras = self.deferred_module(render_colors)
        info["aov_outputs"] = aov_outputs
        if extras is not None:
            render_colors = torch.cat([render_colors, extras], dim=-1)

        if masks is not None:
            render_colors[~masks] = 0

        if self.cfg.post_processing is not None:
            # Create pixel coordinates [H, W, 2] with +0.5 center offset
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(height, device=self.device) + 0.5,
                torch.arange(width, device=self.device) + 0.5,
                indexing="ij",
            )
            pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1)  # [H, W, 2]

            # Split RGB from extra channels (e.g. depth) for post-processing
            rgb = render_colors[..., :3]
            extra = render_colors[..., 3:] if render_colors.shape[-1] > 3 else None

            if self.cfg.post_processing == "bilateral_grid":
                if frame_idcs is not None:
                    grid_xy = (
                        pixel_coords / torch.tensor([width, height], device=self.device)
                    ).unsqueeze(0)
                    rgb = slice(
                        self.post_processing_module,
                        grid_xy.expand(rgb.shape[0], -1, -1, -1),
                        rgb,
                        frame_idcs.unsqueeze(-1),
                    )["rgb"]
            elif self.cfg.post_processing == "ppisp":
                camera_idx = camera_idcs.item() if camera_idcs is not None else None
                frame_idx = frame_idcs.item() if frame_idcs is not None else None
                rgb = self.post_processing_module(
                    rgb=rgb,
                    pixel_coords=pixel_coords,
                    resolution=(width, height),
                    camera_idx=camera_idx,
                    frame_idx=frame_idx,
                    exposure_prior=exposure,
                )

            render_colors = (
                torch.cat([rgb, extra], dim=-1) if extra is not None else rgb
            )

        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.deferred_features_lr_decay:
            # deferred features have a learning rate schedule
            features_decay_final = cfg.deferred_features_lr_decay_final
            if cfg.deferred_lr_scheduler == "cosine":
                schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizers["features"],
                        T_max=max_steps,
                        eta_min=cfg.deferred_features_lr * features_decay_final,
                    )
                )
            else:
                schedulers.append(
                    torch.optim.lr_scheduler.ExponentialLR(
                        self.optimizers["features"],
                        gamma=features_decay_final ** (1.0 / max_steps),
                    )
                )
        if  cfg.deferred_mlp_lr_decay:
            # deferred MLP has a learning rate schedule
            mlp_decay_final = cfg.deferred_mlp_lr_decay_final
            if cfg.deferred_lr_scheduler == "cosine":
                schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.deferred_optimizers[0],
                        T_max=max_steps,
                        eta_min=cfg.deferred_mlp_lr * mlp_decay_final,
                    )
                )
            else:
                schedulers.append(
                    torch.optim.lr_scheduler.ExponentialLR(
                        self.deferred_optimizers[0],
                        gamma=mlp_decay_final ** (1.0 / max_steps),
                    )
                )
        # Post-processing module has a learning rate schedule
        if cfg.post_processing == "bilateral_grid":
            # Linear warmup + exponential decay
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.post_processing_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.post_processing_optimizers[0],
                            gamma=0.01 ** (1.0 / max_steps),
                        ),
                    ]
                )
            )
        elif cfg.post_processing == "ppisp":
            ppisp_schedulers = self.post_processing_module.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=max_steps,
            )
            schedulers.extend(ppisp_schedulers)

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Color refinement phase: step at which geometry freezes and color-only
        # optimization begins. Set to max_steps if disabled (no refinement).
        color_refine_start = (
            max_steps - cfg.color_refine_steps
            if cfg.color_refine_steps > 0
            else max_steps
        )
        in_color_refine = False

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()
            
            # Freeze Gaussians when PPISP controller distillation starts
            if (
                cfg.post_processing == "ppisp"
                and cfg.ppisp_use_controller
                and cfg.ppisp_controller_distillation
                and step >= cfg.ppisp_controller_activation_num_steps
            ):
                self.freeze_gaussians()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            exposure = (
                data["exposure"].to(device) if "exposure" in data else None
            )  # [B,]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            # --- Color refinement phase transition ---
            if not in_color_refine and step >= color_refine_start:
                in_color_refine = True
                frozen_keys = {"means", "scales", "quats", "opacities"}
                for name in frozen_keys:
                    if name in self.optimizers:
                        for pg in self.optimizers[name].param_groups:
                            pg["lr"] = 0.0
                print(
                    f"\n[step {step}] Entering color refinement phase: "
                    f"freezing geometry + opacities, "
                    f"disabling scale/opacity regularization."
                )

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
                frame_idcs=image_ids,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            # AOV losses
            aov_outputs = info.get("aov_outputs", {})
            aov_losses = {}

            for mode_name, gt_key, pred_key, l1_lam, cos_lam, tv_lam in [
                ("lseg", "lseg_features", "lseg",
                 cfg.lseg_loss_lambda, cfg.lseg_cosine_lambda, cfg.lseg_tv_lambda),
                ("dinov3", "dinov3_features", "dinov3",
                 cfg.dinov3_loss_lambda, cfg.dinov3_cosine_lambda, cfg.dinov3_tv_lambda),
            ]:
                if gt_key not in data or pred_key not in aov_outputs:
                    continue
                aov_gt = data[gt_key].to(device).float()
                aov_pred = aov_outputs[pred_key]
                if aov_pred.shape[1:3] != aov_gt.shape[1:3]:
                    aov_pred = F.interpolate(
                        aov_pred.permute(0, 3, 1, 2),
                        size=(aov_gt.shape[1], aov_gt.shape[2]),
                        mode="bilinear", align_corners=False,
                    ).permute(0, 2, 3, 1)
                if l1_lam > 0 and cos_lam > 0:
                    feat_l1, feat_cos = _semantic_loss(aov_pred, aov_gt)
                    loss = loss + l1_lam * feat_l1 + cos_lam * feat_cos
                    aov_losses[f"{mode_name}_l1"] = feat_l1
                    aov_losses[f"{mode_name}_cosine"] = feat_cos
                elif l1_lam > 0:
                    feat_l1 = F.l1_loss(aov_pred, aov_gt)
                    loss = loss + l1_lam * feat_l1
                    aov_losses[f"{mode_name}_l1"] = feat_l1
                elif cos_lam > 0:
                    feat_cos = 1.0 - F.cosine_similarity(aov_pred, aov_gt, dim=-1).mean()
                    loss = loss + cos_lam * feat_cos
                    aov_losses[f"{mode_name}_cosine"] = feat_cos
                if tv_lam > 0:
                    feat_tv = total_variation_loss_features(aov_pred)
                    loss = loss + tv_lam * feat_tv
                    aov_losses[f"{mode_name}_tv"] = feat_tv

                # Same definitions as validation stats (lseg_psnr / lseg_cosine in JSON).
                with torch.no_grad():
                    aov_mse = F.mse_loss(aov_pred, aov_gt)
                    data_range = aov_gt.max() - aov_gt.min()
                    if data_range > 0:
                        aov_losses[f"{mode_name}_psnr"] = 10 * torch.log10(
                            data_range.pow(2) / aov_mse.clamp(min=1e-10)
                        )
                    aov_losses[f"{mode_name}_cos_sim"] = F.cosine_similarity(
                        aov_pred, aov_gt, dim=-1
                    ).mean()

            if cfg.rgb2x_data and "rgb2x" in data and "rgb2x" in aov_outputs:
                rgb2x_gt = data["rgb2x"].to(device) / 255.0
                rgb2x_pred = aov_outputs["rgb2x"]
                if rgb2x_pred.shape[1:3] != rgb2x_gt.shape[1:3]:
                    rgb2x_pred = F.interpolate(
                        rgb2x_pred.permute(0, 3, 1, 2),
                        size=(rgb2x_gt.shape[1], rgb2x_gt.shape[2]),
                        mode="bilinear", align_corners=False,
                    ).permute(0, 2, 3, 1)

                if not hasattr(self, "_rgb2x_weight_vec"):
                    ch_weights: list = []
                    for ch_name in sorted(self.aov_detected_config["rgb2x_channels"].keys()):
                        dim = self.aov_detected_config["rgb2x_channels"][ch_name]
                        w = getattr(cfg, f"rgb2x_{ch_name}_lambda", 1.0)
                        ch_weights.extend([w] * dim)
                    wv = torch.tensor(ch_weights, device=device)
                    self._rgb2x_weight_vec = wv / wv.mean()

                rgb2x_l1 = ((rgb2x_pred - rgb2x_gt).abs() * self._rgb2x_weight_vec).mean()
                loss = loss + cfg.rgb2x_loss_lambda * rgb2x_l1
                aov_losses["rgb2x_l1"] = rgb2x_l1

                if "normal" in self.aov_detected_config["rgb2x_channels"] and cfg.rgb2x_normal_cosine_lambda > 0:
                    layout = self._deferred_mod().rgb2x_channel_layout
                    offset = sum(d for n, d in layout if n < "normal")
                    n_dim = self.aov_detected_config["rgb2x_channels"]["normal"]
                    pred_n = rgb2x_pred[..., offset:offset + n_dim] * 2.0 - 1.0
                    gt_n = rgb2x_gt[..., offset:offset + n_dim] * 2.0 - 1.0
                    cos_loss = 1.0 - F.cosine_similarity(pred_n, gt_n, dim=-1).mean()
                    loss = loss + cfg.rgb2x_loss_lambda * cfg.rgb2x_normal_cosine_lambda * cos_loss
                    aov_losses["nor_cos"] = cos_loss

                if cfg.rgb2x_tv_lambda > 0:
                    rgb2x_tv = total_variation_loss_features(rgb2x_pred)
                    loss = loss + cfg.rgb2x_loss_lambda * cfg.rgb2x_tv_lambda * rgb2x_tv
                    aov_losses["rgb2x_tv"] = rgb2x_tv

            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            if cfg.post_processing == "bilateral_grid":
                post_processing_reg_loss = 10 * total_variation_loss(
                    self.post_processing_module.grids
                )
                loss += post_processing_reg_loss
            elif cfg.post_processing == "ppisp":
                post_processing_reg_loss = (
                    self.post_processing_module.get_regularization_loss()
                )
                loss += post_processing_reg_loss

            # regularizations (disabled during color refinement phase)
            if cfg.opacity_reg > 0.0 and not in_color_refine:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0 and not in_color_refine:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            loss.backward()

            desc = f"loss={loss.item():.3f}| "
            if cfg.depth_loss:
                desc += f"depth={depthloss.item():.6f}| "
            if aov_losses:
                for ak, av in aov_losses.items():
                    short = ak.replace("rgb2x_", "").replace("_cosine", "_cos")
                    if short.startswith(("normal", "albedo", "roughness", "metallic", "irradiance")):
                        short = short[:3] + short[short.index("_"):]
                    desc += f"{short}={av.item():.4f}| "
            if cfg.pose_opt and cfg.pose_noise:
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose={pose_err.item():.6f}| "
            desc += f"#GS={len(self.splats['means'])}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.color_refine_steps > 0:
                    self.writer.add_scalar(
                        "train/color_refine", float(in_color_refine), step
                    )
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.post_processing is not None:
                    self.writer.add_scalar(
                        "train/post_processing_reg_loss",
                        post_processing_reg_loss.item(),
                        step,
                    )
                if self.has_aov:
                    for loss_name, loss_val in aov_losses.items():
                        self.writer.add_scalar(
                            f"train/aov_{loss_name}", loss_val.item(), step
                        )
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                data["aov_config"] = self.aov_detected_config
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if world_size > 1:
                    data["deferred_module"] = (
                        self.deferred_module.module.state_dict()
                    )
                else:
                    data["deferred_module"] = self.deferred_module.state_dict()
                if self._ema_shadow is not None:
                    data["deferred_ema"] = self._ema_shadow
                if self.post_processing_module is not None:
                    data["post_processing"] = self.post_processing_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                dm = self._deferred_mod()
                deferred_state = {
                    "state_dict": dm.state_dict(),
                    "config": {
                        "feature_dim": dm.feature_dim,
                        "enable_view_encoding": dm.enable_view_encoding,
                        "view_encoding_type": dm.view_encoding_type,
                        "mlp_hidden_dim": dm.mlp_hidden_dim,
                        "mlp_num_layers": dm.mlp_num_layers,
                        "sh_degree": dm.view_sh_degree,
                        "sh_scale": dm.sh_scale,
                        "fourier_num_freqs": dm.fourier_num_freqs,
                        "primitive_type": dm.primitive_type,
                        "center_ray_encoding": dm.center_ray_encoding,
                    },
                }
                if self._ema_shadow is not None:
                    deferred_state["ema"] = self._ema_shadow
                export_splats_nht(
                    means=self.splats["means"],
                    scales=self.splats["scales"],
                    quats=self.splats["quats"],
                    opacities=self.splats["opacities"],
                    features=self.splats["features"],
                    deferred_module=deferred_state,
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.deferred_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            self._update_ema(step)
            for optimizer in self.post_processing_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()
            if in_color_refine:
                for name in ("means", "scales", "quats", "opacities", "betas_raw"):
                    if name in self.optimizers:
                        for pg in self.optimizers[name].param_groups:
                            pg["lr"] = 0.0

            self.cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                lr=0.0 if in_color_refine else schedulers[0].get_last_lr()[0],
            )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        cfg = self.cfg
        print(f"Running evaluation (stage={stage})...")
        self._apply_ema()
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )

        # Warmup: several dry runs on first frame to trigger JIT compilation
        # (e.g., tinycudann fused kernels) and stabilize GPU clocks/caches.
        NUM_WARMUP = 5
        warmup_data = next(iter(valloader))
        for _ in range(NUM_WARMUP):
            self.rasterize_splats(
                camtoworlds=warmup_data["camtoworld"].to(device),
                Ks=warmup_data["K"].to(device),
                width=warmup_data["image"].shape[2],
                height=warmup_data["image"].shape[1],
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )
            torch.cuda.synchronize()

        # -- Multi-pass render, save, and compute metrics per frame ---------
        # Runs the full validation set `eval_timing_passes` times. Timing is
        # accumulated across all passes; metrics and image saves only happen on
        # the final pass. 
        ellipse_time = 0
        timing_accum = defaultdict(float)
        num_timed = 0
        metrics = defaultdict(list)

        num_passes = max(1, cfg.eval_timing_passes)
        for pass_idx in range(num_passes):
            is_last_pass = (pass_idx == num_passes - 1)
            desc = f"Eval pass {pass_idx+1}/{num_passes}" if num_passes > 1 else "Eval"

            for i, data in enumerate(tqdm.tqdm(valloader, desc=desc)):
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                pixels = data["image"].to(device) / 255.0
                masks = data["mask"].to(device) if "mask" in data else None
                height, width = pixels.shape[1:3]

                # Exposure metadata is available for any image with EXIF data (train or val)
                exposure = data["exposure"].to(device) if "exposure" in data else None

                torch.cuda.synchronize()
                tic = time.time()
                colors, _, eval_info = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    masks=masks,
                    frame_idcs=None,  # For novel views, pass None (no per-frame parameters available)
                    camera_idcs=data["camera_idx"].to(device),
                    exposure=exposure,
                )  # [1, H, W, 3]
                torch.cuda.synchronize()
                ellipse_time += max(time.time() - tic, 1e-10)

                if not is_last_pass:
                    continue

                colors = torch.clamp(colors, 0.0, 1.0)
                canvas_list = [pixels, colors]
                eval_aov_outputs = eval_info.get("aov_outputs", {})

                if world_rank == 0:
                    # write images
                    canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                    canvas = (canvas * 255).astype(np.uint8)
                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                        canvas,
                    )

                    pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                    metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                    metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                    metrics["lpips_alex"].append(self.lpips_alex(colors_p, pixels_p))
                    # Compute color-corrected metrics for fair comparison across methods
                    if cfg.use_color_correction_metric:
                        if cfg.color_correct_method == "affine":
                            cc_colors = color_correct_affine(colors, pixels)
                        else:
                            cc_colors = color_correct_quadratic(colors, pixels)
                        cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                        metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                        metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                        metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

                    # AOV eval metrics
                    if self.has_aov:
                        for mode_name, gt_key, pca_vis in [
                            ("lseg", "lseg_features", self.lseg_pca),
                            ("dinov3", "dinov3_features", self.dinov3_pca),
                        ]:
                            if mode_name not in eval_aov_outputs or gt_key not in data:
                                continue
                            aov_pred = eval_aov_outputs[mode_name]
                            aov_gt = data[gt_key].to(device).float()
                            if aov_pred.shape[1:3] != aov_gt.shape[1:3]:
                                aov_pred = F.interpolate(
                                    aov_pred.permute(0, 3, 1, 2),
                                    size=(aov_gt.shape[1], aov_gt.shape[2]),
                                    mode="bilinear", align_corners=False,
                                ).permute(0, 2, 3, 1)
                            aov_mse = F.mse_loss(aov_pred, aov_gt)
                            data_range = aov_gt.max() - aov_gt.min()
                            if data_range > 0:
                                metrics[f"{mode_name}_psnr"].append(
                                    10 * torch.log10(data_range ** 2 / aov_mse.clamp(min=1e-10))
                                )
                            metrics[f"{mode_name}_cosine"].append(
                                F.cosine_similarity(aov_pred, aov_gt, dim=-1).mean()
                            )
                            if pca_vis is not None:
                                if not pca_vis._fitted:
                                    pca_vis.fit(aov_gt[0].reshape(-1, aov_gt.shape[-1]))
                                gt_vis = pca_vis.transform(aov_gt[0])
                                pred_vis = pca_vis.transform(aov_pred[0])
                                aov_canvas = torch.cat([gt_vis, pred_vis], dim=1)
                                aov_canvas = (aov_canvas.cpu().numpy() * 255).astype(np.uint8)
                                imageio.imwrite(
                                    f"{self.render_dir}/{stage}_step{step}_{i:04d}_{mode_name}.png",
                                    aov_canvas,
                                )

                        if "rgb2x" in eval_aov_outputs and "rgb2x" in data:
                            dm = self._deferred_mod()
                            rgb2x_split = dm.split_rgb2x(eval_aov_outputs["rgb2x"])
                            rgb2x_gt_all = data["rgb2x"].to(device) / 255.0
                            gt_split = dm.split_rgb2x(rgb2x_gt_all)
                            for ch_name in sorted(self.aov_detected_config.get("rgb2x_channels", {}).keys()):
                                pred_key = f"rgb2x_{ch_name}"
                                if pred_key not in rgb2x_split or pred_key not in gt_split:
                                    continue
                                ch_pred = rgb2x_split[pred_key]
                                ch_gt = gt_split[pred_key]
                                ch_mse = F.mse_loss(ch_pred, ch_gt)
                                metrics[f"rgb2x_{ch_name}_l1"].append(F.l1_loss(ch_pred, ch_gt))
                                metrics[f"rgb2x_{ch_name}_psnr"].append(
                                    10 * torch.log10(torch.tensor(1.0, device=device) / ch_mse.clamp(min=1e-10))
                                )
                                if ch_name == "normal":
                                    pred_n = ch_pred * 2.0 - 1.0
                                    gt_n = ch_gt * 2.0 - 1.0
                                    metrics["rgb2x_normal_cosine"].append(
                                        F.cosine_similarity(pred_n, gt_n, dim=-1).mean()
                                    )
                                ch_pred_vis = ch_pred[0].cpu()
                                ch_gt_vis = ch_gt[0].cpu()
                                if ch_pred_vis.shape[-1] == 1:
                                    ch_pred_vis = ch_pred_vis.expand(-1, -1, 3)
                                    ch_gt_vis = ch_gt_vis.expand(-1, -1, 3)
                                ch_canvas = torch.cat([ch_gt_vis, ch_pred_vis], dim=1)
                                ch_canvas = (ch_canvas.numpy() * 255).astype(np.uint8)
                                imageio.imwrite(
                                    f"{self.render_dir}/{stage}_step{step}_{i:04d}_{ch_name}.png",
                                    ch_canvas,
                                )

        if world_rank == 0:
            ellipse_time /= len(valloader) * num_passes

            per_image_stats = {
                k: [x.item() for x in v] for k, v in metrics.items()
            }
            with open(f"{self.stats_dir}/{stage}_step{step:04d}_per_image.json", "w") as f:
                json.dump(per_image_stats, f)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_color_correction_metric:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )

            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()
        self._restore_from_ema()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        self._apply_ema()
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)

        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]
            renders, _, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB",
            )  # [1, H, W, 3]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)
            canvas_list = [colors]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)

        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")
        self._restore_from_ema()

    @torch.no_grad()
    def export_ppisp_reports(self) -> None:
        """Export PPISP visualization reports (PDF) and parameter JSON."""
        if self.cfg.post_processing != "ppisp":
            return
        print("Exporting PPISP reports...")

        # Compute frames per camera from training dataset
        num_cameras = self.parser.num_cameras
        frames_per_camera = [0] * num_cameras
        for idx in self.trainset.indices:
            cam_idx = self.parser.camera_indices[idx]
            frames_per_camera[cam_idx] += 1

        # Generate camera names from COLMAP camera IDs
        # camera_id_to_idx maps COLMAP ID -> 0-based index
        idx_to_camera_id = {v: k for k, v in self.parser.camera_id_to_idx.items()}
        camera_names = [f"camera_{idx_to_camera_id[i]}" for i in range(num_cameras)]

        # Export reports
        output_dir = Path(self.cfg.result_dir) / "ppisp_reports"
        pdf_paths = export_ppisp_report(
            self.post_processing_module,
            frames_per_camera,
            output_dir,
            camera_names=camera_names,
        )
        print(f"PPISP reports saved to {output_dir}")
        for path in pdf_paths:
            print(f"  - {path.name}")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatNHTRenderTabState)
        self._apply_ema()
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            rgb = render_colors[0, ..., 0:3].clamp(0, 1)
            bkgd = (
                torch.tensor(render_tab_state.backgrounds, device=self.device).float()
                / 255.0
            )
            rgb = rgb + bkgd * (1.0 - render_alphas[0])
            renders = rgb.clamp(0, 1).cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = render_colors[0, ..., 3:4]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        self._restore_from_ema()
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    # Import post-processing modules based on configuration
    # These imports must be here (not in __main__) for distributed workers
    if cfg.post_processing == "bilateral_grid":
        global BilateralGrid, slice, total_variation_loss
        if cfg.bilateral_grid_fused:
            from fused_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
        else:
            from lib_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
    elif cfg.post_processing == "ppisp":
        global PPISP, PPISPConfig, export_ppisp_report
        from ppisp import PPISP, PPISPConfig
        from ppisp.report import export_ppisp_report

    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only -- free optimizer states to save GPU memory
        runner.optimizers = {}
        if hasattr(runner, 'strategy_state'):
            runner.strategy_state = {}
        torch.cuda.empty_cache()

        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        if runner.post_processing_module is not None:
            pp_state = ckpts[0].get("post_processing")
            if pp_state is not None:
                runner.post_processing_module.load_state_dict(pp_state)
        step = ckpts[0]["step"]
        # recover deferred module
        ckpt_sd = ckpts[0]["deferred_module"]
        target = (
            runner.deferred_module.module
            if world_size > 1
            else runner.deferred_module
        )
        target.load_state_dict(ckpt_sd)
        target.eval()
        if runner._ema_shadow is not None and "deferred_ema" in ckpts[0]:
            runner._ema_shadow = ckpts[0]["deferred_ema"]
        # run eval and render trajectory
        runner.eval(step=step)
        runner.render_traj(step=step)
        # run compression
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()
        runner.export_ppisp_reports()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "NHT AOV training with MCMC densification (default).",
            Config(
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
