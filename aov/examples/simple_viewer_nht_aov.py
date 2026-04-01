# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import sys
from pathlib import Path

_NHT_REPO_ROOT = Path(__file__).resolve().parents[2]
_GSPLAT_EXAMPLES = _NHT_REPO_ROOT / "gsplat" / "examples"
for _p in (_GSPLAT_EXAMPLES, _NHT_REPO_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import argparse
import threading
import time
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import viser

from gsplat.distributed import cli
from gsplat.rendering import rasterization
from aov.deferred_shader import DeferredShaderAOVModule

from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer_nht_aov import GsplatNHTAOVViewer, GsplatNHTAOVRenderTabState


class PCAVisualizer:
    """PCA projection of high-dim features to RGB for viewer visualization."""

    def __init__(self):
        self._fitted = False
        self._components = None
        self._mean = None
        self._min = None
        self._max = None

    def fit(self, features: torch.Tensor):
        features = features.float()
        self._mean = features.mean(dim=0)
        centered = features - self._mean
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        self._components = Vt[:3]
        projected = centered @ self._components.T
        self._min = projected.min(dim=0).values
        self._max = projected.max(dim=0).values
        self._fitted = True

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        H, W, D = features.shape
        flat = features.reshape(-1, D).float()
        centered = flat - self._mean
        projected = centered @ self._components.T
        rng = (self._max - self._min).clamp(min=1e-6)
        projected = (projected - self._min) / rng
        return projected.clamp(0, 1).reshape(H, W, 3)


class CLIPTextEncoder:
    """Lightweight wrapper that keeps only the CLIP text encoder and visual
    projection matrix needed for text-based semantic segmentation.

    Pass the already-imported ``open_clip`` module so callers can treat
    ``open_clip_torch`` as an optional dependency."""

    def __init__(self, device: torch.device, open_clip: Any):
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )
        model = model.to(device).eval()
        self._tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self._text_model = model
        self._visual_proj = model.visual.proj.detach().clone()  # (768, 512)
        self._device = device

    @torch.no_grad()
    def encode_text(self, query: str) -> torch.Tensor:
        """Return L2-normalized 512-dim text embedding."""
        tokens = self._tokenizer([query]).to(self._device)
        text_feat = self._text_model.encode_text(tokens)  # (1, 512)
        text_feat = F.normalize(text_feat, dim=-1)
        return text_feat.squeeze(0)  # (512,)

    @torch.no_grad()
    def project_visual(self, features: torch.Tensor) -> torch.Tensor:
        """Project LSEG features (H, W, 768) into CLIP joint space (H, W, 512)."""
        H, W, D = features.shape
        flat = features.reshape(-1, D).float()
        projected = flat @ self._visual_proj  # (H*W, 512)
        projected = F.normalize(projected, dim=-1)
        return projected.reshape(H, W, -1)


class SegmentationState:
    """Thread-safe mutable state for segmentation overlays."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_lseg_features: Optional[torch.Tensor] = None  # (H, W, D)
        self.latest_width: int = 0
        self.latest_height: int = 0
        self.similarity_map: Optional[torch.Tensor] = None  # (H, W, 1)

    def update_features(self, features: torch.Tensor, width: int, height: int):
        with self.lock:
            self.latest_lseg_features = features
            self.latest_width = width
            self.latest_height = height

    def set_similarity(self, sim: Optional[torch.Tensor]):
        with self.lock:
            self.similarity_map = sim

    def get_similarity(self) -> Optional[torch.Tensor]:
        with self.lock:
            return self.similarity_map

    def clear(self):
        with self.lock:
            self.similarity_map = None


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    assert args.ckpt is not None, "NHT AOV viewer requires --ckpt checkpoint(s)"

    # Load checkpoint(s)
    ckpt_means, ckpt_quats, ckpt_scales, ckpt_opacities, ckpt_features = (
        [], [], [], [], [],
    )
    deferred_state_dict = None
    aov_config = None

    for i, ckpt_path in enumerate(args.ckpt):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        splats = ckpt["splats"]
        ckpt_means.append(splats["means"])
        ckpt_quats.append(F.normalize(splats["quats"], p=2, dim=-1))
        ckpt_scales.append(torch.exp(splats["scales"]))
        ckpt_opacities.append(torch.sigmoid(splats["opacities"]))
        ckpt_features.append(splats["features"])
        if i == 0:
            deferred_state_dict = ckpt["deferred_module"]
            aov_config = ckpt.get("aov_config", {})

    means = torch.cat(ckpt_means, dim=0)
    quats = torch.cat(ckpt_quats, dim=0)
    scales = torch.cat(ckpt_scales, dim=0)
    opacities = torch.cat(ckpt_opacities, dim=0)
    features = torch.cat(ckpt_features, dim=0)

    feature_dim = features.shape[-1]
    print(f"Number of Gaussians: {len(means)}, Feature dim: {feature_dim}")

    # Determine AOV head dimensions from checkpoint metadata or CLI overrides
    lseg_feature_dim = aov_config.get("lseg_feature_dim", 0) if aov_config else 0
    dinov3_feature_dim = aov_config.get("dinov3_feature_dim", 0) if aov_config else 0
    rgb2x_channels = aov_config.get("rgb2x_channels", None) if aov_config else None

    if args.lseg_feature_dim is not None:
        lseg_feature_dim = args.lseg_feature_dim
    if args.dinov3_feature_dim is not None:
        dinov3_feature_dim = args.dinov3_feature_dim

    has_aov = lseg_feature_dim > 0 or dinov3_feature_dim > 0 or (rgb2x_channels is not None and len(rgb2x_channels) > 0)
    print(f"AOV config: lseg_dim={lseg_feature_dim}, dinov3_dim={dinov3_feature_dim}, "
          f"rgb2x={rgb2x_channels}")

    # Instantiate and load the deferred shading AOV module
    deferred_module = DeferredShaderAOVModule(
        feature_dim=feature_dim,
        enable_view_encoding=args.enable_view_encoding,
        view_encoding_type=args.view_encoding_type,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_num_layers=args.mlp_num_layers,
        sh_degree=args.deferred_sh_degree,
        sh_scale=args.deferred_sh_scale,
        fourier_num_freqs=args.fourier_num_freqs,
        center_ray_encoding=args.center_ray_encoding,
        lseg_feature_dim=lseg_feature_dim,
        dinov3_feature_dim=dinov3_feature_dim,
        rgb2x_channels=rgb2x_channels,
    ).to(device)
    deferred_module.load_state_dict(deferred_state_dict)
    deferred_module.eval()

    # PCA visualizers (lazily fitted on first render)
    lseg_pca = PCAVisualizer() if lseg_feature_dim > 0 else None
    dinov3_pca = PCAVisualizer() if dinov3_feature_dim > 0 else None

    # Build available render modes
    available_aov_modes = []
    if lseg_feature_dim > 0:
        available_aov_modes.append("lseg_pca")
    if dinov3_feature_dim > 0:
        available_aov_modes.append("dinov3_pca")
    if rgb2x_channels:
        for ch_name in rgb2x_channels:
            available_aov_modes.append(ch_name)
    if available_aov_modes:
        print(f"AOV render modes available: {available_aov_modes}")

    # --- Segmentation infrastructure ---
    has_lseg = lseg_feature_dim > 0
    seg_state = SegmentationState()
    clip_encoder: Optional[CLIPTextEncoder] = None
    if has_lseg:
        try:
            import open_clip
        except ImportError:
            print(
                "[viewer] Optional package open_clip_torch is not installed; "
                "text-query segmentation is disabled. Install in your environment "
                "if needed: pip install open_clip_torch — "
                "https://github.com/mlfoundations/open_clip"
            )
        else:
            print("Loading OpenCLIP text encoder for semantic segmentation ...")
            clip_encoder = CLIPTextEncoder(device, open_clip)
            print("CLIP text encoder ready.")

    def _on_text_query(query: str) -> None:
        """Callback: encode text query and compute similarity against cached LSEG features."""
        if not query or clip_encoder is None:
            return
        with seg_state.lock:
            feats = seg_state.latest_lseg_features
        if feats is None:
            return
        text_feat = clip_encoder.encode_text(query)  # (512,)
        projected = clip_encoder.project_visual(feats)  # (H, W, 512)
        sim = torch.einsum("d,hwd->hw", text_feat, projected)  # (H, W)
        sim = sim.unsqueeze(-1)  # (H, W, 1)
        seg_state.set_similarity(sim)

    def _on_click_seg(screen_pos: Tuple[float, float]) -> None:
        """Callback: use clicked pixel's LSEG feature for similarity search."""
        with seg_state.lock:
            feats = seg_state.latest_lseg_features
            w = seg_state.latest_width
            h = seg_state.latest_height
        if feats is None or w == 0 or h == 0:
            return
        u_norm, v_norm = screen_pos
        px = min(int(u_norm * w), w - 1)
        py = min(int(v_norm * h), h - 1)
        ref_feat = feats[py, px]  # (D,)
        ref_feat = F.normalize(ref_feat.unsqueeze(0), dim=-1)  # (1, D)
        flat = feats.reshape(-1, feats.shape[-1]).float()
        flat = F.normalize(flat, dim=-1)
        sim = (flat @ ref_feat.T).reshape(feats.shape[0], feats.shape[1], 1)  # (H, W, 1)
        seg_state.set_similarity(sim)

    def _on_clear_seg() -> None:
        seg_state.clear()

    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatNHTAOVRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        # Determine rasterization mode
        render_mode = render_tab_state.render_mode
        need_depth = render_mode in ["depth(accumulated)", "depth(expected)"]
        if need_depth:
            depth_suffix = "D" if render_mode == "depth(accumulated)" else "ED"
            rast_render_mode = f"RGB+{depth_suffix}"
        else:
            rast_render_mode = "RGB"

        render_colors, render_alphas, info = rasterization(
            means, quats, scales, opacities, features,
            viewmat[None], K[None], width, height,
            sh_degree=None,
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            render_mode=rast_render_mode,
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
            tile_size=args.tile_size,
            with_ut=args.with_ut,
            with_eval3d=args.with_eval3d,
            nht=True,
            center_ray_mode=args.center_ray_encoding,
            ray_dir_scale=deferred_module.ray_dir_scale,
        )

        render_tab_state.total_gs_count = len(means)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        # Deferred shading: features -> RGB + AOV outputs
        rgb, aov_outputs, extras = deferred_module(render_colors)

        # Cache LSEG features for segmentation every frame
        if has_lseg and "lseg" in aov_outputs:
            seg_state.update_features(aov_outputs["lseg"][0], width, height)

        # Check if this is an AOV render mode
        is_aov_mode = render_mode in available_aov_modes

        if render_mode == "rgb":
            rgb = rgb[0, ..., 0:3].clamp(0, 1)
            bkgd = (
                torch.tensor(render_tab_state.backgrounds, device=device).float()
                / 255.0
            )
            rgb = rgb + bkgd * (1.0 - render_alphas[0])
            renders = rgb.clamp(0, 1)

        elif render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = extras[0, ..., 0:1]
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
            renders = apply_float_colormap(depth_norm, render_tab_state.colormap)

        elif render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = apply_float_colormap(alpha, render_tab_state.colormap)

        elif render_mode == "lseg_pca" and "lseg" in aov_outputs and lseg_pca is not None:
            feat = aov_outputs["lseg"][0]  # [H, W, D]
            if not lseg_pca._fitted:
                lseg_pca.fit(feat.reshape(-1, feat.shape[-1]))
            renders = lseg_pca.transform(feat)

        elif render_mode == "dinov3_pca" and "dinov3" in aov_outputs and dinov3_pca is not None:
            feat = aov_outputs["dinov3"][0]  # [H, W, D]
            if not dinov3_pca._fitted:
                dinov3_pca.fit(feat.reshape(-1, feat.shape[-1]))
            renders = dinov3_pca.transform(feat)

        elif is_aov_mode and rgb2x_channels and render_mode in rgb2x_channels:
            if "rgb2x" in aov_outputs:
                rgb2x_split = deferred_module.split_rgb2x(aov_outputs["rgb2x"])
                ch_key = f"rgb2x_{render_mode}"
                if ch_key in rgb2x_split:
                    ch_data = rgb2x_split[ch_key][0].clamp(0, 1)
                    if ch_data.shape[-1] == 1:
                        ch_data = ch_data.expand(-1, -1, 3)
                    renders = ch_data
                else:
                    renders = rgb[0, ..., 0:3].clamp(0, 1)
            else:
                renders = rgb[0, ..., 0:3].clamp(0, 1)

        else:
            rgb_out = rgb[0, ..., 0:3].clamp(0, 1)
            bkgd = (
                torch.tensor(render_tab_state.backgrounds, device=device).float()
                / 255.0
            )
            rgb_out = rgb_out + bkgd * (1.0 - render_alphas[0])
            renders = rgb_out.clamp(0, 1)

        # --- Segmentation overlay ---
        if render_tab_state.seg_enabled:
            sim_map = seg_state.get_similarity()
            if sim_map is not None:
                sim_h, sim_w = sim_map.shape[:2]
                if sim_h != height or sim_w != width:
                    sim_map = F.interpolate(
                        sim_map.permute(2, 0, 1).unsqueeze(0),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).permute(1, 2, 0)
                threshold = render_tab_state.seg_threshold
                overlay_alpha = render_tab_state.seg_overlay_alpha
                sim_clamped = sim_map.clamp(0, 1)
                heatmap = apply_float_colormap(sim_clamped, "turbo")
                mask = (sim_map > threshold).float()
                renders = renders * (1.0 - mask * overlay_alpha) + heatmap * mask * overlay_alpha

        return renders.clamp(0, 1).cpu().numpy()

    seg_callbacks = {
        "on_text_query": _on_text_query,
        "on_click": _on_click_seg,
        "on_clear": _on_clear_seg,
    }

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = GsplatNHTAOVViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
        aov_modes=available_aov_modes,
        has_lseg=has_lseg,
        seg_callbacks=seg_callbacks,
    )
    print("Viewer running... Ctrl+C to exit.")
    if available_aov_modes:
        print(f"AOV render modes: {available_aov_modes}")
    if has_lseg:
        print("Segmentation available: use text query or click-to-segment in the Segmentation panel.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # View an NHT AOV checkpoint (from repo root or aov/examples)
    CUDA_VISIBLE_DEVICES=0 python aov/examples/simple_viewer_nht_aov.py \
        --ckpt results/garden/ckpts/ckpt_29999_rank0.pt \
        --output_dir results/garden/ \
        --port 8082
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", required=True,
        help="path to NHT AOV .pt checkpoint file(s)",
    )
    parser.add_argument("--port", type=int, default=8080, help="port for the viewer server")
    parser.add_argument(
        "--with_ut", action="store_true", default=True,
        help="use uncentered transform (default: True)",
    )
    parser.add_argument(
        "--with_eval3d", action="store_true", default=True,
        help="use eval 3D (default: True)",
    )
    parser.add_argument("--tile_size", type=int, default=16, help="tile size for rasterization")

    # Deferred shading module configuration
    parser.add_argument(
        "--no_view_encoding", dest="enable_view_encoding", action="store_false",
        help="disable view-dependent encoding",
    )
    parser.set_defaults(enable_view_encoding=True)
    parser.add_argument(
        "--view_encoding_type", type=str, default="sh", choices=["sh", "fourier"],
        help="view encoding type",
    )
    parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="deferred MLP hidden dimension")
    parser.add_argument("--mlp_num_layers", type=int, default=3, help="deferred MLP number of hidden layers")
    parser.add_argument("--deferred_sh_degree", type=int, default=3, help="SH degree for view encoding")
    parser.add_argument("--deferred_sh_scale", type=float, default=3.0, help="scale applied to dirs before SH evaluation")
    parser.add_argument("--fourier_num_freqs", type=int, default=4, help="number of Fourier frequency levels")
    parser.add_argument(
        "--no_center_ray_encoding", dest="center_ray_encoding", action="store_false",
        help="disable center ray encoding (use per-pixel rays)",
    )
    parser.set_defaults(center_ray_encoding=True)

    # AOV overrides (normally auto-detected from checkpoint aov_config)
    parser.add_argument(
        "--lseg_feature_dim", type=int, default=None,
        help="override LSEG feature dim (auto-detected from checkpoint if not set)",
    )
    parser.add_argument(
        "--dinov3_feature_dim", type=int, default=None,
        help="override DINOv3 feature dim (auto-detected from checkpoint if not set)",
    )

    args = parser.parse_args()
    cli(main, args, verbose=True)
