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

"""Tests for the NHT AOV (Arbitrary Output Variable) training mode.

Covers DeferredShaderAOVModule construction & forward (various head configs),
AOVDataset discovery & loading with synthetic data, PCAVisualizer, TV loss,
Config AOV fields, checkpoint round-trip with aov_config, and end-to-end
rasterization + AOV decode pipeline.

Usage (from nht-release repository root, after ``pip install -e .``):
    pytest tests/test_nht_aov.py -v -s

You can also run ``python tests/test_nht_aov.py``; that forwards to pytest. On first import,
``_nht_rasterizer_available()`` may trigger a full NHT CUDA compile and appear idle for several
minutes (not stuck).
"""

import math
import os
import sys
import tempfile
from typing import Dict, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import Tensor

# Superproject: AOV trainer lives under aov/examples; colmap helpers under gsplat/examples.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (
    _REPO_ROOT / "aov" / "examples",
    _REPO_ROOT / "gsplat" / "examples",
    _REPO_ROOT,
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import gsplat
from gsplat.cuda._wrapper import (
    get_encoding_expansion_factor,
    get_feature_divisor,
)

try:
    from gsplat.cuda._wrapper import fused_activate_splat_params
except (ImportError, AttributeError):
    fused_activate_splat_params = None

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

_tcnn_available = False
try:
    import tinycudann as tcnn
    _tcnn_available = True
except ImportError:
    pass


def _nht_rasterizer_available() -> bool:
    if not torch.cuda.is_available() or not gsplat.has_3dgs():
        return False
    try:
        from gsplat.rendering import rasterization
        N = 64
        d = "cuda"
        means = torch.zeros(N, 3, device=d)
        quats = torch.tensor([1., 0., 0., 0.], device=d).expand(N, -1).contiguous()
        scales = torch.full((N, 3), 0.01, device=d)
        opacities = torch.full((N,), 0.5, device=d)
        features = torch.randn(N, 16, device=d)
        K = torch.tensor([[[50., 0., 16.], [0., 50., 12.], [0., 0., 1.]]], device=d)
        c2w = torch.eye(4, device=d).unsqueeze(0)
        c2w[0, 2, 3] = -2.0
        viewmats = torch.linalg.inv(c2w)
        rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=features, viewmats=viewmats, Ks=K, width=32, height=24,
            nht=True, with_eval3d=True, with_ut=True, packed=False, sh_degree=None,
        )
        return True
    except Exception:
        return False


if __name__ == "__main__":
    print(
        "test_nht_aov: probing NHT rasterizer (first CUDA JIT can take several minutes)...",
        flush=True,
    )
_nht_raster_ok = _nht_rasterizer_available()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_camera(width=64, height=48, focal=100.0, device="cuda"):
    K = torch.tensor(
        [[focal, 0.0, width / 2.0],
         [0.0, focal, height / 2.0],
         [0.0, 0.0, 1.0]],
        device=device,
    )
    c2w = torch.eye(4, device=device)
    c2w[2, 3] = -3.0
    return K.unsqueeze(0), c2w.unsqueeze(0), width, height


def _make_splats(N=256, feature_dim=16, device="cuda"):
    means = torch.randn(N, 3, device=device) * 0.3
    quats = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.rand(N, 3, device=device) * 0.5 - 2.0
    opacities = torch.logit(torch.full((N,), 0.5, device=device))
    features = torch.randn(N, feature_dim, device=device)
    return means, quats, scales, opacities, features


def _deferred_raster_channels(mod, *, extra: int = 0) -> int:
    """Last dim for tensors passed to ``DeferredShaderAOVModule`` (matches ``forward``)."""
    if mod.enable_view_encoding:
        return mod.encoded_dim + 3 + extra
    return mod.encoded_dim + extra


# ===================================================================
# 1. total_variation_loss_features tests
# ===================================================================

class TestTotalVariationLoss:

    def test_import(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from simple_trainer_nht_aov import total_variation_loss_features
        assert callable(total_variation_loss_features)

    def test_zero_for_constant(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from simple_trainer_nht_aov import total_variation_loss_features
        feat = torch.ones(1, 8, 8, 3)
        loss = total_variation_loss_features(feat)
        assert loss.item() == 0.0

    def test_positive_for_varying(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from simple_trainer_nht_aov import total_variation_loss_features
        feat = torch.randn(1, 8, 8, 3)
        loss = total_variation_loss_features(feat)
        assert loss.item() > 0.0

    def test_output_is_scalar(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from simple_trainer_nht_aov import total_variation_loss_features
        feat = torch.randn(2, 16, 16, 5)
        loss = total_variation_loss_features(feat)
        assert loss.ndim == 0


# ===================================================================
# 2. PCAVisualizer tests
# ===================================================================

class TestPCAVisualizer:

    def _make_pca(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from simple_trainer_nht_aov import PCAVisualizer
        return PCAVisualizer()

    def test_fit_and_transform(self):
        pca = self._make_pca()
        features_2d = torch.randn(100, 64)
        pca.fit(features_2d)
        assert pca._fitted
        assert pca._components.shape == (3, 64)
        assert pca._mean.shape == (64,)

        features_3d = torch.randn(8, 8, 64)
        result = pca.transform(features_3d)
        assert result.shape == (8, 8, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_deterministic(self):
        pca = self._make_pca()
        torch.manual_seed(42)
        features = torch.randn(50, 32)
        pca.fit(features)
        img = torch.randn(4, 4, 32)
        r1 = pca.transform(img)
        r2 = pca.transform(img)
        torch.testing.assert_close(r1, r2)

    def test_not_fitted_raises(self):
        pca = self._make_pca()
        assert not pca._fitted
        with pytest.raises((AttributeError, TypeError)):
            pca.transform(torch.randn(4, 4, 16))


# ===================================================================
# 3. Config AOV fields tests
# ===================================================================

_aov_config_import_error = None
try:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    from simple_trainer_nht_aov import Config as _AOVConfig
except Exception as _e:
    _aov_config_import_error = str(_e)
    _AOVConfig = None


@pytest.mark.skipif(_AOVConfig is None,
                    reason=f"AOV Config import failed: {_aov_config_import_error}")
class TestAOVConfig:

    def _make_config(self, **overrides):
        return _AOVConfig(**overrides)

    def test_aov_defaults_disabled(self):
        cfg = self._make_config()
        assert cfg.lseg_data is False
        assert cfg.dinov3_data is False
        assert cfg.rgb2x_data is False

    def test_lseg_defaults(self):
        cfg = self._make_config()
        assert cfg.lseg_loss_lambda == 1.0
        assert cfg.lseg_cosine_lambda == 0.1
        assert cfg.lseg_tv_lambda == 0.0
        assert cfg.lseg_data_dir is None

    def test_dinov3_defaults(self):
        cfg = self._make_config()
        assert cfg.dinov3_loss_lambda == 1.0
        assert cfg.dinov3_cosine_lambda == 0.1
        assert cfg.dinov3_tv_lambda == 0.0
        assert cfg.dinov3_data_dir is None

    def test_rgb2x_defaults(self):
        cfg = self._make_config()
        assert cfg.rgb2x_loss_lambda == 1.0
        assert cfg.rgb2x_albedo_lambda == 1.0
        assert cfg.rgb2x_roughness_lambda == 1.0
        assert cfg.rgb2x_metallic_lambda == 1.0
        assert cfg.rgb2x_irradiance_lambda == 1.0
        assert cfg.rgb2x_normal_lambda == 1.0
        assert cfg.rgb2x_normal_cosine_lambda == 0.1
        assert cfg.rgb2x_tv_lambda == 0.0

    def test_nht_options_preserved(self):
        cfg = self._make_config()
        assert cfg.deferred_opt is True
        assert cfg.deferred_opt_feature_dim == 64
        assert cfg.deferred_mlp_hidden_dim == 128

    def test_custom_lambdas(self):
        cfg = self._make_config(
            lseg_data=True,
            lseg_loss_lambda=0.5,
            lseg_cosine_lambda=0.1,
            rgb2x_data=True,
            rgb2x_albedo_lambda=2.0,
        )
        assert cfg.lseg_data is True
        assert cfg.lseg_loss_lambda == 0.5
        assert cfg.lseg_cosine_lambda == 0.1
        assert cfg.rgb2x_data is True
        assert cfg.rgb2x_albedo_lambda == 2.0


# ===================================================================
# 4. DeferredShaderAOVModule tests
# ===================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not _tcnn_available, reason="tinycudann not available")
class TestDeferredShaderAOVModule:

    def _make_module(self, feature_dim=16, enable_view_encoding=True,
                     lseg_feature_dim=0, dinov3_feature_dim=0,
                     rgb2x_channels=None, **kwargs):
        from aov.deferred_shader import DeferredShaderAOVModule
        return DeferredShaderAOVModule(
            feature_dim=feature_dim,
            enable_view_encoding=enable_view_encoding,
            lseg_feature_dim=lseg_feature_dim,
            dinov3_feature_dim=dinov3_feature_dim,
            rgb2x_channels=rgb2x_channels,
            **kwargs,
        ).to(device)

    def test_construction_no_aov(self):
        mod = self._make_module()
        assert mod._mode == "rgb2x"
        assert mod._aov_keys == []
        assert mod.lseg_feature_dim == 0
        assert mod.dinov3_feature_dim == 0

    def test_construction_with_lseg(self):
        mod = self._make_module(lseg_feature_dim=512)
        assert mod._mode == "semantic"
        assert mod.uses_split_rgb_aux_linear
        assert mod.auxiliary_head is not None
        assert mod.auxiliary_head.out_features == 512

    def test_construction_with_dinov3(self):
        mod = self._make_module(dinov3_feature_dim=1024)
        assert mod._mode == "semantic"
        assert mod.auxiliary_head is not None
        assert mod.auxiliary_head.out_features == 1024

    def test_construction_with_rgb2x(self):
        channels = {"albedo": 3, "roughness": 1, "normal": 3}
        mod = self._make_module(rgb2x_channels=channels)
        assert mod._mode == "rgb2x"
        assert mod._aov_keys == ["rgb2x"]
        assert mod.uses_direct_fused_output
        assert mod.fused_tcnn_sigmoid
        assert mod.tcnn_emitted_sigmoid_outputs
        assert mod.rgb2x_channel_layout == [
            ("albedo", 3), ("normal", 3), ("roughness", 1)
        ]

    def test_construction_all_heads(self):
        channels = {"albedo": 3, "metallic": 1}
        mod = self._make_module(
            lseg_feature_dim=512,
            dinov3_feature_dim=768,
            rgb2x_channels=channels,
        )
        assert mod._mode == "semantic"
        assert mod.auxiliary_head is not None
        assert mod.auxiliary_head.out_features == 512 + 768 + 3 + 1

    def test_forward_no_aov_shape(self):
        feature_dim = 16
        mod = self._make_module(feature_dim=feature_dim)
        K, c2w, W, H = _make_camera(device=device)
        rendered = torch.randn(1, H, W, _deferred_raster_channels(mod), device=device)
        colors, aov_outputs, extras = mod(rendered, K, c2w)
        assert colors.shape == (1, H, W, 3)
        assert len(aov_outputs) == 0
        assert extras is None

    def test_forward_with_lseg(self):
        feature_dim = 16
        lseg_dim = 64
        mod = self._make_module(feature_dim=feature_dim, lseg_feature_dim=lseg_dim)
        K, c2w, W, H = _make_camera(device=device)
        rendered = torch.randn(1, H, W, _deferred_raster_channels(mod), device=device)
        colors, aov_outputs, extras = mod(rendered, K, c2w)
        assert colors.shape == (1, H, W, 3)
        assert "lseg" in aov_outputs
        assert aov_outputs["lseg"].shape == (1, H, W, lseg_dim)
        # RGB should be in [0, 1] due to sigmoid
        assert colors.min() >= 0.0
        assert colors.max() <= 1.0

    def test_forward_with_dinov3(self):
        feature_dim = 16
        dinov3_dim = 128
        mod = self._make_module(feature_dim=feature_dim, dinov3_feature_dim=dinov3_dim)
        K, c2w, W, H = _make_camera(device=device)
        rendered = torch.randn(1, H, W, _deferred_raster_channels(mod), device=device)
        colors, aov_outputs, extras = mod(rendered, K, c2w)
        assert "dinov3" in aov_outputs
        assert aov_outputs["dinov3"].shape == (1, H, W, dinov3_dim)

    def test_forward_with_rgb2x(self):
        feature_dim = 16
        channels = {"albedo": 3, "roughness": 1, "normal": 3}
        mod = self._make_module(feature_dim=feature_dim, rgb2x_channels=channels)
        K, c2w, W, H = _make_camera(device=device)
        rendered = torch.randn(1, H, W, _deferred_raster_channels(mod), device=device)
        colors, aov_outputs, extras = mod(rendered, K, c2w)
        assert "rgb2x" in aov_outputs
        total = sum(channels.values())
        assert aov_outputs["rgb2x"].shape == (1, H, W, total)
        parts = mod.split_rgb2x(aov_outputs["rgb2x"])
        assert parts["rgb2x_albedo"].shape == (1, H, W, 3)
        assert parts["rgb2x_roughness"].shape == (1, H, W, 1)
        assert parts["rgb2x_normal"].shape == (1, H, W, 3)
        for t in parts.values():
            assert t.min() >= 0.0
            assert t.max() <= 1.0

    def test_forward_all_heads(self):
        feature_dim = 16
        channels = {"albedo": 3, "metallic": 1}
        mod = self._make_module(
            feature_dim=feature_dim,
            lseg_feature_dim=32,
            dinov3_feature_dim=48,
            rgb2x_channels=channels,
        )
        K, c2w, W, H = _make_camera(device=device)
        rendered = torch.randn(1, H, W, _deferred_raster_channels(mod), device=device)
        colors, aov_outputs, extras = mod(rendered, K, c2w)
        assert colors.shape == (1, H, W, 3)
        assert "lseg" in aov_outputs
        assert "dinov3" in aov_outputs
        assert "rgb2x" in aov_outputs
        parts = mod.split_rgb2x(aov_outputs["rgb2x"])
        assert "rgb2x_albedo" in parts
        assert "rgb2x_metallic" in parts

    def test_forward_with_extras(self):
        feature_dim = 16
        mod = self._make_module(feature_dim=feature_dim, lseg_feature_dim=32)
        K, c2w, W, H = _make_camera(device=device)
        extra_ch = 2
        rendered = torch.randn(1, H, W, _deferred_raster_channels(mod, extra=extra_ch), device=device)
        colors, aov_outputs, extras = mod(rendered, K, c2w)
        assert extras is not None
        assert extras.shape == (1, H, W, extra_ch)

    @pytest.mark.parametrize("view_encoding_type", ["sh", "fourier"])
    @pytest.mark.parametrize("center_ray", [True, False])
    def test_view_encoding_variants(self, view_encoding_type, center_ray):
        mod = self._make_module(
            feature_dim=16,
            lseg_feature_dim=32,
            view_encoding_type=view_encoding_type,
            center_ray_encoding=center_ray,
        )
        K, c2w, W, H = _make_camera(device=device)
        rendered = torch.randn(1, H, W, _deferred_raster_channels(mod), device=device)
        colors, aov_outputs, extras = mod(rendered, K, c2w)
        assert colors.shape == (1, H, W, 3)
        assert "lseg" in aov_outputs

    def test_no_view_encoding(self):
        mod = self._make_module(
            feature_dim=16,
            enable_view_encoding=False,
            dinov3_feature_dim=64,
        )
        K, c2w, W, H = _make_camera(device=device)
        rendered = torch.randn(1, H, W, mod.encoded_dim, device=device)
        colors, aov_outputs, extras = mod(rendered, K, c2w)
        assert colors.shape == (1, H, W, 3)
        assert "dinov3" in aov_outputs

    def test_backward_propagates(self):
        mod = self._make_module(
            feature_dim=16,
            lseg_feature_dim=32,
            rgb2x_channels={"albedo": 3},
        )
        K, c2w, W, H = _make_camera(device=device)
        rendered = torch.randn(
            1, H, W, _deferred_raster_channels(mod), device=device, requires_grad=True
        )
        colors, aov_outputs, _ = mod(rendered, K, c2w)
        loss = colors.mean() + aov_outputs["lseg"].mean() + aov_outputs["rgb2x"].mean()
        loss.backward()
        assert rendered.grad is not None
        assert rendered.grad.abs().sum() > 0

    def test_state_dict_round_trip(self):
        channels = {"albedo": 3, "roughness": 1}
        mod = self._make_module(
            feature_dim=16,
            lseg_feature_dim=32,
            rgb2x_channels=channels,
        )
        sd = mod.state_dict()
        mod2 = self._make_module(
            feature_dim=16,
            lseg_feature_dim=32,
            rgb2x_channels=channels,
        )
        mod2.load_state_dict(sd)
        K, c2w, W, H = _make_camera(device=device)
        rendered = torch.randn(1, H, W, _deferred_raster_channels(mod), device=device)
        c1, a1, _ = mod(rendered, K, c2w)
        c2, a2, _ = mod2(rendered, K, c2w)
        torch.testing.assert_close(c1, c2)
        for key in a1:
            torch.testing.assert_close(a1[key], a2[key])

    def test_split_rgb2x_matches_layout(self):
        channels = {"albedo": 3, "roughness": 1}
        mod = self._make_module(feature_dim=16, rgb2x_channels=channels)
        K, c2w, W, H = _make_camera(device=device)
        rendered = torch.randn(1, H, W, _deferred_raster_channels(mod), device=device)
        _, aov, _ = mod(rendered, K, c2w)
        parts = mod.split_rgb2x(aov["rgb2x"])
        assert set(parts.keys()) == {"rgb2x_albedo", "rgb2x_roughness"}


# ===================================================================
# 5. AOVDataset tests (synthetic data on disk)
# ===================================================================

class TestAOVDataset:

    @pytest.fixture
    def mock_aov_data(self, tmp_path):
        """Create synthetic AOV data directories for testing."""
        import imageio.v2 as imageio
        import numpy as np

        # Simulate an image directory with 4 images
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        stems = ["img_000", "img_001", "img_002", "img_003"]
        for stem in stems:
            img = np.random.randint(0, 255, (24, 32, 3), dtype=np.uint8)
            imageio.imwrite(str(img_dir / f"{stem}.png"), img)

        # LSEG data
        lseg_dir = tmp_path / "images_bin_lseg"
        lseg_dir.mkdir()
        lseg_dim = 64
        for stem in stems:
            feat = torch.randn(lseg_dim, 12, 16)  # [C, H, W]
            torch.save(feat, str(lseg_dir / f"{stem}_fmap_CxHxW.pt"))

        # DINOv3 data
        dinov3_dir = tmp_path / "images_dinov3"
        dinov3_dir.mkdir()
        dinov3_dim = 128
        for stem in stems:
            feat = torch.randn(dinov3_dim, 12, 16)  # [C, H, W]
            torch.save(feat, str(dinov3_dir / f"{stem}_fmap_CxHxW.pt"))

        # RGB2X data (albedo=3ch, roughness=1ch)
        albedo_dir = tmp_path / "images_albedo"
        albedo_dir.mkdir()
        for stem in stems:
            img = np.random.randint(0, 255, (24, 32, 3), dtype=np.uint8)
            imageio.imwrite(str(albedo_dir / f"{stem}.png"), img)

        roughness_dir = tmp_path / "images_roughness"
        roughness_dir.mkdir()
        for stem in stems:
            img = np.random.randint(0, 255, (24, 32), dtype=np.uint8)
            imageio.imwrite(str(roughness_dir / f"{stem}.png"), img)

        return {
            "tmp_path": tmp_path,
            "img_dir": str(img_dir),
            "stems": stems,
            "lseg_dim": lseg_dim,
            "dinov3_dim": dinov3_dim,
            "num_images": len(stems),
        }

    def test_aov_dataset_config_defaults(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from aov.aov_dataset import AOVDatasetConfig
        cfg = AOVDatasetConfig()
        assert cfg.load_lseg is False
        assert cfg.load_dinov3 is False
        assert cfg.load_rgb2x is False

    def test_discovery_lseg(self, mock_aov_data):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from aov.aov_dataset import AOVDataset, AOVDatasetConfig

        class MockParser:
            image_paths = [
                os.path.join(mock_aov_data["img_dir"], f"{s}.png")
                for s in mock_aov_data["stems"]
            ]

        class MockBaseDataset:
            indices = list(range(mock_aov_data["num_images"]))
            def __len__(self): return len(self.indices)
            def __getitem__(self, item): return {"image_id": item}

        config = AOVDatasetConfig(load_lseg=True)
        ds = AOVDataset(MockBaseDataset(), config, MockParser())
        assert ds.lseg_feature_dim == mock_aov_data["lseg_dim"]
        assert len(ds.lseg_paths) == mock_aov_data["num_images"]

    def test_discovery_dinov3(self, mock_aov_data):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from aov.aov_dataset import AOVDataset, AOVDatasetConfig

        class MockParser:
            image_paths = [
                os.path.join(mock_aov_data["img_dir"], f"{s}.png")
                for s in mock_aov_data["stems"]
            ]

        class MockBaseDataset:
            indices = list(range(mock_aov_data["num_images"]))
            def __len__(self): return len(self.indices)
            def __getitem__(self, item): return {"image_id": item}

        config = AOVDatasetConfig(load_dinov3=True)
        ds = AOVDataset(MockBaseDataset(), config, MockParser())
        assert ds.dinov3_feature_dim == mock_aov_data["dinov3_dim"]

    def test_discovery_rgb2x(self, mock_aov_data):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from aov.aov_dataset import AOVDataset, AOVDatasetConfig

        class MockParser:
            image_paths = [
                os.path.join(mock_aov_data["img_dir"], f"{s}.png")
                for s in mock_aov_data["stems"]
            ]

        class MockBaseDataset:
            indices = list(range(mock_aov_data["num_images"]))
            def __len__(self): return len(self.indices)
            def __getitem__(self, item): return {"image_id": item}

        config = AOVDatasetConfig(load_rgb2x=True)
        ds = AOVDataset(MockBaseDataset(), config, MockParser())
        assert "albedo" in ds.rgb2x_channel_dims
        assert "roughness" in ds.rgb2x_channel_dims
        assert ds.rgb2x_channel_dims["albedo"] == 3
        assert ds.rgb2x_channel_dims["roughness"] == 1

    def test_getitem_appends_aov(self, mock_aov_data):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from aov.aov_dataset import AOVDataset, AOVDatasetConfig

        class MockParser:
            image_paths = [
                os.path.join(mock_aov_data["img_dir"], f"{s}.png")
                for s in mock_aov_data["stems"]
            ]

        class MockBaseDataset:
            indices = list(range(mock_aov_data["num_images"]))
            def __len__(self): return len(self.indices)
            def __getitem__(self, item): return {"image_id": item}

        config = AOVDatasetConfig(load_lseg=True, load_dinov3=True, load_rgb2x=True)
        ds = AOVDataset(MockBaseDataset(), config, MockParser())
        sample = ds[0]

        assert "lseg_features" in sample
        assert sample["lseg_features"].shape == (12, 16, mock_aov_data["lseg_dim"])
        assert sample["lseg_features"].dtype == torch.float32

        assert "dinov3_features" in sample
        assert sample["dinov3_features"].shape == (12, 16, mock_aov_data["dinov3_dim"])

        assert "rgb2x" in sample
        assert sample["rgb2x"].shape == (24, 32, 4)

    def test_get_aov_config(self, mock_aov_data):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from aov.aov_dataset import AOVDataset, AOVDatasetConfig

        class MockParser:
            image_paths = [
                os.path.join(mock_aov_data["img_dir"], f"{s}.png")
                for s in mock_aov_data["stems"]
            ]

        class MockBaseDataset:
            indices = list(range(mock_aov_data["num_images"]))
            def __len__(self): return len(self.indices)
            def __getitem__(self, item): return {"image_id": item}

        config = AOVDatasetConfig(load_lseg=True, load_rgb2x=True)
        ds = AOVDataset(MockBaseDataset(), config, MockParser())
        aov_cfg = ds.get_aov_config()
        assert aov_cfg["lseg_feature_dim"] == mock_aov_data["lseg_dim"]
        assert aov_cfg["dinov3_feature_dim"] == 0
        assert aov_cfg["rgb2x_channels"]["albedo"] == 3
        assert aov_cfg["rgb2x_channels"]["roughness"] == 1

    def test_missing_lseg_dir_raises(self, tmp_path):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from aov.aov_dataset import AOVDataset, AOVDatasetConfig

        img_dir = tmp_path / "images"
        img_dir.mkdir()

        class MockParser:
            image_paths = [str(img_dir / "img.png")]

        class MockBaseDataset:
            indices = [0]
            def __len__(self): return 1
            def __getitem__(self, item): return {}

        config = AOVDatasetConfig(load_lseg=True)
        with pytest.raises(ValueError, match="does not exist"):
            AOVDataset(MockBaseDataset(), config, MockParser())

    def test_len_delegates(self, mock_aov_data):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
        from aov.aov_dataset import AOVDataset, AOVDatasetConfig

        class MockParser:
            image_paths = [
                os.path.join(mock_aov_data["img_dir"], f"{s}.png")
                for s in mock_aov_data["stems"]
            ]

        class MockBaseDataset:
            indices = list(range(mock_aov_data["num_images"]))
            def __len__(self): return len(self.indices)
            def __getitem__(self, item): return {}

        config = AOVDatasetConfig()
        ds = AOVDataset(MockBaseDataset(), config, MockParser())
        assert len(ds) == mock_aov_data["num_images"]


# ===================================================================
# 6. End-to-end: rasterize + AOV decode
# ===================================================================

@pytest.mark.skipif(not _nht_raster_ok, reason="NHT rasterizer not available")
@pytest.mark.skipif(not _tcnn_available, reason="tinycudann not available")
class TestEndToEndAOVPipeline:

    def test_rasterize_and_decode_lseg(self):
        from gsplat.rendering import rasterization
        from aov.deferred_shader import DeferredShaderAOVModule

        feature_dim = 16
        lseg_dim = 32
        N = 128
        means, quats, scales, opacities, features = _make_splats(
            N=N, feature_dim=feature_dim, device=device
        )
        scales_act, opacities_act = torch.exp(scales), torch.sigmoid(opacities)

        K, c2w, W, H = _make_camera(device=device)
        viewmats = torch.linalg.inv(c2w)

        render_colors, render_alphas, info = rasterization(
            means=means, quats=quats, scales=scales_act, opacities=opacities_act,
            colors=features, viewmats=viewmats, Ks=K, width=W, height=H,
            nht=True, with_eval3d=True, with_ut=True, packed=False, sh_degree=None,
        )

        mod = DeferredShaderAOVModule(
            feature_dim=feature_dim,
            enable_view_encoding=True,
            lseg_feature_dim=lseg_dim,
        ).to(device)

        colors, aov_outputs, extras = mod(render_colors, K, c2w)
        assert colors.shape == (1, H, W, 3)
        assert "lseg" in aov_outputs
        assert aov_outputs["lseg"].shape == (1, H, W, lseg_dim)

    def test_rasterize_and_decode_rgb2x(self):
        from gsplat.rendering import rasterization
        from aov.deferred_shader import DeferredShaderAOVModule

        feature_dim = 16
        channels = {"albedo": 3, "roughness": 1, "normal": 3}
        N = 128
        means, quats, scales, opacities, features = _make_splats(
            N=N, feature_dim=feature_dim, device=device
        )
        scales_act, opacities_act = torch.exp(scales), torch.sigmoid(opacities)

        K, c2w, W, H = _make_camera(device=device)
        viewmats = torch.linalg.inv(c2w)

        render_colors, render_alphas, info = rasterization(
            means=means, quats=quats, scales=scales_act, opacities=opacities_act,
            colors=features, viewmats=viewmats, Ks=K, width=W, height=H,
            nht=True, with_eval3d=True, with_ut=True, packed=False, sh_degree=None,
        )

        mod = DeferredShaderAOVModule(
            feature_dim=feature_dim,
            enable_view_encoding=True,
            rgb2x_channels=channels,
        ).to(device)

        colors, aov_outputs, extras = mod(render_colors, K, c2w)
        assert colors.shape == (1, H, W, 3)
        total = sum(channels.values())
        assert "rgb2x" in aov_outputs
        assert aov_outputs["rgb2x"].shape == (1, H, W, total)
        parts = mod.split_rgb2x(aov_outputs["rgb2x"])
        for ch_name, ch_dim in channels.items():
            assert parts[f"rgb2x_{ch_name}"].shape == (1, H, W, ch_dim)

    def test_rasterize_and_decode_all(self):
        from gsplat.rendering import rasterization
        from aov.deferred_shader import DeferredShaderAOVModule

        feature_dim = 16
        N = 128
        means, quats, scales, opacities, features = _make_splats(
            N=N, feature_dim=feature_dim, device=device
        )
        scales_act, opacities_act = torch.exp(scales), torch.sigmoid(opacities)

        K, c2w, W, H = _make_camera(device=device)
        viewmats = torch.linalg.inv(c2w)

        render_colors, render_alphas, info = rasterization(
            means=means, quats=quats, scales=scales_act, opacities=opacities_act,
            colors=features, viewmats=viewmats, Ks=K, width=W, height=H,
            nht=True, with_eval3d=True, with_ut=True, packed=False, sh_degree=None,
        )

        channels = {"albedo": 3, "metallic": 1}
        mod = DeferredShaderAOVModule(
            feature_dim=feature_dim,
            enable_view_encoding=True,
            lseg_feature_dim=32,
            dinov3_feature_dim=48,
            rgb2x_channels=channels,
        ).to(device)

        colors, aov_outputs, extras = mod(render_colors, K, c2w)
        assert colors.shape == (1, H, W, 3)
        assert "lseg" in aov_outputs
        assert "dinov3" in aov_outputs
        assert "rgb2x" in aov_outputs
        parts = mod.split_rgb2x(aov_outputs["rgb2x"])
        assert "rgb2x_albedo" in parts
        assert "rgb2x_metallic" in parts

    def test_backward_end_to_end(self):
        from gsplat.rendering import rasterization
        from aov.deferred_shader import DeferredShaderAOVModule

        feature_dim = 16
        N = 128
        means = torch.randn(N, 3, device=device) * 0.3
        means.requires_grad = True
        quats = F.normalize(torch.randn(N, 4, device=device), dim=-1)
        scales = torch.exp(torch.rand(N, 3, device=device) * 0.5 - 2.0)
        opacities = torch.sigmoid(torch.logit(torch.full((N,), 0.5, device=device)))
        features = torch.randn(N, feature_dim, device=device, requires_grad=True)

        K, c2w, W, H = _make_camera(device=device)
        viewmats = torch.linalg.inv(c2w)

        render_colors, render_alphas, info = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=features, viewmats=viewmats, Ks=K, width=W, height=H,
            nht=True, with_eval3d=True, with_ut=True, packed=False, sh_degree=None,
        )

        mod = DeferredShaderAOVModule(
            feature_dim=feature_dim,
            enable_view_encoding=True,
            lseg_feature_dim=32,
        ).to(device)

        colors, aov_outputs, extras = mod(render_colors, K, c2w)
        target_rgb = torch.rand_like(colors)
        target_lseg = torch.randn_like(aov_outputs["lseg"])

        loss = F.l1_loss(colors, target_rgb) + F.l1_loss(aov_outputs["lseg"], target_lseg)
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0


# ===================================================================
# 7. Checkpoint round-trip with aov_config
# ===================================================================

@pytest.mark.skipif(not _nht_raster_ok, reason="NHT rasterizer not available")
@pytest.mark.skipif(not _tcnn_available, reason="tinycudann not available")
class TestCheckpointRoundTrip:

    def test_save_load_with_aov_config(self):
        from aov.deferred_shader import DeferredShaderAOVModule

        feature_dim = 16
        aov_config = {
            "lseg_feature_dim": 32,
            "dinov3_feature_dim": 0,
            "rgb2x_channels": {"albedo": 3, "roughness": 1},
        }

        mod = DeferredShaderAOVModule(
            feature_dim=feature_dim,
            enable_view_encoding=True,
            lseg_feature_dim=aov_config["lseg_feature_dim"],
            rgb2x_channels=aov_config["rgb2x_channels"],
        ).to(device)

        _, _, scales, opacities, features = _make_splats(
            N=64, feature_dim=feature_dim, device=device
        )

        ckpt_data = {
            "step": 1000,
            "splats": {"features": features},
            "deferred_module": mod.state_dict(),
            "aov_config": aov_config,
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(ckpt_data, f.name)
            loaded = torch.load(f.name, map_location=device, weights_only=True)

        assert loaded["aov_config"] == aov_config
        assert loaded["step"] == 1000

        mod2 = DeferredShaderAOVModule(
            feature_dim=feature_dim,
            enable_view_encoding=True,
            lseg_feature_dim=loaded["aov_config"]["lseg_feature_dim"],
            rgb2x_channels=loaded["aov_config"]["rgb2x_channels"],
        ).to(device)
        mod2.load_state_dict(loaded["deferred_module"])

        K, c2w, W, H = _make_camera(device=device)
        rendered = torch.randn(1, H, W, _deferred_raster_channels(mod), device=device)
        c1, a1, _ = mod(rendered, K, c2w)
        c2, a2, _ = mod2(rendered, K, c2w)
        torch.testing.assert_close(c1, c2)
        for key in a1:
            torch.testing.assert_close(a1[key], a2[key])

        os.unlink(f.name)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"] + sys.argv[1:]))
