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

"""AOV (Arbitrary Output Variable) dataset wrapper.

Wraps an existing Dataset and augments each sample with ground-truth AOV data
(LSEG-style features, DINOv3-style dense features, RGB2X material maps).
Channel presence and dimensionality are auto-detected by scanning AOV
directories for matching files.

**Precomputed maps only.** This package does not install or call LSEG, DINOv3/DINOv2,
OpenCLIP, or RGB2X. Prepare tensors/images with external tooling, then place
them under the directory layout.

The ``base_dataset`` and ``parser`` arguments accept any object that exposes
the expected attributes (``indices``, ``image_paths``, ``__getitem__``, etc.)
so that the wrapper is decoupled from a specific Dataset/Parser implementation.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import imageio.v2 as imageio
import numpy as np
import torch

RGB2X_DEFAULT_CHANNEL_DIMS = {
    "albedo": 3,
    "roughness": 1,
    "metallic": 1,
    "irradiance": 3,
    "normal": 3,
}

_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".exr"}
)

_FMAP_SUFFIX = "_fmap_CxHxW.pt"

# External projects for precomputing maps (not gsplat dependencies).
AOV_LSEG_RESOURCES = (
    "https://github.com/isl-org/lang-seg (LSEG); dense CLIP-style features can be "
    "exported with OpenCLIP in a separate environment: "
    "https://github.com/mlfoundations/open_clip"
)
AOV_DINOV3_RESOURCES = (
    "https://github.com/facebookresearch/dinov3 or "
    "https://github.com/facebookresearch/dinov2"
)
AOV_RGB2X_RESOURCES = "https://github.com/zheng95z/rgb2x"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scan_fmap_dir(directory: str) -> Dict[str, str]:
    """Scan *directory* for ``*_fmap_CxHxW.pt`` files.

    Returns ``{stem: full_path}`` where *stem* is the filename prefix before
    the ``_fmap_CxHxW.pt`` suffix.
    """
    lookup: Dict[str, str] = {}
    if not os.path.isdir(directory):
        return lookup
    for fname in os.listdir(directory):
        if fname.endswith(_FMAP_SUFFIX):
            stem = fname[: -len(_FMAP_SUFFIX)]
            lookup[stem] = os.path.join(directory, fname)
    return lookup


def _image_stem(path: str) -> Optional[str]:
    """Return the filename stem if *path* has an image extension, else None."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        return os.path.splitext(os.path.basename(path))[0]
    return None


# ---------------------------------------------------------------------------
# Config and Dataset
# ---------------------------------------------------------------------------

@dataclass
class AOVDatasetConfig:
    load_lseg: bool = False
    lseg_data_dir: Optional[str] = None

    load_dinov3: bool = False
    dinov3_data_dir: Optional[str] = None

    load_rgb2x: bool = False
    rgb2x_data_dir: Optional[str] = None


class AOVDataset(torch.utils.data.Dataset):
    """Wraps a base Dataset and appends AOV ground truth per sample.

    AOV file discovery is based on scanning the AOV directories for matching
    files and associating them with parser images by **filename stem**.  This
    is robust against non-image files that may be present in the dataset
    directories (which can corrupt ``colmap_to_image`` index ordering).

    ``base_dataset`` must expose ``indices`` and support ``__getitem__``/
    ``__len__``.  ``parser`` must expose ``image_paths``.
    """

    def __init__(
        self,
        base_dataset: Any,
        config: AOVDatasetConfig,
        parser: Any,
    ):
        self.base_dataset = base_dataset
        self.config = config
        self.parser = parser

        self._image_dir = os.path.dirname(parser.image_paths[0])

        self._stems: List[Optional[str]] = [
            _image_stem(p) for p in parser.image_paths
        ]

        # Auto-detect AOV channels
        self.lseg_feature_dim: int = 0
        self.lseg_paths: List[Optional[str]] = []

        self.dinov3_feature_dim: int = 0
        self.dinov3_paths: List[Optional[str]] = []

        self.rgb2x_channel_dims: Dict[str, int] = {}
        self.rgb2x_channel_paths: Dict[str, List[Optional[str]]] = {}

        if config.load_lseg:
            self._discover_lseg()
        if config.load_dinov3:
            self._discover_dinov3()
        if config.load_rgb2x:
            self._discover_rgb2x()

    # ------------------------------------------------------------------
    # Discovery (with optional generation)
    # ------------------------------------------------------------------

    def _discover_fmap(self, aov_dir: str, label: str) -> tuple:
        """Scan *aov_dir* for ``*_fmap_CxHxW.pt`` and match to parser images.

        Returns ``(paths, feature_dim)`` where *paths* is a list aligned to
        ``parser.image_paths`` (``None`` for unmatched entries).
        """
        if not os.path.isdir(aov_dir):
            hint = (
                AOV_LSEG_RESOURCES if label == "LSEG" else AOV_DINOV3_RESOURCES
            )
            raise ValueError(
                f"{label} directory {aov_dir} does not exist. "
                f"Expected a folder of *{_FMAP_SUFFIX} tensors aligned with "
                f"dataset image stems. Precompute features using an external "
                f"toolchain (not bundled here). See: {hint}"
            )

        lookup = _scan_fmap_dir(aov_dir)

        paths: List[Optional[str]] = [None] * len(self._stems)
        for idx, stem in enumerate(self._stems):
            if stem is not None and stem in lookup:
                paths[idx] = lookup[stem]

        found = sum(1 for p in paths if p is not None)
        if found == 0:
            raise ValueError(
                f"No {label} features matched in {aov_dir}. "
                f"Directory contains {len(lookup)} *{_FMAP_SUFFIX} files but "
                f"none matched parser image stems."
            )

        first_valid = next(p for p in paths if p is not None)
        probe = torch.load(first_valid, map_location="cpu", weights_only=True)
        feature_dim = probe.shape[0]

        print(f"[AOVDataset] {label}: {found}/{len(self._stems)} matched, "
              f"dim={feature_dim} from {aov_dir}")
        return paths, feature_dim

    def _discover_lseg(self):
        lseg_dir = self.config.lseg_data_dir or (self._image_dir + "_bin_lseg")
        self.lseg_paths, self.lseg_feature_dim = self._discover_fmap(
            lseg_dir, "LSEG"
        )

    def _discover_dinov3(self):
        dinov3_dir = self.config.dinov3_data_dir or (self._image_dir + "_dinov3")
        self.dinov3_paths, self.dinov3_feature_dim = self._discover_fmap(
            dinov3_dir, "DINOv3"
        )

    def _discover_rgb2x(self):
        rgb2x_base = self.config.rgb2x_data_dir or self._image_dir
        for ch_name in RGB2X_DEFAULT_CHANNEL_DIMS:
            ch_dir = rgb2x_base + f"_{ch_name}"

            if not os.path.isdir(ch_dir):
                print(
                    f"[AOVDataset] RGB2X channel '{ch_name}': missing directory "
                    f"{ch_dir}. Generate maps with the RGB2X project and place "
                    f"one image per dataset frame (matching stems). "
                    f"{AOV_RGB2X_RESOURCES}"
                )
                continue

            ch_paths: List[Optional[str]] = [None] * len(self._stems)
            all_found = True
            for idx, stem in enumerate(self._stems):
                if stem is None:
                    continue
                ch_path = None
                for ext in _IMAGE_EXTENSIONS:
                    candidate = os.path.join(ch_dir, f"{stem}{ext}")
                    if os.path.exists(candidate):
                        ch_path = candidate
                        break
                if ch_path is None:
                    print(f"[AOVDataset] Warning: RGB2X {ch_name} missing for "
                          f"{stem}, skipping channel.")
                    all_found = False
                    break
                ch_paths[idx] = ch_path

            if not all_found:
                continue
            found = sum(1 for p in ch_paths if p is not None)
            if found == 0:
                continue

            first_valid = next(p for p in ch_paths if p is not None)
            probe_img = imageio.imread(first_valid)
            if probe_img.ndim == 2:
                detected_dim = 1
            elif probe_img.shape[-1] == 4:
                detected_dim = 3  # RGBA -> RGB
            else:
                detected_dim = probe_img.shape[-1]

            self.rgb2x_channel_dims[ch_name] = detected_dim
            self.rgb2x_channel_paths[ch_name] = ch_paths
            print(f"[AOVDataset] RGB2X {ch_name}: {found}/{len(self._stems)} "
                  f"matched, dim={detected_dim} from {ch_dir}")

        if self.rgb2x_channel_dims:
            print(f"[AOVDataset] RGB2X channels detected: {self.rgb2x_channel_dims}")
        elif self.config.load_rgb2x:
            raise ValueError(
                "RGB2X requested (load_rgb2x=True) but no complete channel "
                "directories were found. Expected sibling folders "
                f"{rgb2x_base}_<channel>/ with per-image maps matching parser "
                f"stems. Use {AOV_RGB2X_RESOURCES} to produce maps."
            )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = self.base_dataset[item]

        # Map from base dataset item -> parser global index
        index = self.base_dataset.indices[item]

        if self.lseg_paths:
            lseg_path = self.lseg_paths[index]
            if lseg_path is not None:
                feat = torch.load(
                    lseg_path, map_location="cpu", weights_only=True,
                )  # [C, H, W]
                data["lseg_features"] = feat.permute(1, 2, 0).float()

        if self.dinov3_paths:
            dinov3_path = self.dinov3_paths[index]
            if dinov3_path is not None:
                feat = torch.load(
                    dinov3_path, map_location="cpu", weights_only=True,
                )  # [C, H, W]
                data["dinov3_features"] = feat.permute(1, 2, 0).float()

        rgb2x_parts = []
        for ch_name in sorted(self.rgb2x_channel_paths.keys()):
            ch_path = self.rgb2x_channel_paths[ch_name][index]
            if ch_path is None:
                continue
            ch_img = imageio.imread(ch_path)
            expected_dim = self.rgb2x_channel_dims[ch_name]
            if ch_img.ndim == 2:
                ch_img = ch_img[:, :, None]
            elif ch_img.shape[-1] == 4:
                ch_img = ch_img[..., :3]
            if expected_dim == 1 and ch_img.shape[-1] == 3:
                ch_img = ch_img[..., :1]
            rgb2x_parts.append(torch.from_numpy(ch_img).float())
        if rgb2x_parts:
            data["rgb2x"] = torch.cat(rgb2x_parts, dim=-1)

        return data

    def get_aov_config(self) -> Dict[str, Any]:
        """Return detected AOV dimensions for ``aov.deferred_shader.DeferredShaderAOVModule``."""
        cfg: Dict[str, Any] = {
            "lseg_feature_dim": self.lseg_feature_dim,
            "dinov3_feature_dim": self.dinov3_feature_dim,
            "rgb2x_channels": self.rgb2x_channel_dims if self.rgb2x_channel_dims else None,
        }
        return cfg
