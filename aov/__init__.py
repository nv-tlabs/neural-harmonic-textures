# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Superproject AOV stack: dataset, deferred heads, and training/view scripts.

Core NHT (rasterization, ``HarmonicFeatures``, ``DeferredShaderModule``) lives in
``gsplat.nht``. This package adds auxiliary-output training on top. Install editable
from the repo root (``pip install -e .``) so ``import aov`` resolves.
"""

from .aov_dataset import (
    AOVDataset,
    AOVDatasetConfig,
    AOV_DINOV3_RESOURCES,
    AOV_LSEG_RESOURCES,
    AOV_RGB2X_RESOURCES,
    RGB2X_DEFAULT_CHANNEL_DIMS,
)
from .deferred_shader import DeferredShaderAOVModule

__all__ = [
    "AOVDataset",
    "AOVDatasetConfig",
    "AOV_LSEG_RESOURCES",
    "AOV_DINOV3_RESOURCES",
    "AOV_RGB2X_RESOURCES",
    "RGB2X_DEFAULT_CHANNEL_DIMS",
    "DeferredShaderAOVModule",
]
