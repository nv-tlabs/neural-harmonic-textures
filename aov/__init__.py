# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Superproject AOV helpers (dataset wrapper and training/view scripts).

Deferred shading and core NHT logic remain in the ``gsplat`` submodule under
``gsplat.nht``. Install this package editable from the repo root so
``import aov`` resolves (``pip install -e .``).
"""

from .aov_dataset import (
    AOVDataset,
    AOVDatasetConfig,
    AOV_DINOV3_RESOURCES,
    AOV_LSEG_RESOURCES,
    AOV_RGB2X_RESOURCES,
    RGB2X_DEFAULT_CHANNEL_DIMS,
)

__all__ = [
    "AOVDataset",
    "AOVDatasetConfig",
    "AOV_LSEG_RESOURCES",
    "AOV_DINOV3_RESOURCES",
    "AOV_RGB2X_RESOURCES",
    "RGB2X_DEFAULT_CHANNEL_DIMS",
]
