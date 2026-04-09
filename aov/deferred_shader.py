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

"""AOV (multi-head) deferred shading on top of gsplat :class:`DeferredShaderModuleAOV`."""

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from gsplat.nht.deferred_shader import DeferredShaderModuleAOV


def _activation_from_name(name: str) -> Callable[[Tensor], Tensor]:
    n = name.lower()
    if n in ("none", "identity", "linear"):
        return lambda t: t
    if n == "sigmoid":
        return torch.sigmoid
    if n == "tanh":
        return torch.tanh
    raise ValueError(f"Unknown activation name: {name!r}")


class DeferredShaderAOVModule(DeferredShaderModuleAOV):
    """Named AOV heads + activations on top of :class:`DeferredShaderModuleAOV`.

    Chooses the same layout as the historical trainer: semantic heads use the
    split RGB + linear-auxiliary path; rgb2x-only uses fused tcnn Sigmoid when
    ``3 + K < 128``; large bundles use full linear readout. When tcnn already
    applies Sigmoid (RGB-only or fused rgb+aux), wrapper RGB / rgb2x activations
    default to identity so outputs are not double-squashed.
    """

    def __init__(
        self,
        feature_dim: int,
        enable_view_encoding: bool,
        view_encoding_type: str = "sh",
        mlp_hidden_dim: int = 128,
        mlp_num_layers: int = 3,
        sh_degree: int = 3,
        sh_scale: float = 1.0,
        fourier_num_freqs: int = 4,
        primitive_type: str = "3dgs",
        center_ray_encoding: bool = False,
        lseg_feature_dim: int = 0,
        dinov3_feature_dim: int = 0,
        rgb2x_channels: Optional[Dict[str, int]] = None,
        rgb_activation: str = "sigmoid",
        rgb2x_activation: str = "sigmoid",
        semantic_activation: str = "none",
    ):
        self.lseg_feature_dim = lseg_feature_dim
        self.dinov3_feature_dim = dinov3_feature_dim
        self.rgb2x_channels = rgb2x_channels or {}

        self._rgb2x_names: List[str] = sorted(self.rgb2x_channels.keys())
        self._rgb2x_total_dim: int = sum(
            self.rgb2x_channels[n] for n in self._rgb2x_names
        )

        semantic_dim = lseg_feature_dim + dinov3_feature_dim
        auxiliary_output_dim = semantic_dim + self._rgb2x_total_dim

        self._semantic_dim = semantic_dim
        self._aov_keys: List[str] = []
        self._semantic_split_sizes: List[int] = []
        self._semantic_keys: List[str] = []
        if lseg_feature_dim > 0:
            self._semantic_split_sizes.append(lseg_feature_dim)
            self._semantic_keys.append("lseg")
            self._aov_keys.append("lseg")
        if dinov3_feature_dim > 0:
            self._semantic_split_sizes.append(dinov3_feature_dim)
            self._semantic_keys.append("dinov3")
            self._aov_keys.append("dinov3")
        if self._rgb2x_total_dim > 0:
            self._aov_keys.append("rgb2x")

        self._mode = "semantic" if semantic_dim > 0 else "rgb2x"

        if auxiliary_output_dim == 0:
            split_rgb_head = False
            fused_tcnn_sigmoid = False
            rgb_only_tcnn_sigmoid = True
        elif semantic_dim > 0:
            split_rgb_head = True
            fused_tcnn_sigmoid = False
            rgb_only_tcnn_sigmoid = True
        else:
            split_rgb_head = False
            fused_tcnn_sigmoid = (3 + auxiliary_output_dim) < 128
            rgb_only_tcnn_sigmoid = True

        super().__init__(
            feature_dim=feature_dim,
            enable_view_encoding=enable_view_encoding,
            auxiliary_output_dim=auxiliary_output_dim,
            view_encoding_type=view_encoding_type,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_num_layers=mlp_num_layers,
            sh_degree=sh_degree,
            sh_scale=sh_scale,
            fourier_num_freqs=fourier_num_freqs,
            primitive_type=primitive_type,
            center_ray_encoding=center_ray_encoding,
            split_rgb_head=split_rgb_head,
            fused_tcnn_sigmoid=fused_tcnn_sigmoid,
            rgb_only_tcnn_sigmoid=rgb_only_tcnn_sigmoid,
        )

        if self.tcnn_emitted_sigmoid_outputs:
            eff_rgb = "none"
            eff_rgb2x = "none"
        else:
            eff_rgb = rgb_activation
            eff_rgb2x = rgb2x_activation

        self._apply_rgb = _activation_from_name(eff_rgb)
        self._apply_rgb2x = _activation_from_name(eff_rgb2x)
        self._apply_semantic = _activation_from_name(semantic_activation)

    @property
    def rgb2x_channel_layout(self) -> List[Tuple[str, int]]:
        return [(n, self.rgb2x_channels[n]) for n in self._rgb2x_names]

    def split_rgb2x(self, rgb2x: Tensor) -> Dict[str, Tensor]:
        sizes = [self.rgb2x_channels[n] for n in self._rgb2x_names]
        parts = rgb2x.split(sizes, dim=-1)
        return {f"rgb2x_{n}": p for n, p in zip(self._rgb2x_names, parts)}

    def forward(
        self,
        rendered_data: Tensor,
        Ks: Optional[Tensor] = None,
        camtoworlds: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        rgb_raw, aux_raw, extras = super().forward(rendered_data, Ks, camtoworlds)
        colors = self._apply_rgb(rgb_raw)

        aov_outputs: Dict[str, Tensor] = {}
        if aux_raw is None:
            return colors, aov_outputs, extras

        offset = 0
        for key, sz in zip(self._semantic_keys, self._semantic_split_sizes):
            part = aux_raw[..., offset : offset + sz]
            part = self._apply_semantic(part)
            aov_outputs[key] = part
            offset += sz

        if self._rgb2x_total_dim > 0:
            rgb2x_raw = aux_raw[..., offset : offset + self._rgb2x_total_dim]
            aov_outputs["rgb2x"] = self._apply_rgb2x(rgb2x_raw)

        return colors, aov_outputs, extras
