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

import threading
from typing import Callable, List, Literal, Optional

import viser
from nerfview import Viewer

from gsplat_viewer_nht import GsplatNHTViewer, GsplatNHTRenderTabState


class GsplatNHTAOVRenderTabState(GsplatNHTRenderTabState):
    """Render tab state for NHT AOV viewer with semantic segmentation support."""

    aov_modes: tuple = ()

    seg_text_query: str = ""
    seg_threshold: float = 0.5
    seg_enabled: bool = False
    seg_overlay_alpha: float = 0.6

    # Mutable state not tracked via dataclass fields; managed externally.
    # seg_click_feature, seg_similarity_map stored on the main script side.


class GsplatNHTAOVViewer(GsplatNHTViewer):
    """Viewer for gsplat NHT AOV models with PCA, RGB2X, and segmentation UI."""

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
        aov_modes: Optional[List[str]] = None,
        has_lseg: bool = False,
        seg_callbacks: Optional[dict] = None,
    ):
        self._aov_modes = tuple(aov_modes or [])
        self._has_lseg = has_lseg
        self._seg_callbacks = seg_callbacks or {}
        self._click_seg_active = False
        self._seg_lock = threading.Lock()
        self._seg_enabled_checkbox = None
        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat NHT AOV viewer")

    def _init_rendering_tab(self):
        self.render_tab_state = GsplatNHTAOVRenderTabState()
        self.render_tab_state.aov_modes = self._aov_modes
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        server = self.server

        base_modes = ("rgb", "depth(accumulated)", "depth(expected)", "alpha")
        all_modes = base_modes + self._aov_modes

        with self._rendering_folder:
            with server.gui.add_folder("Gsplat NHT AOV"):
                total_gs_count_number = server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_gs_count,
                    disabled=True,
                    hint="Total number of splats in the scene.",
                )
                rendered_gs_count_number = server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_gs_count,
                    disabled=True,
                    hint="Number of splats rendered.",
                )

                near_far_plane_vec2 = server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane for rendering.",
                )

                @near_far_plane_vec2.on_update
                def _(_) -> None:
                    self.render_tab_state.near_plane = near_far_plane_vec2.value[0]
                    self.render_tab_state.far_plane = near_far_plane_vec2.value[1]
                    self.rerender(_)

                radius_clip_slider = server.gui.add_number(
                    "Radius Clip",
                    initial_value=self.render_tab_state.radius_clip,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    hint="2D radius clip for rendering.",
                )

                @radius_clip_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radius_clip = radius_clip_slider.value
                    self.rerender(_)

                eps2d_slider = server.gui.add_number(
                    "2D Epsilon",
                    initial_value=self.render_tab_state.eps2d,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    hint="Epsilon added to the eigenvalues of projected 2D covariance matrices.",
                )

                @eps2d_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.eps2d = eps2d_slider.value
                    self.rerender(_)

                backgrounds_slider = server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )

                @backgrounds_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.backgrounds = backgrounds_slider.value
                    self.rerender(_)

                render_mode_dropdown = server.gui.add_dropdown(
                    "Render Mode",
                    all_modes,
                    initial_value=self.render_tab_state.render_mode,
                    hint="Render mode to use.",
                )

                @render_mode_dropdown.on_update
                def _(_) -> None:
                    mode_val = render_mode_dropdown.value
                    is_depth = "depth" in mode_val
                    is_alpha = mode_val == "alpha"
                    normalize_nearfar_checkbox.disabled = not is_depth
                    inverse_checkbox.disabled = not (is_depth or is_alpha)
                    self.render_tab_state.render_mode = mode_val
                    self.rerender(_)

                normalize_nearfar_checkbox = server.gui.add_checkbox(
                    "Normalize Near/Far",
                    initial_value=self.render_tab_state.normalize_nearfar,
                    disabled=True,
                    hint="Normalize depth with near/far plane.",
                )

                @normalize_nearfar_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.normalize_nearfar = (
                        normalize_nearfar_checkbox.value
                    )
                    self.rerender(_)

                inverse_checkbox = server.gui.add_checkbox(
                    "Inverse",
                    initial_value=self.render_tab_state.inverse,
                    disabled=True,
                    hint="Inverse the depth.",
                )

                @inverse_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.inverse = inverse_checkbox.value
                    self.rerender(_)

                colormap_dropdown = server.gui.add_dropdown(
                    "Colormap",
                    ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                    initial_value=self.render_tab_state.colormap,
                    hint="Colormap used for rendering depth/alpha.",
                )

                @colormap_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.colormap = colormap_dropdown.value
                    self.rerender(_)

                rasterize_mode_dropdown = server.gui.add_dropdown(
                    "Anti-Aliasing",
                    ("classic", "antialiased"),
                    initial_value=self.render_tab_state.rasterize_mode,
                    hint="Whether to use classic or antialiased rasterization.",
                )

                @rasterize_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.rasterize_mode = rasterize_mode_dropdown.value
                    self.rerender(_)

                camera_model_dropdown = server.gui.add_dropdown(
                    "Camera",
                    ("pinhole", "ortho", "fisheye"),
                    initial_value=self.render_tab_state.camera_model,
                    hint="Camera model used for rendering.",
                )

                @camera_model_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.camera_model = camera_model_dropdown.value
                    self.rerender(_)

            # --- Segmentation folder (only when LSEG features are available) ---
            if self._has_lseg:
                with server.gui.add_folder(
                    "Segmentation", expand_by_default=False
                ):
                    seg_text_input = server.gui.add_text(
                        "Text Query",
                        initial_value=self.render_tab_state.seg_text_query,
                        hint='Type a semantic label (e.g. "grass", "building").',
                    )

                    @seg_text_input.on_update
                    def _(_) -> None:
                        self.render_tab_state.seg_text_query = seg_text_input.value

                    seg_query_button = server.gui.add_button(
                        "Run Text Query",
                        hint="Compute segmentation from the text query using CLIP.",
                        color="blue",
                    )

                    @seg_query_button.on_click
                    def _(_) -> None:
                        cb = self._seg_callbacks.get("on_text_query")
                        if cb is not None:
                            cb(self.render_tab_state.seg_text_query)
                        self.render_tab_state.seg_enabled = True
                        seg_enabled_checkbox.value = True
                        self.rerender(_)

                    seg_click_checkbox = server.gui.add_checkbox(
                        "Click to Segment",
                        initial_value=False,
                        hint="Enable click-to-segment mode. Click a pixel to find similar regions.",
                    )

                    @seg_click_checkbox.on_update
                    def _(_) -> None:
                        if seg_click_checkbox.value:
                            self._enable_click_seg()
                        else:
                            self._disable_click_seg()

                    seg_threshold_slider = server.gui.add_slider(
                        "Threshold",
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        initial_value=self.render_tab_state.seg_threshold,
                        hint="Cosine similarity threshold for segmentation.",
                    )

                    @seg_threshold_slider.on_update
                    def _(_) -> None:
                        self.render_tab_state.seg_threshold = seg_threshold_slider.value
                        self.rerender(_)

                    seg_alpha_slider = server.gui.add_slider(
                        "Overlay Alpha",
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        initial_value=self.render_tab_state.seg_overlay_alpha,
                        hint="Transparency of the segmentation overlay.",
                    )

                    @seg_alpha_slider.on_update
                    def _(_) -> None:
                        self.render_tab_state.seg_overlay_alpha = seg_alpha_slider.value
                        self.rerender(_)

                    seg_enabled_checkbox = server.gui.add_checkbox(
                        "Show Overlay",
                        initial_value=self.render_tab_state.seg_enabled,
                        hint="Toggle segmentation overlay on the render output.",
                    )
                    self._seg_enabled_checkbox = seg_enabled_checkbox

                    @seg_enabled_checkbox.on_update
                    def _(_) -> None:
                        self.render_tab_state.seg_enabled = seg_enabled_checkbox.value
                        self.rerender(_)

                    seg_clear_button = server.gui.add_button(
                        "Clear Segmentation",
                        hint="Clear the current segmentation overlay.",
                        color="red",
                    )

                    @seg_clear_button.on_click
                    def _(_) -> None:
                        cb = self._seg_callbacks.get("on_clear")
                        if cb is not None:
                            cb()
                        seg_enabled_checkbox.value = False
                        self.render_tab_state.seg_enabled = False
                        self.rerender(_)

        self._rendering_tab_handles.update(
            {
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
                "near_far_plane_vec2": near_far_plane_vec2,
                "radius_clip_slider": radius_clip_slider,
                "eps2d_slider": eps2d_slider,
                "backgrounds_slider": backgrounds_slider,
                "render_mode_dropdown": render_mode_dropdown,
                "normalize_nearfar_checkbox": normalize_nearfar_checkbox,
                "inverse_checkbox": inverse_checkbox,
                "colormap_dropdown": colormap_dropdown,
                "rasterize_mode_dropdown": rasterize_mode_dropdown,
                "camera_model_dropdown": camera_model_dropdown,
            }
        )
        Viewer._populate_rendering_tab(self)

    # ------------------------------------------------------------------
    # Click-to-segment helpers
    # ------------------------------------------------------------------

    def _enable_click_seg(self) -> None:
        with self._seg_lock:
            if self._click_seg_active:
                return
            self._click_seg_active = True

        @self.server.scene.on_pointer_event("click")
        def _pointer_cb(event: viser.ScenePointerEvent) -> None:
            screen_pos = event.screen_pos[0]  # (u_norm, v_norm) in [0, 1]
            cb = self._seg_callbacks.get("on_click")
            if cb is not None:
                cb(screen_pos)
            self.render_tab_state.seg_enabled = True
            if self._seg_enabled_checkbox is not None:
                self._seg_enabled_checkbox.value = True
            self.rerender(event)

    def _disable_click_seg(self) -> None:
        with self._seg_lock:
            if not self._click_seg_active:
                return
            self._click_seg_active = False
        self.server.scene.remove_pointer_callback()

    def _after_render(self):
        self._rendering_tab_handles[
            "total_gs_count_number"
        ].value = self.render_tab_state.total_gs_count
        self._rendering_tab_handles[
            "rendered_gs_count_number"
        ].value = self.render_tab_state.rendered_gs_count
