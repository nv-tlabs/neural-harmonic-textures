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

"""NHT Rendering Benchmark.

Measures execution time of the NHT rendering pipeline using CUDA events
for accurate GPU timing.

By DEFAULT the fully-fused inference path is timed end-to-end: one
``NHTInferenceRenderer`` launch per frame (projection + binning + the
fused rasterize+MLP kernel). Pass ``--unfused`` to instead time the
legacy two-stage path with a per-component breakdown — rasterization and
the tcnn MLP are timed in SEPARATE passes to avoid GPU power-state
coupling (a heavy rasterization kernel draws enough power to throttle GPU
clocks, making the immediately-following MLP appear slower than it really
is). The benchmark auto-falls back to the unfused breakdown when the
checkpoint's shader config has no compiled fused instantiation.

Pass ``--train`` for the fused-vs-tcnn comparison: forward-only (fused
``NHTInferenceRenderer`` vs the two-stage rasterization + tcnn MLP) and a
full training step (``nht_fused_render`` fwd+bwd vs the rasterizer/tcnn
autograd path). This reports per-path timings and speedups instead of the
inference-only breakdown.

Each scene is benchmarked in an isolated subprocess.

Usage (single scene, fused — default):
    python benchmark_nht.py --ckpt results/garden/ckpts/ckpt_29999_rank0.pt \
        --data_dir data/360_v2/garden --data_factor 4

Usage (single scene, unfused component breakdown):
    python benchmark_nht.py --unfused \
        --ckpt results/garden/ckpts/ckpt_29999_rank0.pt \
        --data_dir data/360_v2/garden --data_factor 4

Usage (single scene, fused vs tcnn forward + training step):
    python benchmark_nht.py --train \
        --ckpt results/garden/ckpts/ckpt_29999_rank0.pt \
        --data_dir data/360_v2/garden --data_factor 4

Usage (all scenes under a results folder):
    python benchmark_nht.py --results_dir results/nht_mcmc \
        --scene_dir data/360_v2

Usage (collect previously saved timing JSONs without re-running):
    python benchmark_nht.py --results_dir results/nht_mcmc --collect_only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict


def _prepend_gsplat_examples_path() -> None:
    """Colmap dataset loaders live under gsplat/examples/datasets (not an installable package)."""
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(bench_dir)
    examples = os.path.join(repo_root, "gsplat", "examples")
    if os.path.isdir(examples):
        ap = os.path.abspath(examples)
        if ap not in sys.path:
            sys.path.insert(0, ap)


_prepend_gsplat_examples_path()


def _subprocess_cwd_for_gsplat_jit() -> str | None:
    """Windows JIT in gsplat_internal uses sources as paths relative to the gsplat_internal root."""
    if sys.platform != "win32":
        return None
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(bench_dir)
    gsi = os.path.join(repo_root, "gsplat_internal")
    marker = os.path.join(gsi, "gsplat", "cuda", "csrc", "AdamCUDA.cu")
    if os.path.isfile(marker):
        return gsi
    return None


M360_INDOOR = {"bonsai", "counter", "kitchen", "room"}
M360_OUTDOOR = {"garden", "bicycle", "stump", "treehill", "flowers"}
TANDT_SCENES = {"train", "truck"}
DB_SCENES = {"drjohnson", "playroom"}
INDOOR_SCENES = M360_INDOOR  # backward compat
TIMING_KEYS = ["rasterization", "deferred_mlp", "total"]
# Keys used by the fused-vs-tcnn comparison (--train).
TRAIN_TIMING_KEYS = ["fwd_fused", "fwd_tcnn", "fwd_bwd_fused", "fwd_bwd_tcnn"]


def _time_once(fn) -> float:
    """Time a single GPU call with CUDA events (ms)."""
    import torch

    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def _cuda_time(fn, n_warmup: int, n_iters: int) -> float:
    """Mean GPU time (ms) over ``n_iters`` calls after ``n_warmup`` warmups."""
    import torch

    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    total = 0.0
    for _ in range(n_iters):
        total += _time_once(fn)
    return total / n_iters if n_iters else 0.0


def get_scene_factor(scene: str) -> int:
    if scene in M360_INDOOR:
        return 2
    if scene in M360_OUTDOOR:
        return 4
    if scene in TANDT_SCENES or scene in DB_SCENES:
        return 1
    return 4


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------

def load_timing_json(results_dir, scene_name):
    path = os.path.join(results_dir, scene_name, "stats", "timing.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_timing_json(out_path, data):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def aggregate_timing(all_results, scene_list):
    m360_in  = [s for s in scene_list if s in M360_INDOOR and s in all_results]
    m360_out = [s for s in scene_list if s in M360_OUTDOOR and s in all_results]
    m360_all = m360_in + m360_out
    tandt    = [s for s in scene_list if s in TANDT_SCENES and s in all_results]
    db       = [s for s in scene_list if s in DB_SCENES and s in all_results]
    all_valid = [s for s in scene_list if s in all_results]

    def _avg_row(label, scenes):
        if not scenes:
            return None
        row = OrderedDict([("split", label), ("n", len(scenes))])
        for k in TIMING_KEYS:
            key = f"{k}_ms"
            vals = [all_results[s].get(key, all_results[s].get(k, 0)) for s in scenes]
            row[key] = sum(vals) / len(vals) if vals else 0
        # Fused rows carry no raster/MLP split (both 0) -> no separate overhead.
        if row["rasterization_ms"] == 0 and row["deferred_mlp_ms"] == 0:
            row["overhead_ms"] = 0.0
        else:
            row["overhead_ms"] = row["total_ms"] - row["rasterization_ms"] - row["deferred_mlp_ms"]
        row["fps"] = 1000.0 / row["total_ms"] if row["total_ms"] > 0 else 0
        rml = row["rasterization_ms"] + row["deferred_mlp_ms"]
        row["fps_raster_mlp"] = 1000.0 / rml if rml > 0 else 0
        avg_gs = sum(all_results[s].get("num_gs", 0) for s in scenes) / len(scenes)
        row["avg_num_gs"] = int(avg_gs)
        return row

    rows = []
    for label, subset in [
        ("M360-In", m360_in), ("M360-Out", m360_out), ("M360", m360_all),
        ("T&T", tandt), ("DB", db), ("Overall", all_valid),
    ]:
        r = _avg_row(label, subset)
        if r:
            rows.append(r)
    return rows


def print_scene_results(name, d):
    if d.get("mode") == "train" or "fwd_bwd_fused_ms" in d:
        _print_training_scene(name, d)
        return
    total = d.get("total_ms", 0) or d.get("total", 0)
    raster = d.get("rasterization_ms", 0) or d.get("rasterization", 0)
    mlp = d.get("deferred_mlp_ms", 0) or d.get("deferred_mlp", 0)
    overhead = total - raster - mlp
    num_gs = d.get("num_gs", 0)
    w, h = d.get("width", 0), d.get("height", 0)
    n_img = d.get("num_images", 0)
    mode = d.get("mode", "fused" if (raster == 0 and mlp == 0) else "unfused")

    print(f"\n  {name} ({num_gs} GS, {w}x{h}, {n_img} imgs) [{mode}]")
    if mode != "fused":
        for label, val in [("rasterization", raster), ("deferred_mlp", mlp)]:
            pct = val / total * 100 if total > 0 else 0
            print(f"    {label:23s}: {val:8.2f} ms  ({pct:5.1f}%)")
        if abs(overhead) > 0.01:
            print(f"    {'overhead':23s}: {overhead:8.2f} ms  ({overhead/total*100:5.1f}%)")
    fps = 1000.0 / total if total > 0 else 0
    print(f"    {'total':23s}: {total:8.2f} ms  ({fps:.1f} FPS)")


def _print_training_scene(name, d):
    ff = d.get("fwd_fused_ms", 0)
    ft = d.get("fwd_tcnn_ms", 0)
    fbf = d.get("fwd_bwd_fused_ms", 0)
    fbt = d.get("fwd_bwd_tcnn_ms", 0)
    num_gs = d.get("num_gs", 0)
    w, h = d.get("width", 0), d.get("height", 0)
    n_img = d.get("num_images", 0)

    print(f"\n  {name} ({num_gs:,} GS, {w}x{h}, {n_img} imgs) [train]")
    print("    --- forward only (inference) ---")
    print(f"    {'fwd fused':23s}: {ff:8.2f} ms  ({1000.0/ff:.1f} FPS)" if ff > 0
          else f"    {'fwd fused':23s}: n/a")
    print(f"    {'fwd tcnn':23s}: {ft:8.2f} ms  ({1000.0/ft:.1f} FPS)" if ft > 0
          else f"    {'fwd tcnn':23s}: n/a")
    if ff > 0 and ft > 0:
        print(f"    {'fwd speedup':23s}: {ft/ff:8.2f}x")
    print("    --- forward + backward (training step) ---")
    print(f"    {'fwd+bwd fused':23s}: {fbf:8.2f} ms  ({1000.0/fbf:.1f} it/s)" if fbf > 0
          else f"    {'fwd+bwd fused':23s}: n/a")
    print(f"    {'fwd+bwd tcnn':23s}: {fbt:8.2f} ms  ({1000.0/fbt:.1f} it/s)" if fbt > 0
          else f"    {'fwd+bwd tcnn':23s}: n/a")
    if fbf > 0 and fbt > 0:
        print(f"    {'fwd+bwd speedup':23s}: {fbt/fbf:8.2f}x")


def aggregate_training(all_results, scene_list):
    groups = [
        ("M360-In", M360_INDOOR), ("M360-Out", M360_OUTDOOR),
        ("M360", M360_INDOOR | M360_OUTDOOR),
        ("T&T", TANDT_SCENES), ("DB", DB_SCENES), ("Overall", None),
    ]
    rows = []
    for label, members in groups:
        if members is None:
            scenes = [s for s in scene_list if s in all_results]
        else:
            scenes = [s for s in scene_list if s in members and s in all_results]
        if not scenes:
            continue
        row = OrderedDict([("split", label), ("n", len(scenes))])
        for k in TRAIN_TIMING_KEYS:
            vals = [all_results[s].get(f"{k}_ms", 0) for s in scenes]
            row[f"{k}_ms"] = sum(vals) / len(vals) if vals else 0
        ff, ft = row["fwd_fused_ms"], row["fwd_tcnn_ms"]
        fbf, fbt = row["fwd_bwd_fused_ms"], row["fwd_bwd_tcnn_ms"]
        row["fwd_speedup"] = ft / ff if ff > 0 else 0
        row["fwd_bwd_speedup"] = fbt / fbf if fbf > 0 else 0
        row["avg_num_gs"] = int(
            sum(all_results[s].get("num_gs", 0) for s in scenes) / len(scenes))
        rows.append(row)
    return rows


def print_training_table(rows):
    print(f"\n{'='*100}")
    print("  Aggregated Fused-vs-tcnn Timing Results")
    print(f"{'='*100}")
    print("  | Split    |  N | fwd_fused | fwd_tcnn | fwd_x | fb_fused | fb_tcnn | fwd_bwd_x |   Avg #GS |")
    print("  |----------|----|-----------|----------|-------|----------|---------|-----------|-----------|")
    for r in rows:
        print(f"  | {r['split']:<8} | {r['n']:>2} | {r['fwd_fused_ms']:>9.2f} | "
              f"{r['fwd_tcnn_ms']:>8.2f} | {r['fwd_speedup']:>4.2f}x | "
              f"{r['fwd_bwd_fused_ms']:>8.2f} | {r['fwd_bwd_tcnn_ms']:>7.2f} | "
              f"{r['fwd_bwd_speedup']:>8.2f}x | {r['avg_num_gs']:>9,} |")
    print(f"{'='*100}\n")


def print_aggregation_table(rows):
    print(f"\n{'='*80}")
    print(f"  Aggregated Timing Results")
    print(f"{'='*80}")
    hdr = (f"  {'Split':<10} {'N':>3}  {'Raster(ms)':>11}  {'MLP(ms)':>9}  "
           f"{'Over(ms)':>9}  {'Total(ms)':>10}  {'FPS':>7}  {'FPS(R+M)':>9}  {'Avg #GS':>10}")
    print(hdr)
    print(f"  {'-'*len(hdr.strip())}")
    for r in rows:
        print(f"  {r['split']:<10} {r['n']:>3}  {r['rasterization_ms']:>11.2f}  {r['deferred_mlp_ms']:>9.2f}"
              f"  {r['overhead_ms']:>9.2f}  {r['total_ms']:>10.2f}  {r['fps']:>7.1f}  {r['fps_raster_mlp']:>9.1f}"
              f"  {r['avg_num_gs']:>10,}")
    print()
    print("  --- Markdown table ---")
    print("  | Split    |  N | Raster(ms) | MLP(ms) | Overhead(ms) | Total(ms) |   FPS | FPS(R+M) | Avg #GS   |")
    print("  |----------|----|------------|---------|--------------|-----------|-------|----------|-----------|")
    for r in rows:
        print(f"  | {r['split']:<8} | {r['n']:>2} | {r['rasterization_ms']:>10.2f} | {r['deferred_mlp_ms']:>7.2f}"
              f" | {r['overhead_ms']:>12.2f} | {r['total_ms']:>9.2f} | {r['fps']:>5.1f} | {r['fps_raster_mlp']:>8.1f}"
              f" | {r['avg_num_gs']:>9,} |")
    print(f"{'='*80}\n")


def save_summary_json(results_dir, all_results, rows):
    out_path = os.path.join(results_dir, "timing_summary.json")
    data = OrderedDict([
        ("per_scene", {s: all_results[s] for s in sorted(all_results)}),
        ("aggregated", rows),
    ])
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Summary saved: {out_path}")


# ---------------------------------------------------------------------------
#  Single-scene benchmark  (runs in a subprocess when launched from batch)
# ---------------------------------------------------------------------------

def _run_scene_benchmark(args):
    from collections import defaultdict
    import torch
    from gsplat.nht import NHTParams
    from gsplat.nht.deferred_shader import DeferredShaderModule
    from gsplat.rendering import rasterization

    def _t(x):
        return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)

    def _load_dm(ckpt, splats, args, device):
        import yaml

        dm_state = ckpt.get("deferred_module")
        if dm_state is None:
            raise ValueError("Checkpoint has no deferred_module state")
        if isinstance(dm_state, dict) and "config" in dm_state:
            dm = DeferredShaderModule(**dm_state["config"]).to(device)
            dm.load_state_dict(dm_state["state_dict"])
            ema = dm_state.get("ema")
        else:
            cfg_path = os.path.join(
                os.path.dirname(os.path.dirname(args.ckpt)), "cfg.yml"
            )
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
                fd = cfg.get("deferred_opt_feature_dim", splats["features"].shape[-1])
                dm = DeferredShaderModule(
                    feature_dim=fd,
                    enable_view_encoding=cfg.get("deferred_opt_enable_view_encoding", True),
                    view_encoding_type=cfg.get("deferred_opt_view_encoding_type", "sh"),
                    mlp_hidden_dim=cfg.get("deferred_mlp_hidden_dim", 128),
                    mlp_num_layers=cfg.get("deferred_mlp_num_layers", 3),
                    sh_degree=cfg.get("deferred_opt_sh_degree", 3),
                    sh_scale=cfg.get("deferred_opt_sh_scale", 3.0),
                    fourier_num_freqs=cfg.get("deferred_opt_fourier_num_freqs", 4),
                    center_ray_encoding=cfg.get("deferred_opt_center_ray_encoding", True),
                    decode_activation=cfg.get("deferred_decode_activation", "sigmoid"),
                ).to(device)
            else:
                nf = splats["features"].shape[-1]
                fd = args.feature_dim if args.feature_dim > 0 else nf
                dm = DeferredShaderModule(
                    feature_dim=fd,
                    enable_view_encoding=args.enable_view_encoding,
                    view_encoding_type=args.view_encoding_type,
                    sh_degree=args.sh_degree,
                    sh_scale=args.sh_scale,
                    center_ray_encoding=args.center_ray_encoding,
                ).to(device)
            dm.load_state_dict(dm_state)
            ema = ckpt.get("deferred_ema")
        dm.eval()
        if ema is not None:
            for n, p in dm.named_parameters():
                if n in ema:
                    p.data.copy_(ema[n])
        return dm

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    from datasets.colmap import Dataset, Parser

    ckpt = torch.load(args.ckpt, map_location=device)
    splats = {k: v.to(device) for k, v in ckpt["splats"].items()}
    num_gs = splats["means"].shape[0]
    dm = _load_dm(ckpt, splats, args, device)

    factor = args.data_factor if args.data_factor > 0 else get_scene_factor(
        args.scene_name or os.path.basename(args.data_dir))
    parser = Parser(data_dir=args.data_dir, factor=factor, normalize=True)
    dataset = Dataset(parser, split="val")
    n_val = len(dataset)

    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = splats["features"]
    width = height = 0

    def _prepare(i):
        nonlocal width, height
        data = dataset[i]
        c2w = _t(data["camtoworld"]).float().to(device)[None]
        K = _t(data["K"]).float().to(device)[None]
        img = _t(data["image"]).float().to(device)[None] / 255.0
        _, height, width, _ = img.shape
        vm = torch.linalg.inv(c2w)
        return K, vm

    def _rasterize(K, vm):
        return rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=vm, Ks=K,
            width=width, height=height, tile_size=16,
            packed=False, absgrad=False,
            # 3DGUT (with_ut + with_eval3d, the NHT path) only supports
            # rasterize_mode="classic"; this mirrors the trainer's default.
            rasterize_mode="classic", render_mode="RGB",
            camera_model="pinhole",
            with_ut=True, with_eval3d=True,
            nht_params=NHTParams(
                center_ray_mode=dm.center_ray_encoding,
                ray_dir_scale=dm.ray_dir_scale,
            ),
        )

    # ------------------------------------------------------------------
    #  Fused-vs-tcnn comparison (--train): forward-only (fused inference
    #  renderer vs two-stage rasterization + tcnn MLP) and a full training
    #  step (nht_fused_render fwd+bwd vs the rasterizer/tcnn autograd path).
    # ------------------------------------------------------------------
    if args.train:
        import torch.nn.functional as F
        from gsplat.nht import nht_fused_render, nht_fused_supported
        from gsplat.nht._inference_renderer import (
            NHTInferenceConfig,
            NHTInferenceRenderer,
        )

        supported, reason = nht_fused_supported(dm, for_training=True)
        if not supported:
            raise SystemExit(
                f"  --train requires a fused-trainable shader config; "
                f"unsupported: {reason}"
            )

        features = colors.half()
        renderer = NHTInferenceRenderer(
            dm,
            NHTInferenceConfig(tile_size=16, center_ray_mode=dm.center_ray_encoding),
        )
        splats_dict = dict(splats)

        def _tcnn_forward(K, vm):
            rc, _, _ = rasterization(
                means=means, quats=F.normalize(quats, dim=-1),
                scales=scales, opacities=opacities, colors=features,
                viewmats=vm, Ks=K, width=width, height=height,
                sh_degree=None, near_plane=0.01, far_plane=1e10, packed=False,
                with_ut=True, with_eval3d=True,
                nht_params=NHTParams(
                    center_ray_mode=dm.center_ray_encoding,
                    ray_dir_scale=dm.ray_dir_scale,
                ),
                render_mode="RGB",
            )
            feat_ray = rc[0].reshape(-1, rc.shape[-1])
            out = dm._run_backbone(feat_ray.half())
            if not dm.tcnn_emitted_sigmoid_outputs:
                out = torch.sigmoid(out.float())
            return out[:, :3].float().view(height, width, 3)

        warmup_frames = min(args.warmup_frames, n_val * 2)
        print(f"  JIT warmup ({warmup_frames} frames, fused + tcnn) ...")
        with torch.no_grad():
            for j in range(warmup_frames):
                K, vm = _prepare(j % n_val)
                renderer.render(splats_dict, vm[0], K[0], width, height)
                _tcnn_forward(K, vm)
        torch.cuda.synchronize()

        # Forward-only comparison, per view (interleaved to share power state).
        fwd_fused_accum = fwd_tcnn_accum = 0.0
        fwd_count = 0
        print(f"  Timing forward fused vs tcnn "
              f"({args.num_passes} passes x {n_val} imgs) ...")
        with torch.no_grad():
            for _ in range(args.num_passes):
                for i in range(n_val):
                    K, vm = _prepare(i)
                    fwd_fused_accum += _time_once(
                        lambda: renderer.render(splats_dict, vm[0], K[0], width, height))
                    fwd_tcnn_accum += _time_once(lambda: _tcnn_forward(K, vm))
                    fwd_count += 1

        # Full training step on a single fixed view.
        K0, vm0 = _prepare(0)
        target = torch.rand(height, width, 3, device=device)
        means_t = means.detach().clone().requires_grad_(True)
        quats_t = quats.detach().clone().requires_grad_(True)
        scales_t = splats["scales"].detach().clone().requires_grad_(True)
        opac_t = splats["opacities"].detach().clone().requires_grad_(True)
        feat_t = features.detach().float().clone().requires_grad_(True)
        params_t = dm.backbone.params.detach().clone().requires_grad_(True)

        def _clear_grads():
            for t in (means_t, quats_t, scales_t, opac_t, feat_t, params_t):
                t.grad = None

        def step_fwd_bwd_tcnn():
            _clear_grads()
            rc, _, _ = rasterization(
                means=means_t, quats=F.normalize(quats_t, dim=-1),
                scales=torch.exp(scales_t), opacities=torch.sigmoid(opac_t),
                colors=feat_t.half(), viewmats=vm0, Ks=K0,
                width=width, height=height, sh_degree=None,
                near_plane=0.01, far_plane=1e10, packed=False,
                with_ut=True, with_eval3d=True,
                nht_params=NHTParams(
                    center_ray_mode=dm.center_ray_encoding,
                    ray_dir_scale=dm.ray_dir_scale,
                ),
                render_mode="RGB",
            )
            feat_ray = rc[0].reshape(-1, rc.shape[-1])
            out = dm._run_backbone(feat_ray.half())
            if not dm.tcnn_emitted_sigmoid_outputs:
                out = torch.sigmoid(out.float())
            rgb = out[:, :3].float().view(height, width, 3)
            (rgb - target).abs().mean().backward()

        def step_fwd_bwd_fused():
            _clear_grads()
            rgb, _ = nht_fused_render(
                means=means_t, quats=F.normalize(quats_t, dim=-1),
                scales=torch.exp(scales_t), features=feat_t,
                opacities=torch.sigmoid(opac_t), mlp_params=params_t,
                viewmat=vm0[0], K=K0[0], width=width, height=height,
                tile_size=16, ray_dir_scale=dm.ray_dir_scale,
                center_ray_mode=dm.center_ray_encoding,
                mlp_hidden_dim=dm.mlp_hidden_dim, mlp_num_layers=dm.mlp_num_layers,
            )
            (rgb - target).abs().mean().backward()

        bwd_iters = args.num_passes * max(1, args.training_iters)
        print(f"  Timing fwd+bwd fused ({bwd_iters} iters) ...")
        fb_fused = _cuda_time(step_fwd_bwd_fused, args.warmup_frames, bwd_iters)
        print(f"  Timing fwd+bwd tcnn ({bwd_iters} iters) ...")
        fb_tcnn = _cuda_time(step_fwd_bwd_tcnn, args.warmup_frames, bwd_iters)

        ff = fwd_fused_accum / fwd_count if fwd_count else 0
        ft = fwd_tcnn_accum / fwd_count if fwd_count else 0
        result = OrderedDict([
            ("scene", args.scene_name or os.path.basename(args.data_dir)),
            ("num_gs", num_gs),
            ("width", width),
            ("height", height),
            ("num_images", fwd_count),
            ("fwd_fused_ms", ff),
            ("fwd_tcnn_ms", ft),
            ("fwd_bwd_fused_ms", fb_fused),
            ("fwd_bwd_tcnn_ms", fb_tcnn),
            ("fwd_speedup", ft / ff if ff > 0 else 0),
            ("fwd_bwd_speedup", fb_tcnn / fb_fused if fb_fused > 0 else 0),
            ("fwd_fps_fused", 1000.0 / ff if ff > 0 else 0),
            ("fwd_fps_tcnn", 1000.0 / ft if ft > 0 else 0),
            ("fwd_bwd_fps_fused", 1000.0 / fb_fused if fb_fused > 0 else 0),
            ("fwd_bwd_fps_tcnn", 1000.0 / fb_tcnn if fb_tcnn > 0 else 0),
            ("mode", "train"),
        ])

        print_scene_results(result["scene"], result)
        if args.save_json:
            save_timing_json(args.save_json, result)
            print(f"    Saved: {args.save_json}")
        return result

    # ------------------------------------------------------------------
    #  Fused inference timing (default): one fully-fused rasterize+MLP
    #  kernel per frame via NHTInferenceRenderer. Falls back to the
    #  unfused component breakdown below when --unfused is passed or the
    #  checkpoint's shader config has no compiled fused instantiation.
    # ------------------------------------------------------------------
    use_fused = not args.unfused
    if use_fused:
        from gsplat.nht import nht_fused_supported

        supported, reason = nht_fused_supported(dm, for_training=False)
        if not supported:
            print(f"  Fused kernel unavailable ({reason}); using unfused breakdown.")
            use_fused = False

    if use_fused:
        from gsplat.nht._inference_renderer import (
            NHTInferenceConfig,
            NHTInferenceRenderer,
        )

        renderer = NHTInferenceRenderer(
            dm,
            NHTInferenceConfig(tile_size=16, center_ray_mode=dm.center_ray_encoding),
        )
        splats_dict = dict(splats)

        warmup_frames = min(args.warmup_frames, n_val * 2)
        print(f"  JIT warmup ({warmup_frames} frames, fused) ...")
        with torch.no_grad():
            for j in range(warmup_frames):
                K, vm = _prepare(j % n_val)
                renderer.render(splats_dict, vm[0], K[0], width, height)
        torch.cuda.synchronize()

        # End-to-end fused timing (projection + binning + fused kernel).
        # Views are scanned sequentially so the renderer's per-view
        # projection cache never short-circuits a timed frame.
        total_accum = 0.0
        total_count = 0
        print(f"  Timing fused end-to-end ({args.num_passes} passes x {n_val} imgs) ...")
        with torch.no_grad():
            for _ in range(args.num_passes):
                for i in range(n_val):
                    K, vm = _prepare(i)
                    torch.cuda.synchronize()
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    renderer.render(splats_dict, vm[0], K[0], width, height)
                    e.record()
                    torch.cuda.synchronize()
                    total_accum += s.elapsed_time(e)
                    total_count += 1

        avg_total = total_accum / total_count if total_count else 0

        result = OrderedDict([
            ("scene", args.scene_name or os.path.basename(args.data_dir)),
            ("num_gs", num_gs),
            ("width", width),
            ("height", height),
            ("num_images", total_count),
            ("rasterization_ms", 0.0),
            ("deferred_mlp_ms", 0.0),
            ("total_ms", avg_total),
            ("overhead_ms", 0.0),
            ("fps", 1000.0 / avg_total if avg_total > 0 else 0),
            ("fps_raster_mlp", 1000.0 / avg_total if avg_total > 0 else 0),
            ("mode", "fused"),
        ])

        print_scene_results(result["scene"], result)

        if args.save_json:
            save_timing_json(args.save_json, result)
            print(f"    Saved: {args.save_json}")

        return result

    # ------------------------------------------------------------------
    # Unfused component breakdown (--unfused).
    # JIT warmup: run a few full frames so tcnn compiles its kernels.
    # ------------------------------------------------------------------
    warmup_frames = min(args.warmup_frames, n_val * 2)
    print(f"  JIT warmup ({warmup_frames} frames) ...")
    with torch.no_grad():
        for j in range(warmup_frames):
            K, vm = _prepare(j % n_val)
            rc, ra, info = _rasterize(K, vm)
            dm(rc)
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Phase 1: Rasterization-only timing
    # ------------------------------------------------------------------
    raster_accum = 0.0
    raster_count = 0
    print(f"  Timing rasterization ({args.num_passes} passes x {n_val} imgs) ...")
    with torch.no_grad():
        for _ in range(args.num_passes):
            for i in range(n_val):
                K, vm = _prepare(i)
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                _rasterize(K, vm)
                e.record()
                torch.cuda.synchronize()
                raster_accum += s.elapsed_time(e)
                raster_count += 1

    # ------------------------------------------------------------------
    # Phase 2: MLP-only timing
    # Pre-compute ONE rasterization output, then time MLP repeatedly.
    # The MLP kernel processes (H*W, F) — same cost regardless of which
    # camera produced the features.
    # ------------------------------------------------------------------
    with torch.no_grad():
        K0, vm0 = _prepare(0)
        rc_ref, _, _ = _rasterize(K0, vm0)
    torch.cuda.synchronize()

    mlp_accum = 0.0
    mlp_count = 0
    mlp_iters = args.num_passes * n_val
    print(f"  Timing MLP ({mlp_iters} iterations) ...")
    with torch.no_grad():
        for _ in range(mlp_iters):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            dm(rc_ref)
            e.record()
            torch.cuda.synchronize()
            mlp_accum += s.elapsed_time(e)
            mlp_count += 1

    # ------------------------------------------------------------------
    # Phase 3: End-to-end timing (raster + MLP back-to-back, as in
    # real rendering).
    # ------------------------------------------------------------------
    total_accum = 0.0
    total_count = 0
    print(f"  Timing end-to-end ({args.num_passes} passes x {n_val} imgs) ...")
    with torch.no_grad():
        for _ in range(args.num_passes):
            for i in range(n_val):
                K, vm = _prepare(i)
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                rc, ra, info = _rasterize(K, vm)
                dm(rc)
                e.record()
                torch.cuda.synchronize()
                total_accum += s.elapsed_time(e)
                total_count += 1

    avg_raster = raster_accum / raster_count if raster_count else 0
    avg_mlp = mlp_accum / mlp_count if mlp_count else 0
    avg_total = total_accum / total_count if total_count else 0
    num_timed = total_count

    result = OrderedDict([
        ("scene", args.scene_name or os.path.basename(args.data_dir)),
        ("num_gs", num_gs),
        ("width", width),
        ("height", height),
        ("num_images", num_timed),
        ("rasterization_ms", avg_raster),
        ("deferred_mlp_ms", avg_mlp),
        ("total_ms", avg_total),
        ("overhead_ms", avg_total - avg_raster - avg_mlp),
        ("fps", 1000.0 / avg_total if avg_total > 0 else 0),
        ("fps_raster_mlp", 1000.0 / (avg_raster + avg_mlp)
         if (avg_raster + avg_mlp) > 0 else 0),
        ("mode", "unfused"),
    ])

    print_scene_results(result["scene"], result)

    if args.save_json:
        save_timing_json(args.save_json, result)
        print(f"    Saved: {args.save_json}")

    return result


# ---------------------------------------------------------------------------
#  Batch mode (parent process)
# ---------------------------------------------------------------------------

def _discover_scenes(args):
    if args.scenes:
        return [s.strip() for s in args.scenes.split(",") if s.strip()]
    return sorted([
        d for d in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, d))
    ])


def _resolve_scene_dir(args, scene: str) -> str:
    """Resolve the data directory for a scene, trying dataset-specific subdirs."""
    if args.scene_dir:
        direct = os.path.join(args.scene_dir, scene)
        if os.path.isdir(direct):
            return direct
        for sub in ["mipnerf360", "tandt_db/tandt", "tandt_db/db",
                     "tandt", "db", "TanksAndTemples", "DeepBlending"]:
            candidate = os.path.join(args.scene_dir, sub, scene)
            if os.path.isdir(candidate):
                return candidate
        return direct
    return args.data_dir


def _run_batch(args):
    scene_list = _discover_scenes(args)
    if not scene_list:
        print(f"No scene directories found in {args.results_dir}")
        return

    if args.collect_only:
        _collect_only(args, scene_list)
        return

    print(f"{'='*80}")
    print(f"  NHT Timing Benchmark -- {len(scene_list)} scenes (subprocess isolation)")
    print(f"  {args.num_passes} timed passes, {args.warmup_frames} warmup frames")
    print(f"  Source: {args.results_dir}")
    print(f"{'='*80}")

    script_path = os.path.abspath(__file__)
    all_results = {}
    missing = []

    for scene in scene_list:
        ckpt_dir = os.path.join(args.results_dir, scene, "ckpts")
        if not os.path.isdir(ckpt_dir):
            print(f"\n  {scene}: no ckpts/ directory -- skipping")
            missing.append(scene)
            continue
        ckpts = sorted(
            glob.glob(os.path.join(ckpt_dir, "*.pt")),
            key=lambda p: int(re.search(r"ckpt_(\d+)", os.path.basename(p)).group(1))
            if re.search(r"ckpt_(\d+)", os.path.basename(p)) else 0,
        )
        if not ckpts:
            print(f"\n  {scene}: no .pt files -- skipping")
            missing.append(scene)
            continue

        data_factor = args.data_factor if args.data_factor > 0 else get_scene_factor(scene)
        scene_data_dir = _resolve_scene_dir(args, scene)
        json_path = os.path.join(args.results_dir, scene, "stats", "timing.json")

        print(f"\n  [{scene}] Launching subprocess (factor={data_factor}) ...")

        cmd = [
            sys.executable, script_path,
            "--ckpt", ckpts[-1],
            "--data_dir", scene_data_dir,
            "--data_factor", str(data_factor),
            "--num_passes", str(args.num_passes),
            "--warmup_frames", str(args.warmup_frames),
            "--scene_name", scene,
            "--save_json", json_path,
        ]
        if args.unfused:
            cmd.append("--unfused")
        if args.train:
            cmd += ["--train", "--training_iters", str(args.training_iters)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        jit_cwd = _subprocess_cwd_for_gsplat_jit()
        proc = subprocess.run(cmd, env=env, cwd=jit_cwd)
        if proc.returncode != 0:
            print(f"    ERROR: subprocess exited with code {proc.returncode}")
            missing.append(scene)
            continue

        saved = load_timing_json(args.results_dir, scene)
        if saved is None:
            print(f"    ERROR: timing.json not written")
            missing.append(scene)
            continue
        all_results[scene] = saved

    if all_results:
        _aggregate_and_report(args, all_results, scene_list)

    if missing:
        print(f"  MISSING scenes: {', '.join(missing)}")
        print(f"  Re-run with: --scenes {','.join(missing)}")
        print()


def _aggregate_and_report(args, all_results, scene_list):
    """Pick the inference or fused-vs-tcnn aggregation based on result shape."""
    is_train = any(v.get("mode") == "train" or "fwd_bwd_fused_ms" in v
                   for v in all_results.values())
    if is_train:
        rows = aggregate_training(all_results, scene_list)
        print_training_table(rows)
    else:
        rows = aggregate_timing(all_results, scene_list)
        print_aggregation_table(rows)
    save_summary_json(args.results_dir, all_results, rows)


def _collect_only(args, scene_list):
    print(f"{'='*80}")
    print(f"  NHT Timing -- Collecting {len(scene_list)} scenes")
    print(f"  Source: {args.results_dir}")
    print(f"{'='*80}")

    all_results = {}
    missing = []
    for scene in scene_list:
        saved = load_timing_json(args.results_dir, scene)
        if saved is None:
            print(f"\n  {scene}: no timing.json -- skipping")
            missing.append(scene)
            continue
        all_results[scene] = saved
        print_scene_results(scene, saved)

    if all_results:
        _aggregate_and_report(args, all_results, scene_list)
    if missing:
        print(f"  MISSING scenes: {', '.join(missing)}")
        print()


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="NHT Rendering Benchmark")

    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--data_factor", type=int, default=0,
                    help="Data downsample factor (0 = auto-detect per scene)")
    p.add_argument("--scene_name", type=str, default="")
    p.add_argument("--save_json", type=str, default="")

    p.add_argument("--results_dir", type=str, default="")
    p.add_argument("--scene_dir", type=str, default="")
    p.add_argument("--scenes", type=str, default="")
    p.add_argument("--collect_only", action="store_true")
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--num_passes", type=int, default=3)
    p.add_argument("--warmup_frames", type=int, default=10,
                    help="Frames of full pipeline to run for JIT warmup (default 10)")
    p.add_argument("--unfused", action="store_true",
                    help="Time the legacy two-stage path (rasterization + tcnn "
                         "MLP) with a per-component breakdown instead of the "
                         "default fused end-to-end inference timing.")
    p.add_argument("--train", action="store_true",
                    help="Benchmark the fused fwd+bwd training step vs the "
                         "rasterizer/tcnn autograd path (and forward-only fused "
                         "vs tcnn), reporting per-path timings and speedups "
                         "instead of the inference-only breakdown.")
    p.add_argument("--training_iters", type=int, default=20,
                    help="Timed iterations per fwd+bwd measurement (with --train).")

    p.add_argument("--feature_dim", type=int, default=0)
    p.add_argument("--enable_view_encoding", action="store_true", default=True)
    p.add_argument("--view_encoding_type", type=str, default="sh")
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--sh_scale", type=float, default=3.0)
    p.add_argument("--center_ray_encoding", action="store_true", default=True)

    args = p.parse_args()

    if args.results_dir:
        _run_batch(args)
    elif args.ckpt:
        _run_scene_benchmark(args)
    else:
        print("Provide --ckpt (single scene) or --results_dir (batch mode)")


if __name__ == "__main__":
    main()
