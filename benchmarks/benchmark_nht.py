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

Measures per-component execution time of the NHT rendering pipeline
using CUDA events for accurate GPU timing.

Each scene is benchmarked in an isolated subprocess.  Rasterization and
MLP are timed in SEPARATE passes to avoid GPU power-state coupling
(a heavy rasterization kernel draws enough power to throttle GPU clocks,
making the immediately-following MLP appear slower than it really is).

Usage (single scene):
    python benchmark_nht.py --ckpt results/garden/ckpts/ckpt_29999_rank0.pt \
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
    total = d.get("total_ms", 0) or d.get("total", 0)
    raster = d.get("rasterization_ms", 0) or d.get("rasterization", 0)
    mlp = d.get("deferred_mlp_ms", 0) or d.get("deferred_mlp", 0)
    overhead = total - raster - mlp
    num_gs = d.get("num_gs", 0)
    w, h = d.get("width", 0), d.get("height", 0)
    n_img = d.get("num_images", 0)

    print(f"\n  {name} ({num_gs} GS, {w}x{h}, {n_img} imgs)")
    for label, val in [("rasterization", raster), ("deferred_mlp", mlp)]:
        pct = val / total * 100 if total > 0 else 0
        print(f"    {label:23s}: {val:8.2f} ms  ({pct:5.1f}%)")
    if abs(overhead) > 0.01:
        print(f"    {'overhead':23s}: {overhead:8.2f} ms  ({overhead/total*100:5.1f}%)")
    fps = 1000.0 / total if total > 0 else 0
    print(f"    {'total':23s}: {total:8.2f} ms  ({fps:.1f} FPS)")


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
            rasterize_mode="antialiased", render_mode="RGB",
            camera_model="pinhole",
            with_ut=True, with_eval3d=True, nht=True,
            center_ray_mode=dm.center_ray_encoding,
            ray_dir_scale=dm.ray_dir_scale,
        )

    # ------------------------------------------------------------------
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
        rows = aggregate_timing(all_results, scene_list)
        print_aggregation_table(rows)
        save_summary_json(args.results_dir, all_results, rows)

    if missing:
        print(f"  MISSING scenes: {', '.join(missing)}")
        print(f"  Re-run with: --scenes {','.join(missing)}")
        print()


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
        rows = aggregate_timing(all_results, scene_list)
        print_aggregation_table(rows)
        save_summary_json(args.results_dir, all_results, rows)
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
