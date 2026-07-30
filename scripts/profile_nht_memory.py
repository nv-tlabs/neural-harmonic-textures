# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile GPU memory of the NHT render paths across image resolutions.

Measures peak VRAM for one full train-style iteration (render + L1/SSIM loss +
backward) or an inference forward, at a range of image scales, for:

  fused_infer   -- fused kernel, save_state=False, no grad
  fused_train   -- fused kernel fwd + fused bwd (nht_fused_render)
  unfused_train -- NHT feature rasterizer + tcnn DeferredShaderModule

tinycudann allocates from its own cuMemMap arena OUTSIDE the torch caching
allocator, so torch.cuda.max_memory_allocated() alone under-reports the
unfused path. A sampler thread polls torch.cuda.mem_get_info() during the
measured region to capture the true device-level low-water mark.

Each (case, scale) runs in a fresh subprocess: tcnn OOM aborts via C++
terminate (no Python exception), which would otherwise kill the whole sweep.

Usage:
  python scripts/profile_nht_memory.py                      # full sweep
  python scripts/profile_nht_memory.py --cases fused_train --scales 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = (
    REPO_ROOT / "results" / "nht_strawberry_1000000" / "ckpts" / "ckpt_6999_rank0.pt"
)
DEFAULT_DATA_DIR = Path(r"C:\Users\jorge\GaussianSplatting\strawberry")

CASES = ("fused_infer", "fused_train", "unfused_infer", "unfused_train")
MARKER = "MEMPROF_JSON "


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class _FreeMemSampler(threading.Thread):
    """Poll torch.cuda.mem_get_info to catch non-torch (tcnn arena) allocations."""

    def __init__(self, device: int, interval_s: float = 0.001):
        super().__init__(daemon=True)
        self.device = device
        self.interval_s = interval_s
        self.min_free = None
        self._stop_evt = threading.Event()

    def run(self):
        import torch

        while not self._stop_evt.is_set():
            free, _ = torch.cuda.mem_get_info(self.device)
            if self.min_free is None or free < self.min_free:
                self.min_free = free
            time.sleep(self.interval_s)

    def stop(self):
        self._stop_evt.set()
        self.join()


def _load_scene(ckpt_path: Path, data_dir: Path, device):
    import torch

    sys.path.insert(0, str(REPO_ROOT / "gsplat" / "examples"))
    from datasets.colmap import Dataset, Parser  # noqa: E402

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    splats = {k: v.to(device) for k, v in ck["splats"].items()}
    shader_cfg = ck["deferred_module_config"]
    mlp_params = ck["deferred_module"]["backbone.params"].to(device)

    parser = Parser(data_dir=str(data_dir), factor=1, normalize=True, test_every=8)
    data = Dataset(parser, split="train")[0]
    camtoworld = data["camtoworld"].to(device)  # [4, 4]
    K_native = data["K"].to(device)  # [3, 3]
    h_native, w_native = data["image"].shape[:2]
    viewmat = torch.linalg.inv(camtoworld)
    return splats, shader_cfg, mlp_params, viewmat, K_native, w_native, h_native


def _scaled_camera(K_native, w_native: int, h_native: int, scale: float):
    W = max(1, round(w_native / scale))
    H = max(1, round(h_native / scale))
    K = K_native.clone()
    K[0] *= W / w_native
    K[1] *= H / h_native
    return K, W, H


def _loss(rgb_bhwc, target_bhwc):
    import torch.nn.functional as F
    from fused_ssim import fused_ssim

    l1 = F.l1_loss(rgb_bhwc, target_bhwc)
    ssim = 1.0 - fused_ssim(
        rgb_bhwc.permute(0, 3, 1, 2), target_bhwc.permute(0, 3, 1, 2), padding="valid"
    )
    return 0.8 * l1 + 0.2 * ssim


def _make_unfused_module(shader_cfg, mlp_params, device):
    import torch

    from gsplat.nht.deferred_shader import DeferredShaderModule

    dm = DeferredShaderModule(**shader_cfg).to(device)
    with torch.no_grad():
        dm.backbone.params.copy_(mlp_params)
    return dm


def _run_case(case, splats, shader_cfg, mlp_params, viewmat, K, W, H, device, dm=None):
    """One measured iteration. Returns dict of case-specific stats.

    ``dm`` (the unfused DeferredShaderModule) is constructed once by the
    caller and passed in when looping multiple iterations, since
    reconstructing it per-iteration would dominate the timing for large
    ``--iters`` and doesn't reflect real training (the module persists
    across steps; only fresh leaf tensors are re-detached each iteration,
    matching how each training step re-derives activations from the
    persistent splat parameters).
    """
    import torch
    import torch.nn.functional as F

    from gsplat.nht._fused_train import nht_fused_render

    hidden = shader_cfg["mlp_hidden_dim"]
    layers = shader_cfg["mlp_num_layers"]
    ray_dir_scale = (
        shader_cfg["sh_scale"] if shader_cfg.get("view_encoding_type") == "sh" else 1.0
    )
    center_ray = bool(shader_cfg.get("center_ray_encoding", False))

    train = case.endswith("_train")
    means = splats["means"].detach().requires_grad_(train)
    quats_raw = splats["quats"].detach().requires_grad_(train)
    scales_raw = splats["scales"].detach().requires_grad_(train)
    opac_raw = splats["opacities"].detach().requires_grad_(train)
    features = splats["features"].detach().requires_grad_(train)
    params = mlp_params.detach().requires_grad_(train)

    stats = {}
    if case in ("fused_infer", "fused_train"):
        with torch.enable_grad() if train else torch.no_grad():
            rgb, alpha = nht_fused_render(
                means=means,
                quats=F.normalize(quats_raw, dim=-1),
                scales=torch.exp(scales_raw),
                features=features,
                opacities=torch.sigmoid(opac_raw),
                mlp_params=params,
                viewmat=viewmat,
                K=K,
                width=W,
                height=H,
                tile_size=16,
                ray_dir_scale=ray_dir_scale,
                center_ray_mode=center_ray,
                mlp_hidden_dim=hidden,
                mlp_num_layers=layers,
            )
            if train:
                target = torch.rand_like(rgb.unsqueeze(0))
                loss = _loss(rgb.unsqueeze(0), target)
                loss.backward()
        stats["mean_rgb"] = float(rgb.detach().mean())
    else:  # unfused_infer / unfused_train
        from gsplat.nht._rendering import NHTParams
        from gsplat.rendering import rasterization

        with torch.enable_grad() if train else torch.no_grad():
            rendered, alphas, _ = rasterization(
                means=means,
                quats=F.normalize(quats_raw, dim=-1),
                scales=torch.exp(scales_raw),
                opacities=torch.sigmoid(opac_raw),
                colors=features,
                viewmats=viewmat[None],
                Ks=K[None],
                width=W,
                height=H,
                tile_size=16,
                packed=False,
                with_ut=True,
                with_eval3d=True,
                nht_params=NHTParams(
                    center_ray_mode=center_ray, ray_dir_scale=ray_dir_scale
                ),
            )
            rgb, _ = dm(rendered)
            if train:
                target = torch.rand_like(rgb)
                loss = _loss(rgb, target)
                loss.backward()
        stats["mean_rgb"] = float(rgb.detach().mean())
    return stats


def worker(args) -> None:
    import torch

    device = torch.device("cuda:0")
    torch.cuda.init()

    splats, shader_cfg, mlp_params, viewmat, K_native, w_native, h_native = _load_scene(
        Path(args.ckpt), Path(args.data_dir), device
    )

    dm = (
        _make_unfused_module(shader_cfg, mlp_params, device)
        if args.case.startswith("unfused_")
        else None
    )

    # Warm-up at 1/8 scale: loads lazy CUDA modules, compiles tcnn JIT, and
    # primes both allocators so the measured region reflects the big batch only.
    K_s, W_s, H_s = _scaled_camera(K_native, w_native, h_native, 8.0)
    _run_case(args.case, splats, shader_cfg, mlp_params, viewmat, K_s, W_s, H_s, device, dm=dm)
    torch.cuda.synchronize()

    K, W, H = _scaled_camera(K_native, w_native, h_native, args.scale)
    free_before, total = torch.cuda.mem_get_info(0)
    torch.cuda.reset_peak_memory_stats()
    alloc_before = torch.cuda.memory_allocated()

    sampler = _FreeMemSampler(0)
    sampler.start()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        stats = _run_case(
            args.case, splats, shader_cfg, mlp_params, viewmat, K, W, H, device, dm=dm
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    sampler.stop()

    free_after, _ = torch.cuda.mem_get_info(0)
    min_free = sampler.min_free if sampler.min_free is not None else free_after
    result = {
        "case": args.case,
        "scale": args.scale,
        "width": W,
        "height": H,
        "mpix": W * H / 1e6,
        "n_splats": int(splats["means"].shape[0]),
        "iters": args.iters,
        "torch_peak_alloc_gb": (torch.cuda.max_memory_allocated() - alloc_before)
        / 2**30,
        "torch_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "device_peak_used_gb": (total - min_free) / 2**30,
        "device_used_before_gb": (total - free_before) / 2**30,
        "device_used_after_gb": (total - free_after) / 2**30,
        "total_gb": total / 2**30,
        "elapsed_s": elapsed,
        "elapsed_per_iter_s": elapsed / args.iters if args.iters else elapsed,
        **stats,
    }
    print(MARKER + json.dumps(result), flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def driver(args) -> None:
    results = []
    for case in args.cases:
        skip_larger = False
        for scale in sorted(args.scales, reverse=True):  # small images first
            if skip_larger:
                results.append({"case": case, "scale": scale, "status": "skipped"})
                continue
            cmd = [
                sys.executable,
                __file__,
                "--worker",
                "--case",
                case,
                "--scale",
                str(scale),
                "--ckpt",
                str(args.ckpt),
                "--data-dir",
                str(args.data_dir),
                "--iters",
                str(args.iters),
            ]
            print(f"[{case} @ 1/{scale:g}] running ...", flush=True)
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=900
                )
            except subprocess.TimeoutExpired:
                results.append({"case": case, "scale": scale, "status": "timeout"})
                skip_larger = True
                continue
            line = next(
                (l for l in proc.stdout.splitlines() if l.startswith(MARKER)), None
            )
            if proc.returncode == 0 and line:
                rec = json.loads(line[len(MARKER) :])
                rec["status"] = "ok"
                results.append(rec)
                per_iter_ms = rec["elapsed_per_iter_s"] * 1e3
                its = 1000.0 / per_iter_ms if per_iter_ms > 0 else 0
                print(
                    f"    {rec['width']}x{rec['height']} ({rec['mpix']:.1f} Mpix), "
                    f"{rec['iters']} iters: "
                    f"torch peak +{rec['torch_peak_alloc_gb']:.2f} GB, "
                    f"device peak {rec['device_peak_used_gb']:.2f} GB, "
                    f"{per_iter_ms:.1f} ms/it ({its:.1f} it/s)",
                    flush=True,
                )
            else:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
                results.append(
                    {
                        "case": case,
                        "scale": scale,
                        "status": f"crash rc={proc.returncode}",
                        "tail": tail,
                    }
                )
                print(
                    f"    CRASHED (rc={proc.returncode}). stderr tail:", flush=True
                )
                for t in tail:
                    print(f"      {t}", flush=True)
                skip_larger = True

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")

    # Summary table
    print(f"\n{'case':<14} {'res':>12} {'Mpix':>6} {'torch peak':>11} "
          f"{'device peak':>12} {'ms/it':>8} {'it/s':>8}  status")
    for r in results:
        if r.get("status") == "ok":
            per_iter_ms = r["elapsed_per_iter_s"] * 1e3
            its = 1000.0 / per_iter_ms if per_iter_ms > 0 else 0
            print(
                f"{r['case']:<14} {r['width']:>5}x{r['height']:<6} {r['mpix']:>6.1f} "
                f"{r['torch_peak_alloc_gb']:>9.2f}GB {r['device_peak_used_gb']:>10.2f}GB "
                f"{per_iter_ms:>7.1f} {its:>8.1f}  ok"
            )
        else:
            print(f"{r['case']:<14} {'1/' + str(r['scale']):>12} {'':>6} "
                  f"{'':>11} {'':>12} {'':>8} {'':>8}  {r['status']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--case", choices=CASES)
    ap.add_argument("--scale", type=float)
    ap.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    ap.add_argument("--scales", nargs="+", type=float, default=[4, 2, 1.5, 1])
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--output", default=str(REPO_ROOT / "results" / "memprof_nht.json"))
    ap.add_argument(
        "--iters", type=int, default=1,
        help="Repeat the measured region this many times per (case, scale) "
             "subprocess, e.g. to approximate a few thousand training steps "
             "instead of a single synthetic iteration.",
    )
    args = ap.parse_args()

    if args.worker:
        if not args.case or args.scale is None:
            ap.error("--worker requires --case and --scale")
        worker(args)
    else:
        driver(args)


if __name__ == "__main__":
    main()
