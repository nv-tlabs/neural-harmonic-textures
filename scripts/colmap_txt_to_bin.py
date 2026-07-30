#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a COLMAP text-format sparse model to COLMAP's binary format.

Reads cameras.txt / images.txt / points3D.txt from --input and writes
cameras.bin / images.bin / points3D.bin to --output, using a self-contained
reader/writer for COLMAP's stable binary reconstruction layout
(https://colmap.github.io/format.html#binary-file-format).

This exists because gsplat's COLMAP Parser (examples/datasets/colmap.py)
loads reconstructions via ``pycolmap.Reconstruction(dir)``, which prefers a
directory's binary files over its text files when both are present. Some
datasets (e.g. the "strawberry" rigid-COLMAP export used by
train_strawberry.ps1) trip a text-format parsing issue in the installed
pycolmap; converting to binary first sidesteps it, since the binary loader
path is exercised instead.

Usage:
    python colmap_txt_to_bin.py --input <dir> --output <dir> [--force]
"""

import argparse
import struct
from pathlib import Path
from typing import BinaryIO, Dict, NamedTuple

import numpy as np

# (model_id, num_params), matching COLMAP's CameraModelId enum. Stable across
# COLMAP/pycolmap versions -- see src/colmap/sensor/models.h.
CAMERA_MODELS = {
    "SIMPLE_PINHOLE": (0, 3),
    "PINHOLE": (1, 4),
    "SIMPLE_RADIAL": (2, 4),
    "RADIAL": (3, 5),
    "OPENCV": (4, 8),
    "OPENCV_FISHEYE": (5, 8),
    "FULL_OPENCV": (6, 12),
    "FOV": (7, 5),
    "SIMPLE_RADIAL_FISHEYE": (8, 4),
    "RADIAL_FISHEYE": (9, 5),
    "THIN_PRISM_FISHEYE": (10, 12),
}


class Camera(NamedTuple):
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


class Image(NamedTuple):
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3d_ids: np.ndarray


class Point3D(NamedTuple):
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2d_idxs: np.ndarray


# ---------------------------------------------------------------------------
# Text readers.
# ---------------------------------------------------------------------------


def read_cameras_text(path: Path) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            width, height = int(elems[2]), int(elems[3])
            params = np.array([float(x) for x in elems[4:]], dtype=np.float64)
            cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras


def read_images_text(path: Path) -> Dict[int, Image]:
    images: Dict[int, Image] = {}
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue

        elems = line.split()
        image_id = int(elems[0])
        qvec = np.array([float(x) for x in elems[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in elems[5:8]], dtype=np.float64)
        camera_id = int(elems[8])
        name = elems[9]

        # The 2D-point line always immediately follows the pose line, even
        # when empty (an image with zero matched keypoints).
        points_line = lines[i].strip() if i < len(lines) else ""
        i += 1
        p_elems = points_line.split()
        if p_elems:
            xs = [float(x) for x in p_elems[0::3]]
            ys = [float(x) for x in p_elems[1::3]]
            pids = [int(x) for x in p_elems[2::3]]
            xys = np.column_stack([xs, ys])
        else:
            xys = np.zeros((0, 2), dtype=np.float64)
            pids = []
        point3d_ids = np.array(pids, dtype=np.int64)

        images[image_id] = Image(
            image_id, qvec, tvec, camera_id, name, xys, point3d_ids
        )
    return images


def read_points3d_text(path: Path) -> Dict[int, Point3D]:
    points3d: Dict[int, Point3D] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            point_id = int(elems[0])
            xyz = np.array([float(x) for x in elems[1:4]], dtype=np.float64)
            rgb = np.array([int(x) for x in elems[4:7]], dtype=np.uint8)
            error = float(elems[7])
            track = elems[8:]
            image_ids = np.array([int(x) for x in track[0::2]], dtype=np.int64)
            point2d_idxs = np.array([int(x) for x in track[1::2]], dtype=np.int64)
            points3d[point_id] = Point3D(
                point_id, xyz, rgb, error, image_ids, point2d_idxs
            )
    return points3d


# ---------------------------------------------------------------------------
# Binary writers (inverse of COLMAP's read_{cameras,images,points3D}_binary).
# ---------------------------------------------------------------------------


def _write(f: BinaryIO, fmt: str, *values) -> None:
    f.write(struct.pack("<" + fmt, *values))


def write_cameras_binary(cameras: Dict[int, Camera], path: Path) -> None:
    with open(path, "wb") as f:
        _write(f, "Q", len(cameras))
        for cam in cameras.values():
            if cam.model not in CAMERA_MODELS:
                raise ValueError(f"Camera {cam.id}: unknown model {cam.model!r}")
            model_id, num_params = CAMERA_MODELS[cam.model]
            if len(cam.params) != num_params:
                raise ValueError(
                    f"Camera {cam.id} ({cam.model}) expects {num_params} params, "
                    f"got {len(cam.params)}"
                )
            _write(f, "iiQQ", cam.id, model_id, cam.width, cam.height)
            _write(f, "d" * num_params, *cam.params.tolist())


def write_images_binary(images: Dict[int, Image], path: Path) -> None:
    with open(path, "wb") as f:
        _write(f, "Q", len(images))
        for img in images.values():
            _write(f, "i", img.id)
            _write(f, "dddd", *img.qvec.tolist())
            _write(f, "ddd", *img.tvec.tolist())
            _write(f, "i", img.camera_id)
            f.write(img.name.encode("utf-8") + b"\x00")
            _write(f, "Q", len(img.point3d_ids))
            for (x, y), p3d_id in zip(img.xys, img.point3d_ids):
                _write(f, "ddq", x, y, int(p3d_id))


def write_points3d_binary(points3d: Dict[int, Point3D], path: Path) -> None:
    with open(path, "wb") as f:
        _write(f, "Q", len(points3d))
        for pt in points3d.values():
            _write(f, "Q", pt.id)
            _write(f, "ddd", *pt.xyz.tolist())
            _write(f, "BBB", *pt.rgb.tolist())
            _write(f, "d", pt.error)
            _write(f, "Q", len(pt.image_ids))
            for img_id, p2d_idx in zip(pt.image_ids, pt.point2d_idxs):
                _write(f, "ii", int(img_id), int(p2d_idx))


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

CONVERSIONS = (
    ("cameras.txt", "cameras.bin", read_cameras_text, write_cameras_binary),
    ("images.txt", "images.bin", read_images_text, write_images_binary),
    ("points3D.txt", "points3D.bin", read_points3d_text, write_points3d_binary),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True,
        help="Directory containing cameras.txt, images.txt, points3D.txt",
    )
    parser.add_argument(
        "--output", required=True,
        help="Directory to write cameras.bin, images.bin, points3D.bin (may "
             "be the same as --input)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite .bin files that already exist in --output",
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for txt_name, bin_name, reader, writer in CONVERSIONS:
        txt_path = in_dir / txt_name
        bin_path = out_dir / bin_name
        if not txt_path.exists():
            raise FileNotFoundError(f"Required COLMAP file missing: {txt_path}")
        if bin_path.exists() and not args.force:
            print(f"  {bin_path} already exists - skipping (use --force to overwrite).")
            continue
        print(f"  Converting {txt_path} -> {bin_path} ...")
        records = reader(txt_path)
        writer(records, bin_path)
        print(f"    wrote {len(records)} records")

    print("Done.")


if __name__ == "__main__":
    main()
