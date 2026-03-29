# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:29:37 2026

@author: eduar
"""

import argparse
import json
import shutil
import uuid
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np
from scipy.io import loadmat

try:
    import tifffile
except Exception:
    tifffile = None

try:
    import imageio.v3 as iio
except Exception:
    iio = None


def first_mat_key(mat_dict: dict) -> str:
    return next(k for k in mat_dict.keys() if not k.startswith("__"))


def load_mat_array(path: Path) -> np.ndarray:
    mat = loadmat(path, simplify_cells=True)
    arr = np.asarray(mat[first_mat_key(mat)])
    return arr


def ensure_2d_grayscale(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            return arr[..., :3].mean(axis=-1)
        return arr[..., 0]
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def save_image_file(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext in [".tif", ".tiff"]:
        if tifffile is None:
            raise RuntimeError("tifffile is required to save TIFF files")
        tifffile.imwrite(str(path), image)
        return

    if iio is None:
        raise RuntimeError("imageio is required to save non-TIFF image files")

    out = np.asarray(image)
    if out.dtype != np.uint8:
        out = out.astype(np.float32)
        mn = float(np.min(out))
        mx = float(np.max(out))
        if mx > mn:
            out = (255.0 * (out - mn) / (mx - mn)).clip(0, 255)
        else:
            out = np.zeros_like(out)
        out = out.astype(np.uint8)

    iio.imwrite(str(path), out)


def maybe_load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = load_mat_array(path)
    return np.asarray(arr)


def convert_one_legacy_folder(
    legacy_dir: Path,
    output_root: Path,
    *,
    image_ext: str = ".tif",
    write_stub_annotation_json: bool = True,
) -> Path:
    in_path = legacy_dir / "in.mat"
    out_path = legacy_dir / "out.mat"

    if not in_path.exists() or not out_path.exists():
        raise FileNotFoundError(f"{legacy_dir} must contain in.mat and out.mat")

    heads_path = legacy_dir / "heads_mask.mat"
    tails_path = legacy_dir / "tails_mask.mat"

    raw = ensure_2d_grayscale(load_mat_array(in_path))
    instance_labels = np.asarray(load_mat_array(out_path))

    if instance_labels.ndim != 2:
        raise ValueError(f"Expected 2D out.mat in {legacy_dir}, got {instance_labels.shape}")

    heads = maybe_load_mask(heads_path)
    tails = maybe_load_mask(tails_path)

    if heads is None:
        heads = np.zeros_like(instance_labels, dtype=np.uint8)
    if tails is None:
        tails = np.zeros_like(instance_labels, dtype=np.uint8)

    # normalize dtypes
    instance_labels = instance_labels.astype(np.uint16)
    worm_mask = (instance_labels > 0).astype(np.uint8)
    heads = (np.asarray(heads) > 0).astype(np.uint8)
    tails = (np.asarray(tails) > 0).astype(np.uint8)

    sample_uuid = str(uuid.uuid4())
    sample_dir = output_root / sample_uuid
    sample_dir.mkdir(parents=True, exist_ok=False)

    image_name = f"raw{image_ext}"
    image_path = sample_dir / image_name
    save_image_file(image_path, raw)

    metadata = {
        "source_type": "legacy_matlab",
        "legacy_folder": str(legacy_dir),
        "image_path": str(image_path),
        "image_shape": [int(raw.shape[0]), int(raw.shape[1])],
        "has_heads_mask": bool(heads_path.exists()),
        "has_tails_mask": bool(tails_path.exists()),
        "n_instances": int(np.max(instance_labels)),
    }

    bundle_path = sample_dir / "training_masks.npz"
    np.savez_compressed(
        bundle_path,
        instance_labels=instance_labels,
        worm_mask=worm_mask,
        head_mask=heads,
        tail_mask=tails,
        metadata_json=np.array(json.dumps(metadata), dtype=object),
    )

    with open(sample_dir / "sample_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if write_stub_annotation_json:
        stub = {
            "image_path": str(image_path),
            "image_shape": [int(raw.shape[0]), int(raw.shape[1])],
            "annotations": [],
            "note": "Converted from legacy MATLAB masks. Original GUI geometry annotations are not available.",
        }
        with open(sample_dir / "raw_annotations.json", "w", encoding="utf-8") as f:
            json.dump(stub, f, indent=2)

    return sample_dir


def convert_legacy_dataset(
    legacy_root: Path,
    output_root: Path,
    *,
    image_ext: str = ".tif",
    write_stub_annotation_json: bool = True,
) -> list[Path]:
    legacy_root = Path(legacy_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(p for p in legacy_root.iterdir() if p.is_dir())
    if not sample_dirs:
        raise RuntimeError(f"No sample folders found under {legacy_root}")

    converted = []
    for sample_dir in sample_dirs:
        if not (sample_dir / "in.mat").exists():
            continue
        if not (sample_dir / "out.mat").exists():
            continue

        new_dir = convert_one_legacy_folder(
            sample_dir,
            output_root,
            image_ext=image_ext,
            write_stub_annotation_json=write_stub_annotation_json,
        )
        print(f"Converted {sample_dir} -> {new_dir}")
        converted.append(new_dir)

    if not converted:
        raise RuntimeError("No valid legacy sample folders with in.mat + out.mat were found")

    return converted


def build_argparser(argv: Iterable[str] | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert legacy MATLAB worm training folders into UUID-based training bundles."
    )
    p.add_argument("--legacy-root", required=True, help="Folder containing legacy sample subfolders")
    p.add_argument("--output-root", required=True, help="Folder where new UUID sample folders will be created")
    p.add_argument(
        "--image-ext",
        default=".tif",
        choices=[".tif", ".tiff", ".png"],
        help="Format for saved raw image copy",
    )
    p.add_argument(
        "--no-stub-annotation-json",
        action="store_true",
        help="Do not write an empty/stub annotation json",
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None):
    args = build_argparser(argv)
    convert_legacy_dataset(
        legacy_root=Path(args.legacy_root),
        output_root=Path(args.output_root),
        image_ext=args.image_ext,
        write_stub_annotation_json=not args.no_stub_annotation_json,
    )


if __name__ == "__main__":
    main(
        ["--legacy-root", "C:\Projects\Github\Datasets\Duplicate Data\Training Data",
          "--output-root", "C:\Projects\Github\Datasets\Duplicate Data\JSON data"
        ])