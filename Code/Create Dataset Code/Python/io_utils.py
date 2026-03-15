# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 19:11:53 2026

@author: eduar
"""
import os
import numpy as np
try:
    import tifffile
except Exception:
    tifffile = None

try:
    import imageio.v3 as iio
except Exception:
    iio = None

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

def ensure_grayscale_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            return arr[..., :3].mean(axis=-1)
        return arr[..., 0]
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def load_image_any(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext in [".tif", ".tiff"] and tifffile is not None:
        arr = tifffile.imread(path)
        return ensure_grayscale_image(arr)

    if ext == ".mat":
        if loadmat is None:
            raise ImportError("scipy is required to open .mat files")
        data = loadmat(path)
        if "in" not in data:
            raise KeyError("MAT file does not contain a variable named 'in'")
        arr = data["in"]
        arr = np.asarray(arr)
        if arr.ndim == 3:
            if arr.shape[2] >= 2:
                arr = arr[:, :, 1]
            else:
                arr = arr[:, :, 0]
        return ensure_grayscale_image(arr)

    if iio is not None:
        arr = iio.imread(path)
        return ensure_grayscale_image(arr)

    raise RuntimeError("Could not load image. Install tifffile and/or imageio.")
