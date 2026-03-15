# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 19:05:32 2026

@author: eduar
"""

from typing import List, Optional
import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon
from skimage.morphology import binary_closing, disk

def euclidean_length(points: List[List[float]]) -> float:
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=float)
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


def pair_distance(points: List[List[float]]) -> float:
    if len(points) != 2:
        return 0.0
    a = np.asarray(points[0], dtype=float)
    b = np.asarray(points[1], dtype=float)
    return float(np.linalg.norm(a - b))

def resample_centerline(C0: np.ndarray, step: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    C0 = np.asarray(C0, dtype=float)
    if C0.shape[0] < 2:
        return C0.copy(), np.linspace(0.0, 1.0, C0.shape[0], dtype=float)

    d = np.hypot(np.diff(C0[:, 0]), np.diff(C0[:, 1]))
    L = np.concatenate([[0.0], np.cumsum(d)])
    if L[-1] < np.finfo(float).eps:
        return C0.copy(), np.linspace(0.0, 1.0, C0.shape[0], dtype=float)

    tq = np.arange(0.0, L[-1], step, dtype=float)
    if tq.size == 0 or tq[-1] < L[-1]:
        tq = np.concatenate([tq, [L[-1]]])

    xq = np.interp(tq, L, C0[:, 0])
    yq = np.interp(tq, L, C0[:, 1])
    C = np.column_stack([xq, yq])
    s = tq / L[-1]

    if C.shape[0] > 1:
        keep = np.ones(C.shape[0], dtype=bool)
        keep[1:] = np.any(np.diff(C, axis=0) != 0, axis=1)
        C = C[keep]
        s = s[keep]

    return C, s


def default_width_profile_rel(n: int) -> np.ndarray:
    full = np.array([
        0.18, 0.4, 0.55, 0.68, 0.77, 0.83, 0.89, 0.92, 0.95, 0.97, 0.98, 1.0,
        1.0, 0.98, 0.96, 0.93, 0.88, 0.81, 0.72, 0.61, 0.51, 0.4, 0.31, 0.17,
    ], dtype=float)
    x = np.linspace(0.0, 1.0, full.size)
    xq = np.linspace(0.0, 1.0, n)
    w_rel = np.interp(xq, x, full)
    if w_rel.size:
        w_rel = w_rel / np.max(w_rel)
    return w_rel


def symmetric_width_profile_rel(n: int) -> np.ndarray:
    side = np.array([0.175, 0.355, 0.515, 0.645, 0.745, 0.82, 0.885, 0.925, 0.955, 0.975, 0.985, 1.0], dtype=float)
    left = np.interp(np.linspace(0.0, 1.0, n // 2 + 1), np.linspace(0.0, 1.0, side.size), side)
    right = np.interp(np.linspace(0.0, 1.0, n - (n // 2)), np.linspace(0.0, 1.0, side.size), side[::-1])
    idx_peak = n // 2
    w_rel = np.zeros(n, dtype=float)
    w_rel[:idx_peak + 1] = left
    w_rel[idx_peak:] = right
    w_rel[idx_peak] = 1.0
    return w_rel


def centerline_to_polygon(C: np.ndarray, w_px: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    w_px = np.asarray(w_px, dtype=float).ravel()
    M = C.shape[0]
    if M < 2:
        return C.copy()
    if w_px.size != M:
        w_px = np.full(M, float(w_px[0]), dtype=float)

    T = np.zeros_like(C, dtype=float)
    if M >= 3:
        T[1:M - 1, :] = C[2:M, :] - C[:M - 2, :]
    T[0, :] = C[1, :] - C[0, :]
    T[-1, :] = C[-1, :] - C[-2, :]
    nrm = np.hypot(T[:, 0], T[:, 1]) + np.finfo(float).eps
    T = T / nrm[:, None]
    N = np.column_stack([-T[:, 1], T[:, 0]])
    halfw = 0.5 * w_px
    left = C + N * halfw[:, None]
    right = C - N * halfw[:, None]
    return np.vstack([left, np.flipud(right)])


def polygon_to_mask(poly: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if poly.shape[0] < 3:
        return np.zeros((h, w), dtype=bool)
    rr, cc = polygon(poly[:, 1], poly[:, 0], shape=(h, w))
    out = np.zeros((h, w), dtype=bool)
    out[rr, cc] = True
    return out


def postprocess_mask(mask: np.ndarray, close_radius: int = 2) -> np.ndarray:
    mask = mask.astype(bool)
    if close_radius > 0:
        mask = binary_closing(mask, footprint=disk(close_radius))
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(bool)


def generate_preview_mask(annotation: dict, image_shape: tuple[int, int], width_multiplier: Optional[float] = None) -> np.ndarray:
    centerline = np.asarray(annotation.get("centerline_points", []), dtype=float)
    width_points = np.asarray(annotation.get("width_points", []), dtype=float)
    head_point = annotation.get("head_point", None)
    if centerline.ndim != 2 or centerline.shape[0] < 2 or centerline.shape[1] != 2:
        return np.zeros(image_shape, dtype=bool)
    if width_points.ndim != 2 or width_points.shape[0] < 2 or width_points.shape[1] != 2:
        return np.zeros(image_shape, dtype=bool)

    C, _ = resample_centerline(centerline, step=1.5)
    if C.shape[0] < 2:
        return np.zeros(image_shape, dtype=bool)

    max_width = pair_distance(width_points[:2].tolist())
    if width_multiplier is None:
        width_multiplier = float(annotation.get("width_multiplier", 1.0))
    if head_point is not None:
        hp = np.asarray(head_point, dtype=float)
        d0 = np.linalg.norm(hp - C[0])
        d1 = np.linalg.norm(hp - C[-1])
        if d1 < d0:
            C = np.flipud(C)
        w_rel = default_width_profile_rel(C.shape[0])
    else:
        w_rel = symmetric_width_profile_rel(C.shape[0])

    w_px = np.maximum(1.0, max_width * width_multiplier * w_rel)
    poly = centerline_to_polygon(C, w_px)
    mask = polygon_to_mask(poly, image_shape)
    mask = postprocess_mask(mask, close_radius=2)
    return mask