# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 12:54:17 2025

@author: eduar
"""

# targets.py
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import binary_dilation, disk, skeletonize
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from segmentor_utils import keep_big_components

def make_targets(label, boundary_width=4, seed_method="skeleton"):
    """
    label: (H, W) int array. 0=background, 1..N = instance ids
    returns dict of float32 arrays in [0,1]: 'fg', 'boundary', 'seed'
    """
    label = np.asarray(label).astype(int)
    fg = (label > 0).astype(np.float32)

    # 1) Boundary: thin band along each object rim (and between touching neighbors)
    b = find_boundaries(label, mode="outer")  # ~1px
    if boundary_width and boundary_width > 1:
        b = binary_dilation(b, footprint=disk(max(1, boundary_width // 2)))
    boundary = (b & (fg > 0)).astype(np.float32)

     2) Seeds (one compact blob per instance)
    if seed_method == "skeleton":
          use a pruned skeleton as a learnable "seed" blob
        skel = make_skels(label)
         tidy/thicken a bit so the target isn’t too wispy
        
        skel = binary_dilation(skel, footprint=np.ones((2,2), dtype=bool)) 
        skel = skel & ~boundary.astype(bool)
        seed = skel.astype(np.float32)
        seed = keep_big_components(seed, min_area=40, connectivity=2)
    elif seed_method == "distance":
         Alternative: distance peaks inside each object
        rim = binary_dilation(boundary.astype(bool), structure=np.ones((3,3)))
        rim = binary_dilation(boundary.astype(bool), footprint=np.ones((3,3))) 
        dist = ndi.distance_transform_edt(fg)
        dist[rim] = 0.0
        dist /= (dist.max() + 1e-6)
       
        seed = dist.astype(np.float32)
    else:
        raise ValueError("seed_method must be 'skeleton' or 'distance'")

    return {"fg": fg, "boundary": boundary, "seed": seed}

def make_skels(label):
    skel = np.zeros(label.shape, dtype=bool)
    for i in range(1, np.max(label)+1):
        worm_skel = skeletonize(label == i)
        worm_skel = trim_skeleton_ends(worm_skel, frac=0.15)
        skel=skel | (worm_skel & ~(skel&worm_skel))
    #plt.imshow(skel)
    return skel
        


def _endpoints(skel_bool: np.ndarray) -> np.ndarray:
    """8-neighborhood endpoints (degree==1)."""
    sk = skel_bool.astype(bool)
    # neighbors count (includes center pixel)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3,3), np.uint8), mode="constant", cval=0)
    return sk & (nb == 2)  # 1 neighbor + self

def trim_skeleton_ends(skel_bool: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """
    Shorten each connected skeleton component by ~`frac` from *each* end.
    Works best for open, unbranched skeletons. Leaves loops unchanged (no endpoints).
    """
    skel = skel_bool.astype(bool).copy()
    st = ndi.generate_binary_structure(2, 2)  # 8-connectivity
    labels, n = ndi.label(skel, structure=st)
    out = np.zeros_like(skel, dtype=bool)

    for lab in range(1, n+1):
        comp = (labels == lab)
        L = int(comp.sum())             # pixel length proxy
        # closed loop? no endpoints → skip
        if not _endpoints(comp).any():
            out |= comp
            continue
        # number of pruning iterations to remove ~10% from both ends
        k = max(1, int(round(frac * L / 2.0)))
        cur = comp.copy()
        for _ in range(k):
            ends = _endpoints(cur)
            if not ends.any():
                break  # nothing left to trim (short or loop)
            cur[ends] = False
        out |= cur
    return out



