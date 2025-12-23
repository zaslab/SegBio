# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 16:20:46 2025

@author: eduar
"""

# postproc.py
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from segmentor_utils import keep_big_components, _endpoints
from skimage.segmentation import find_boundaries,watershed,relabel_sequential
from skimage.morphology import remove_small_objects, binary_erosion, thin, medial_axis
from skimage.measure import regionprops, label

# --- add near the top of postproc.py (after imports) -------------------------
# --- Size filtering utilities -----------------------------------------------


def _get_skel_length(p):
    skel, dist = medial_axis(p.image, return_distance=True)
    length_px = np.count_nonzero(skel)
    return length_px

def _get_skel_width(p):
    
    skel, dist = medial_axis(p.image, return_distance=True)
    local_width = 2 * dist[skel]
    max_width = local_width.max()           # in pixels
    return(max_width)
  
def _get_labels_from_logits(fg_prob=None,
                            bnd=None,                # edited boundary (bool/0-1)
                            seed=None,
                            bnd_history=None,
                            fg_thr=0.1,
                            outer_outline=False,
                            initial=True):
    
    p = fg_prob
    
        
    # 2. “Core” mask – very strict threshold
    core = seed>0.2 #p_work > 0.9  # tune
    core = remove_small_objects(core,  min_size=200)

    # 3. Seeds = connected components of cores
    markers = label(core)

    # 4. Foreground mask (looser)
    fg = p > fg_thr
    
    # 5. Watershed on *inverted* original probs, constrained to fg
    elevation = -p       # or -gaussian(p,1)
    labels = watershed(elevation, markers=markers, mask=fg)
        
    if initial == True:
        bnd = np.zeros(labels.shape,dtype=bool)
        for i in range(1, np.max(labels)+1): 
            mask = labels.copy()
            mask[mask!=i]=0
            mask = mask.astype(bool)
            eroded = binary_erosion(mask, footprint=np.ones((3,3), dtype=bool))
            bnd = bnd | (mask & ~eroded)
        
            labels[ mask & ~eroded] = 0
            
    if bnd is None:
        bnd = np.zeros(p.shape,dtype=bool)
    
    return labels, bnd
    
def _sizes_from_labels(labels: np.ndarray) -> tuple[list[float], list[float]]:
    """
    Return (lengths, widths) for all nonzero labels.
    The i-th entry corresponds to the i-th region in ascending label order.
    """
    props = [p for p in regionprops(labels.astype(np.int32)) if p.area > 0]
    props.sort(key=lambda p: p.label)  # ensure deterministic label order

    lengths = [_get_skel_length(p) for p in props]
    widths  = [_get_skel_width(p)  for p in props]
    return lengths, widths

def calculate_sizes_in_image(labels):
    area = list()
    for i in range(1, np.max(labels)+1):
        worm = np.where(labels==i)
        x_idx = worm[0]
        y_idx = worm[1]
        area.append(len(x_idx))
    
    lengths, widths = _sizes_from_labels(labels)
    ratios = [x / y for x, y in zip(lengths, widths)]
    
    df = pd.DataFrame({
        "Area": area,
        "Lengths": lengths,
        "Widths": widths,
        "Ratios": ratios})
    return df

def filter_labels_by_axes(labels: np.ndarray,
                          *,
                          min_length=None, max_length=None,
                          min_width=None,  max_width=None,
                          min_ratio=None,
                          remove_touching_border: bool = False):
    """
    Threshold by skeleton-based length/width from _sizes_from_labels.
    Returns (labels_filtered, removed_ids).
    """
    if labels.size == 0:
        return labels, []

    H, W = labels.shape
    props_all = regionprops(labels.astype(np.int32))

    # work only with nonzero, positive-area regions, sorted by label
    props = [p for p in props_all if p.area > 0]
    props.sort(key=lambda p: p.label)

    # get sizes in the same order, then map by label
    lengths, widths = _sizes_from_labels(labels)
    sizes_by_label = {
        p.label: (float(L), float(Wd))
        for p, L, Wd in zip(props, lengths, widths)
    }

    remove_ids = []
    for p in props:
        L, Wd = sizes_by_label[p.label]
        R = (L / Wd) if Wd > 0 else float("inf")

        bad = False
        if (min_length is not None) and (L < min_length): bad = True
        if (max_length is not None) and (L > max_length): bad = True
        if (min_width  is not None) and (Wd < min_width):  bad = True
        if (max_width  is not None) and (Wd > max_width):  bad = True
        if (min_ratio  is not None) and (R < min_ratio):   bad = True

        if remove_touching_border:
            r0, c0, r1, c1 = p.bbox
            if (r0 == 0) or (c0 == 0) or (r1 == H) or (c1 == W):
                bad = True

        if bad:
            remove_ids.append(p.label)

    if remove_ids:
        m = np.isin(labels, remove_ids)
        labels = labels.copy()
        labels[m] = 0
        labels, _, _ = relabel_sequential(labels)  # pack IDs to 1..N

    return labels.astype(np.int32), remove_ids

def _instances_from_maps(fg_bool, bnd_bool, seed_bool, *,
                         min_size=32, compactness=0.001):
    """
    fg_bool, bnd_bool, seed_bool: HxW boolean arrays (after any thresholds/edits)
    Returns int32 labels.
    """
    


    # outer rim discourages merges at object borders
    rim = find_boundaries(fg_bool.astype(int), mode="outer")
    mask = fg_bool & (~(bnd_bool | rim))

    # consolidate/generate markers (your existing heuristic)
    markers = consolidate_markers(seed_bool, mask)

    # distance-guided watershed
    dist = ndi.distance_transform_edt(mask)
    labels = watershed(-dist, markers=markers, mask=mask, compactness=compactness)

    # clean tiny fragments
    labels = remove_small_objects(labels, min_size=min_size)

    
    return labels.astype(np.int32)

# --- replace your split_instances with this overloaded version ---------------
def split_instances_v2(pred_logits=None, *,
                    fg_prob=None, bnd_edit=None, seed_edit=None,
                    fg_thr=0.5, bnd_thr=0.5, seed_thr=0.5,
                    min_size=32, compactness=0.001,
                    min_length=None, max_length=None,
                    min_width=None,  max_width=None,
                    min_ratio=None,
                    remove_touching_border=False):
    """
    EITHER:
      pred_logits: (3,H,W) tensor/ndarray (logits) for [fg, boundary, seed]
    OR:
      fg_prob:  HxW float in [0,1] (from UNet)
      bnd_edit: HxW bool/0-1  (edited boundary from GUI)
      seed_edit:HxW bool/0-1  (edited seeds from GUI)

    Returns labels (H,W) int32.
    """
    import numpy as np

    

    if pred_logits is not None:
        # original path: threshold UNet outputs
        if hasattr(pred_logits, "detach"):
            p = pred_logits.detach().cpu().sigmoid().numpy()
        else:
            p = 1.0 / (1.0 + np.exp(-pred_logits))  # sigmoid

        fg   = p[0] > fg_thr
        seed = p[2] > seed_thr
        bnd  = p[1] > bnd_thr

    elif np.asarray(bnd_edit).dtype == np.bool_:
        fg   = (np.asarray(fg_prob) > fg_thr)
        seed = seed_edit
        bnd = bnd_edit
        
    else:
        # edited-maps path from GUI
        assert fg_prob is not None and bnd_edit is not None and seed_edit is not None, \
            "Provide fg_prob, bnd_edit, seed_edit when pred_logits=None"
        fg   = (np.asarray(fg_prob) > fg_thr)
        seed = (np.asarray(seed_edit) > 0)
        bnd  = (np.asarray(bnd_edit) > 0)

    fg   = keep_big_components(fg,   min_area=400, connectivity=2)

    # NOTE: rim is added inside _instances_from_maps, but we keep your filtering here:
    seed = keep_big_components(seed, min_area=200, connectivity=2)

    bnd  = keep_big_components(bnd,  min_area=600, connectivity=2)
    bnd  = bnd & (~seed)              # avoid blocking seeds
    bnd  = thin(bnd, max_num_iter=20) # sharpen strokes

    labels = _instances_from_maps(fg, bnd, seed,
                                min_size=min_size, compactness=compactness)
    
    labels, removed = filter_labels_by_axes(
           labels,
           min_length=min_length, max_length=max_length,
           min_width=min_width,   max_width=max_width,
           min_ratio=min_ratio,
           remove_touching_border=remove_touching_border
           )
    return labels



def consolidate_markers(seed_bool, mask_bool):
    """
    seed_bool: predicted seed map thresholded (H,W) bool
    mask_bool: allowed region (fg & ~boundary) (H,W) bool
    returns: markers (H,W) int, one marker per connected component of mask
             (if a comp has seed pixels, pick the best one; otherwise synthesize one)
    """
    mask = mask_bool.astype(bool)
    mask = binary_erosion(mask, footprint=np.ones((3,3), dtype=bool))
    mask = keep_big_components(mask, min_area=500, connectivity=2)
    comp = label(mask, connectivity=2)     
    dist = ndi.distance_transform_edt(mask)     # far-from-edge score

    markers = np.zeros_like(comp, dtype=np.int32)
    k = 0
    for r in regionprops(comp):
        coords = r.coords
        # among this component's pixels, prefer those that are seeds; else all pixels
        yy, xx = coords[:,0], coords[:,1]
        in_seed = seed_bool[yy, xx]
        if in_seed.any():
            yy, xx = yy[in_seed], xx[in_seed]
        # pick the pixel with max distance (best-centered)
        idx = np.argmax(dist[yy, xx])
        y0, x0 = yy[idx], xx[idx]
        k += 1
        markers[y0, x0] = k
    return markers
