# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 19:19:05 2025

@author: eduar
"""


import torch
import numpy as np
import cv2


## Augmentations for unet and RNN

import random
import torch.nn.functional as FF
from typing import Optional, Tuple
from scipy import ndimage as ndi
# ─────────────────────────────────────────────────────────────────────
#  Generic worm augmentation
# ─────────────────────────────────────────────────────────────────────


import torchvision.transforms.functional as TF


class WormAugUniversal:
    def __init__(
        self,
        zoom_p: float = 0.5,
        zoom_range: Tuple[float, float] = (0.75, 0.90),
        rot_deg: int = 15,
        hflip: bool = True,
        vflip: bool = True,
        brightness: float = 0.2,
        contrast: float = 0.2,
        blur_p: float = 0.2,
    ):
        self.zoom_p, self.zoom_range = zoom_p, zoom_range
        self.rot_deg, self.hflip, self.vflip = rot_deg, hflip, vflip
        self.brightness, self.contrast, self.blur_p = brightness, contrast, blur_p

    def __call__(
        self,
        img:   torch.Tensor,                     # C×H×W, float in [0,1] (1/3/5 ch)
        worm:  Optional[torch.Tensor] = None,   
    ):
        # Pack tensors and remember which are label maps
        pack, is_label = [img], [False]
        worm1 = worm[:,0,:,:]
        worm2 = worm[:,1,:,:]
        worm3 = worm[:,2,:,:]
        
        
        if worm1  is not None:  pack.append(worm1);  is_label.append(True)
        if worm2  is not None:  pack.append(worm2);  is_label.append(True)
        if worm3  is not None:  pack.append(worm3);  is_label.append(True)

        # Sanity: channel-first
        for t in pack:
            assert t.ndim == 3, f"expected C×H×W, got {t.shape}"

        # -------- 1) ZOOM-OUT (shared s, py, px) -------------------
        if random.random() < self.zoom_p:
            s = random.uniform(*self.zoom_range)
            C, H, W = pack[0].shape
            newH, newW = int(H * s), int(W * s)
            py = random.randint(0, H - newH)
            px = random.randint(0, W - newW)

            new_pack = []
            for t, lbl in zip(pack, is_label):
                if lbl:
                    # interpolate as float, NEAREST; then round & cast back
                    t_float = t.float()
                    shr = FF.interpolate(t_float[None], (newH, newW), mode="nearest")[0]
                    canvas = torch.zeros_like(t_float)
                    canvas[:, py:py + newH, px:px + newW] = shr
                    t = canvas.round().to(t.dtype)
                else:
                    # image: bilinear + mean-pad
                    shr = FF.interpolate(t[None], (newH, newW),
                                        mode="bilinear", align_corners=False)[0]
                    pad_val = shr.mean(dim=(1, 2), keepdim=True)
                    canvas = pad_val.expand_as(t).clone()
                    canvas[:, py:py + newH, px:px + newW] = shr
                    t = canvas
                new_pack.append(t)
            pack = new_pack

        # -------- 2) FLIPS (shared) --------------------------------
        if self.hflip and random.random() < 0.5:
            pack = [torch.flip(t, dims=[2]) for t in pack]  # horizontal
        if self.vflip and random.random() < 0.5:
            pack = [torch.flip(t, dims=[1]) for t in pack]  # vertical

        # -------- 3) AFFINE (shared angle/scale) -------------------
        angle = random.uniform(-self.rot_deg, self.rot_deg)
        scale = random.uniform(0.95, 1.05)

        new_pack = []
        for i, (t, lbl) in enumerate(zip(pack, is_label)):
            interp = TF.InterpolationMode.NEAREST if lbl else TF.InterpolationMode.BILINEAR
            out = TF.affine(
                t, angle=angle, translate=(0, 0), scale=scale, shear=0, interpolation=interp
            )
            if lbl:
                out = out.round().to(t.dtype)
            new_pack.append(out)
        pack = new_pack

        # -------- 4) Photometric only on visual channels -----------
        img = pack[0]
        n_vis = 3 if img.shape[0] >= 3 else 1
        vis, rest = img[:n_vis], img[n_vis:]

        if self.brightness:
            vis = vis * (1 + random.uniform(-self.brightness, self.brightness))
        if self.contrast:
            mean = vis.mean(dim=(1, 2), keepdim=True)
            vis = (vis - mean) * (1 + random.uniform(-self.contrast, self.contrast)) + mean
        if random.random() < self.blur_p:
            vis = TF.gaussian_blur(vis, kernel_size=[3, 3])

        pack[0] = torch.cat([vis.clamp(0, 1), rest], dim=0)

        # -------- 5) Unpack in original order ----------------------
        it = iter(pack)
        img_out    = next(it)
        worm_out1  = next(it) if worm   is not None else None
        worm_out2  = next(it) if worm   is not None else None
        worm_out3  = next(it) if worm   is not None else None
        worm_out1 = worm_out1[:,None,...]
        worm_out2 = worm_out2[:,None,...]
        worm_out3 = worm_out3[:,None,...]
        worm_out = torch.cat((worm_out1, worm_out2,worm_out3), axis = 1)
        return img_out, worm_out


def _interp_for_scale(src_hw, dst_hw):
    sh, sw = src_hw; dh, dw = dst_hw
    return cv2.INTER_AREA if (dh < sh or dw < sw) else cv2.INTER_LINEAR

def resize_image_512(img_u8: np.ndarray) -> np.ndarray:
    ih, iw = img_u8.shape[:2]
    return cv2.resize(img_u8, (512, 512), interpolation=_interp_for_scale((ih, iw), (512, 512)))

def resize_mask_512(mask_u8: np.ndarray) -> np.ndarray:
    # preserves labels/edges
    return cv2.resize(mask_u8, (512, 512), interpolation=cv2.INTER_NEAREST)

def keep_big_components(mask, min_area=64, connectivity=2):
    """
    mask: (H,W) bool or 0/1
    connectivity: 1=4-conn, 2=8-conn (use 2 for diagonal touches)
    """
    mask = mask.astype(bool)
    struct = ndi.generate_binary_structure(rank=2, connectivity=connectivity)
    labels, _ = ndi.label(mask, structure=struct)
    # compute component sizes
    sizes = np.bincount(labels.ravel())
    keep = np.zeros_like(sizes, dtype=bool)
    keep[np.where(sizes >= min_area)] = True
    keep[0] = False  # background
    cleaned = keep[labels]  # boolean mask with small blobs removed

    return cleaned

def _endpoints(skel_bool: np.ndarray) -> np.ndarray:
    """8-neighborhood endpoints (degree==1)."""
    sk = skel_bool.astype(bool)
    # neighbors count (includes center pixel)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3,3), np.uint8), mode="constant", cval=0)
    return sk & (nb == 2)  # 1 neighbor + self