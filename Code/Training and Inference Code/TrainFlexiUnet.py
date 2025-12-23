# -*- coding: utf-8 -*-
from __future__ import annotations
"""wormseg.cli.train_unet_seg
================================
Train the parametrisable **3‑layer UNet** on either **head** or **tail** worm
masks stored in the legacy MATLAB dataset.  Uses the **original masked BCE +
Dice loss** (0.6 × BCE + 0.4 × Dice) and saves checkpoints every
``--save-every`` epochs (default 10).

Example (Spyder/IPython)
-----------------------
```python
from wormseg.cli.train_unet_seg import main

main([
    "--root", r"D:\\WormProject\\NNdata",
    "--epochs", "80",
    "--batch-size", "8",
    "--device", "cuda",
    "--base-filters", "32",
])
```
"""


import argparse
import math
import random

import numpy as np
import torch
import torch.nn as nn
import csv, time
import torchvision.transforms.functional as F
import torch.nn.functional as FF
from pathlib import Path
from typing import Iterable, Tuple
from targets import make_targets
from scipy.io import loadmat
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from FlexiUnet import UNet
from segmentor_utils import WormAugUniversal, resize_mask_512, resize_image_512

# ----------------------------------------------------------------------------
# Dataset --------------------------------------------------------------------
# ----------------------------------------------------------------------------
class MatSegDataset(Dataset):
    """Reads *in.mat*, *heads_mask.mat*, *tails_mask.mat* from each sample dir."""

    def __init__(self, root: str | Path, *, transform=None):  # noqa: D401
        self.root = Path(root)
        self.samples = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not self.samples:
            raise RuntimeError(f"No sample folders found under {self.root}")
        self.transform = transform  
    # helper ------------------------------------------------------------------
    @staticmethod
    def _first_key(mat: dict) -> str:
        return next(k for k in mat.keys() if not k.startswith("__"))

    def _load_sample(self, path: Path) -> Tuple[Tensor, Tensor, Tensor]:
        img_mat = loadmat(path / "in.mat", simplify_cells=True)
        
        bf = img_mat[self._first_key(img_mat)].astype(np.float32)#[..., 0]  # H×W
        out_mask = loadmat(path / "out.mat", simplify_cells=True)
        out = out_mask[self._first_key(out_mask)].astype(np.float32)
        bf  = resize_image_512(bf)    # uint8 H=W=512
        out = resize_mask_512(out)
        
        bf = bf / 255.0  # ensure 0‑1 range
        
        t = make_targets(out, boundary_width=4, seed_method="skeleton")  # or "distance"
        target = np.stack([t["fg"], t["boundary"], t["seed"]], axis=0).astype("float32")  # (3,H,W)
        target = torch.from_numpy(target) 
        
        out[out > 0] = 1.0
        bf_stacked = bf[None, ...]    
        
        if self.transform is not None:
            bf_stacked, out = self.transform(torch.from_numpy(bf_stacked),

                                                      torch.from_numpy(out[None]))
        return (
 
            bf_stacked,
            out,
            target
        )

    def __getitem__(self, idx: int):  
        return self._load_sample(self.samples[idx])

    def __len__(self) -> int:  
        return len(self.samples)



# ----------------------------------------------------------------------------
# Loss utilities -------------------------------------------------------------
# ----------------------------------------------------------------------------


def multihead_loss(logits, target, chan_w=(1,4,2), pos_w=(1,6,2)):
    
    # logits, target: (B, 3, H, W)
    B, C, H, W = logits.shape

    cw = torch.as_tensor(chan_w, device=logits.device, dtype=logits.dtype)          # (C,)
    pw = torch.as_tensor(pos_w,  device=logits.device, dtype=logits.dtype)          # (C,)
    pw = pw.view(1, C, 1, 1)  # <-- align to channel dim for NCHW inputs

    # BCE with per-channel pos weights, then per-channel task weights
    bce = FF.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    bce = (cw.view(1, C, 1, 1) * bce).mean()

    # Dice (channel-weighted), using flattened spatial dims to avoid shape gotchas
    probs = torch.sigmoid(logits)
    eps = 1e-5
    p = probs.flatten(2)   # (B, C, HW)
    t = target.flatten(2)  # (B, C, HW)
    num = 2.0 * (p * t).sum(-1)                 # (B, C)
    den = (p*p + t*t).sum(-1) + eps             # (B, C)
    dice_per_c = 1.0 - (num + eps) / (den + eps)  # (B, C)
    dice = (cw.view(1, C) * dice_per_c).mean()

    return bce + 0.5 * dice


class ClassWeightedLoss(nn.Module):
    def __init__(self,
                 chan_w : tuple = (1,4,2),
                 pos_w  : tuple = (1,6,2)):      # ←  >1 boosts the “1” class
        super().__init__()
    
        self.chan_w  = chan_w
        self.pos_w = pos_w
            # store for use in forward()

    def forward(self, prob, target):
        return multihead_loss(prob, target, chan_w=self.chan_w, pos_w=self.pos_w)

# ----------------------------------------------------------------------------
# Metrics --------------------------------------------------------------------
# ----------------------------------------------------------------------------
@torch.no_grad()
def dice_coeff(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    prob = torch.sigmoid(pred)
    pred_bin = (prob > 0.5).float()
    inter = (pred_bin * target).sum(dim=(1,2,3))
    union = (pred_bin).sum(dim=(1,2,3)) + (target).sum(dim=(1,2,3))
    
    return ((2 * inter + eps) / (union + eps)).mean()

# ----------------------------------------------------------------------------
# Training loop --------------------------------------------------------------
# ----------------------------------------------------------------------------

def validate(model, loader, loss_fn, device):
    model.eval()
    losses, dices = [], []
    for bf, mask, target in loader:
        if bf.size()[0] > 1:
            bf = bf.squeeze()
            bf = bf[:,None,...]
    
        bf, target = bf.to(device), target.to(device)
        
        pred  = (model(bf))
        loss  = loss_fn(pred, target)
        dice  = dice_coeff(pred, target)
        losses.append(loss.item());  dices.append(dice.item())
    return float(np.mean(losses)), float(np.mean(dices))

def train_one_epoch(
    model,
    loader,
    opt,
    loss_fn,
    device: torch.device,
    scaler: GradScaler | None = None,
    augs_per_sample: int = 1,
):
    model.train()
    losses, dices = [], []
    aug = WormAugUniversal()
    batch_track = 0
    for img, _mask_unused, target in loader:
        if batch_track % 2 == 0:
            print(batch_track)
        batch_track += 1
        # Ensure shapes are (B, C, H, W) for image and (B, 3, H, W) for target
        if img.ndim == 3:  # (B,H,W) -> (B,1,H,W)
            img = img.unsqueeze(1)
        assert img.ndim == 4 and target.ndim == 4, "Expected batched tensors"

        opt.zero_grad(set_to_none=True)
        micro_dices = []
        micro_losses = []

        for k in range(max(1, augs_per_sample)):
            # ---- build ONE augmented view of the whole minibatch on CPU ----
            imgs_aug, worms_aug = [], []
            B = img.shape[0]
            for b in range(B):
                img_b   = img[b]      # (1,H,W) or (C,H,W)
                worm_b  = target[b].unsqueeze(0)   # (3,H,W)
                img_a, worm_a = aug(img=img_b, worm=worm_b)   # random each pass
                imgs_aug.append(img_a.unsqueeze(0))           # (1,C,H,W)
                worms_aug.append(worm_a)         # (1,3,H,W)

            bf   = torch.cat(imgs_aug,  dim=0).to(device, non_blocking=True)  # (B,C,H,W)
            out  = torch.cat(worms_aug, dim=0).to(device, non_blocking=True)  # (B,3,H,W)

            target_t = out

            with autocast(enabled=scaler is not None):
                pred = model(bf)
                loss = loss_fn(pred, target_t)

            # ---- accumulate gradients; scale loss so total grad matches 1 full pass ----
            if scaler is None:
                (loss / max(1, augs_per_sample)).backward()
            else:
                scaler.scale(loss / max(1, augs_per_sample)).backward()

            # track metrics (report unscaled loss; dice of this microbatch)
            with torch.no_grad():
                micro_losses.append(loss.detach().item())
                micro_dices.append(dice_coeff(pred, target_t).detach().item())

            # Free asap
            del bf, out, target_t, pred
            if batch_track % 3 == 0:
                torch.cuda.empty_cache() if device.type == "cuda" else None

        # ---- single optimizer step after K micro-passes ----
        if scaler is None:
            opt.step()
        else:
            scaler.step(opt)
            scaler.update()
        

        # Log mean over the K micro-passes
        losses.append(float(np.mean(micro_losses)))
        dices.append(float(np.mean(micro_dices)))

    return float(np.mean(losses)), float(np.mean(dices))


# ----------------------------------------------------------------------------
# CLI ------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(description="Train 3‑layer UNet on worm heads or tails.")
    p.add_argument("--root", required=True, default=r"/sci/labs/zaslab/eduard.bokman/WormSegmentor/midline_ignore_heads/TrainData", help="Folder containing NNdata/* sample dirs")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--base-filters", type=int, default=32,
                   help="Number of filters in the first UNet layer")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="checkpoints_unet",
                   help="Directory to save .pth checkpoints")
    p.add_argument("--save-every", type=int, default=10,
                   help="Save a checkpoint every N epochs (0 to disable)")
    p.add_argument("--amp", action="store_true", help="Enable mixed‑precision training (CUDA only)")
    p.add_argument("--val-split", type=float, default=0.15,
                   help="Fraction of samples reserved for validation")
    p.add_argument("--val-every", type=int, default=5,
                   help="Run validation every N epochs (0 = never)")
    p.add_argument("--augs-per-sample", type=int, default=1,
                   help="How many random augmentations to generate per image per epoch (stacked in-batch).")
    # CLI
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile to JIT-compile the model graph.")
    p.add_argument("--compile-mode", default="max-autotune",
               choices=["default", "reduce-overhead", "max-autotune"],
                   help="Compilation optimization mode.")
    p.add_argument("--dynamic", action="store_true",
                   help="Allow dynamic shapes (slower compile, fewer recompiles).")

    
    return p.parse_args(argv)

# ----------------------------------------------------------------------------
# Main -----------------------------------------------------------------------
# ----------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None):
    import torch, torch.backends.cudnn as cudnn
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")  
    cudnn.benchmark = True
    args = parse_args(argv)
    print(f'Batch Size: {args.batch_size} lr: {args.lr} Filters: {args.base_filters}') 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    ds  = MatSegDataset(args.root, transform=None)
    n_val = int(len(ds) * args.val_split)
    n_trn = len(ds) - n_val
    trn_ds, val_ds = torch.utils.data.random_split(ds, [n_trn, n_val],
                                                  generator=torch.Generator().manual_seed(args.seed))

    loader_trn = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    loader_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)    

    model = UNet(depth=args.layers, base_filters=args.base_filters).to(device)
    if args.compile:
        model = torch.compile(
            model,
            backend="aot_eager",                 
            mode=getattr(args, "compile_mode", "reduce-overhead"),
            dynamic=getattr(args, "dynamic", False),
            )
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    
    scaler = GradScaler() if (args.amp and device.type == "cuda") else None

    loss_fn = ClassWeightedLoss()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = f"bf{args.base_filters}_depth{args.layers}"
    metrics_path = out_dir / f"{ckpt_stem}_metrics.csv"
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch","split","loss","dice","lr","batch", "filters", "timestamp"])

    best_dice = -math.inf
    # Patience didn't work that well. leaving here for potential future use.
    # Meant to be an early stop if there is no improvement for multiple epochs
    patience_left = 12   
    lr_sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    for epoch in range(1, args.epochs + 1):

        trn_loss, trn_dice = train_one_epoch(model, loader_trn, opt, loss_fn,
                                            device, scaler,augs_per_sample=args.augs_per_sample)
        
        
        lr_sched.step()
        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}  lr={current_lr:.6f}")
        lr_now = opt.param_groups[0]["lr"]

       # ── optional validation ────────────────────────────────────────
        if args.val_every and epoch % args.val_every == 0:
            val_loss, val_dice = validate(model, loader_val, loss_fn, device)
            
            split_metrics = [("train", trn_loss, trn_dice),
                            ("val",   val_loss, val_dice)]
            
            if val_dice > best_dice:
                best_ep = epoch
                best_dice = val_dice
                
                patience_left = 12          # reset
            else:
                patience_left -= 1
                if patience_left == 13:
                    print(f"Early stop at {epoch}")
                    break
        else:
            split_metrics = [("train", trn_loss, trn_dice)]

       # ── write CSV rows ─────────────────────────────────────────────
        now = time.time()
        with open(metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            for split, L, D in split_metrics:
                w.writerow([epoch, split, f"{L:.6f}", f"{D:.6f}", lr_now,args.batch_size, args.base_filters, int(now)])

       # ── console print ──────────────────────────────────────────────
        msg = f"Ep {epoch:03}/{args.epochs} · "
        msg += f"train  L={trn_loss:.4f}  D={trn_dice:.3f}"
        if "val" in [s for s,_,_ in split_metrics]:
            msg += f" | val  L={val_loss:.4f}  D={val_dice:.3f}"
        msg += f" | lr={lr_now:.2e}"
        print(msg)

       # ── checkpointing ─────────────────────────────────────────────
        dice_for_ckpt = val_dice if "val" in msg else trn_dice
        if dice_for_ckpt > best_dice:
            best_dice = dice_for_ckpt
            torch.save(model.state_dict(), out_dir / f"{ckpt_stem}_best.pth")

        # Periodic checkpoint --------------------------------------------------
        if args.save_every and epoch % args.save_every == 0:
            torch.save(model.state_dict(), out_dir / f"{ckpt_stem}_ep{epoch:03d}.pth")
        
     
if __name__ == "__main__":
 
    main(
            [       
    "--root", r"C:\Projects\Github\WormSegmentor\Output\TrainData\Duplicates",
    "--epochs", "200",
    "--batch-size","4" ,
    "--lr", '2e-3',
    "--layers", '4',
    "--device", "cuda",
    "--base-filters", '50',
    "--save-every", "5",
    "--val-split", "0.12",
    "--val-every", "5",
    "--augs-per-sample", "5",
    "--compile",
]
)
