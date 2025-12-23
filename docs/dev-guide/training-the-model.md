# Train UNet (fg / boundary / seed)

Big picture:

Train a UNet with customizable depth and width (layers/filters) based on the masks produced by the matlab scripts.

The UNet takes an image and a 3 channel mask that contains:

- Foreground (fg) - all pixels that belong to a worm, indicating the "interesting" areas in the image.
- Boundary - the outlines of each worm
- Seed - the center line of each worm

The network learns to identify these features, and in the next step they will be used to separate individual worms from each other.

This works with a relatively small number of images, in part due to extensive augmentation during training. Each image is scaled, flipped, rotated, blurred, sheared, and darkened randomly multiple times in each epoch.

Many of the design choices here were done because of limited GPU memory. On-the-fly augmentations, deleting certain variables every few epochs, gradient accumulation, etc.

## Run training
Run through TrainFlexiUnet
Parameters:
```python
   "--root", "path/to/training data" # Path to folder with outputs of CreateDataSet.
   "--epochs", "200", # Number of training epochs 
   "--batch-size", "4" , # Number of images in training batch. Limited by GPU RAM
   "--lr", '2e-3', # Learning rate. Should scale with batch number
   "--layers", '4', # Number of layers in Unet. 4/5 works pretty well
   "--device", "cuda", # use GPU vs CPU
   "--base-filters", '32', # Width of Unet. 16/32 work pretty well
   "--save-every", "5", # Checkpoint for weights every X epochs
   "--val-split", "0.12", # Percent of data used for validation
   "--val-every", "5", # Run validation every X eps. Slower but diagnostically helpful
   "--augs-per-sample", "5", # Number of times each image is augmented in training 
```
## Step-by-step
Here is chatGPT's guide to the code, along with (manually edited) justifications for specific choices, and suggestions for alternatives:

Below is a step-by-step walkthrough of the WormSegmentor UNet training script, with:
(1) what each piece does,
(2) how it works,
(3) why those design choices were made, and
(4) simple alternatives for the big choices.


## 1) segmentor_utils.py - shared augmentations + 512 resizing helpers

What this file is for

It centralizes data augmentation (geometric + photometric) that can be applied consistently to both an image and its label maps, plus a couple utility functions to resize images/masks to 512x512.

WormAugUniversal (the main augmentation class)

__init__(...)

Stores augmentation hyperparameters:

- zoom-out probability + range
- rotation range
- optional flips
- brightness/contrast jitter
- blur probability

Why: classic "cheap but effective" set for segmentation robustness: you get viewpoint/pose variation (rot/flip), scale variation (zoom), and lighting/noise robustness (brightness/contrast/blur).

Simple alternatives

- Use torchvision.transforms.v2 with RandomAffine, RandomHorizontalFlip, etc.
- Use albumentations (often nicer for mask-safe transforms).
- Do only geometry (skip photometric) if your imaging conditions are stable.


Step-by-step: augmentation pipeline

The method "packs" image + label maps and applies the same random transform to all of them.

Step 0: Pack tensors

pack = [img]
is_label = [False]

Then it appends worm1,worm2,worm3 as label maps (is_label=True)

Why: guarantees geometry transforms stay aligned between image and labels.


Step 1: "Zoom-out" augmentation (with probability zoom_p)

- Picks a scale s in zoom_range (e.g. 0.75-0.9)
- Resizes to (newH,newW)
- Places the resized content on a full-size canvas at a random offset (py,px)

Different handling for:
- labels: nearest neighbor, then round back to integer-like values
- image: bilinear, and canvas is filled with the resized image's mean (mean-padding)

Why these choices
- Nearest for labels preserves crisp classes (no fractional labels).
- Mean-padding for the image avoids harsh black borders which can become a spurious cue.
- Zoom-out specifically helps when worms appear at different scales/positions.

Simple alternatives
- Random crop + resize (common in segmentation).
- Pad with 0 or reflect-pad instead of mean-pad.
- Use scale>1 "zoom-in" too (currently it only shrinks content).


Step 2: flips

- horizontal flip with 50% chance (if enabled)
- vertical flip with 50% chance (if enabled)

Why: worms don't have a canonical orientation in many datasets, so this is free invariance.


Step 3: affine (rotation + slight scale)

- Random angle in +/-rot_deg
- Random scale in [0.95, 1.05]
- Applies torchvision.transforms.functional.affine

Uses:
- bilinear for image
- nearest for labels, then round

Why: keeps label integrity while adding pose and size jitter.

Alternative
- Use RandomRotation + separate RandomResizedCrop.
- Use elastic deformations (powerful but can distort anatomy if overdone).


Step 4: photometric jitter (only on "visual channels")

If img has >=3 channels, apply to first 3 channels, else to 1 channel.

- brightness multiplicative jitter
- contrast adjustment around per-channel mean
- optional gaussian blur

Alternative
- Add Gaussian noise instead of blur.
- Use histogram equalization / CLAHE for microscopy-style data (careful: can change semantics).


Step 5: unpack + reassemble worm tensor

Gets worm_out1/2/3
Adds a singleton dimension back, then concatenates into (?,3,?,?)

Why: the training code wants target shaped like (B,3,H,W).


Resize utilities

- _interp_for_scale: chooses INTER_AREA when downscaling, else INTER_LINEAR for upscaling.
- resize_image_512: uses that heuristic.
- resize_mask_512: always uses INTER_NEAREST.

Why
- INTER_AREA is often best for downsampling images (less aliasing).
- masks must be nearest-neighbor.

Alternative
- Always use linear for image resizing (simpler).
- Do resizing in PyTorch (F.interpolate) to keep everything in one framework.


## 2) FlexiUnet.py - configurable-depth UNet with GroupNorm

What this file is for

Defines a parameterized UNet where you can change:
- depth (how many downsamplings)
- base_filters (width)
- n_channels, n_classes

And provides a load_unet(...) helper to load a checkpoint state_dict.


Norm(num_channels, max_groups=8) -> GroupNorm selection

Picks the largest group count g <= max_groups that divides num_channels.
Returns nn.GroupNorm(g, num_channels).

Why
- GroupNorm is much more stable than BatchNorm for small batch sizes (common in segmentation due to memory).
- The "largest divisor <= 8" gives a reasonable normalization granularity without forcing awkward channel counts.

Alternatives
- BatchNorm2d if batch sizes are reliably large.
- InstanceNorm2d (often used in style/medical imaging).
- No norm + careful init (less common).


DoubleConv

Two blocks of: Conv -> Norm -> ReLU

Why
- This is the standard UNet building block.
- bias=False because normalization has affine parameters, making conv bias redundant.

Alternatives
- Add dropout.
- Use residual blocks (ResUNet).
- Replace ReLU with SiLU/GELU.


UNet.__init__

Key logic:
- Build encoder channel plan: chans = [base_filters * 2**i for i in range(depth)]

Encoder:
- enc0 at level 0 (no pooling)
- then downs: MaxPool + DoubleConv for each deeper level

Bottleneck:
- DoubleConv at 2x deepest channels

Decoder:
- ConvTranspose2d upsampling
- concat skip features
- DoubleConv to fuse

Output head:
- 1x1 conv to n_classes

Why
- Depth lets you trade off receptive field vs compute.
- Transposed conv learns upsampling rather than fixed interpolation.

Alternatives
- Upsample (bilinear) + Conv instead of ConvTranspose (avoids checkerboard artifacts).
- Add attention gates (Attention UNet).
- Use a pretrained encoder backbone (ResNet/ConvNeXt) if you want stronger features.


forward

- Save skip tensors during encoding.
- Bottleneck.
- For each decoder step:
  - upsample
  - pad if needed to match skip shape
  - concat + decode

Why the padding
- Handles odd input sizes where pooling/upsampling can cause off-by-one shape mismatches.

Alternative
- Force input sizes divisible by 2**depth (e.g., always 512 makes this mostly moot).


load_unet(ckpt, ...)

- Instantiates UNet with given hyperparams
- loads state_dict strict
- sets eval()

Why
- Ensures architecture matches weights exactly.
- eval() prevents norm layers behaving like training.

Alternative
- Save the full model (less flexible but simpler).
- Store hyperparams in the checkpoint and reconstruct automatically.


## 3) targets.py - build a 3-channel supervision target (fg/boundary/seed)

What this file is for

It converts a label mask into three supervision maps:
- fg: foreground (worm pixels)
- boundary: boundary band near worm edges
- seed: interior "seed" targets to encourage separability / instance structure

This is a lightweight way to train a model to produce something you can later post-process into clean segmentation (especially when worms touch).


make_targets(label, boundary_width=4, seed_method="skeleton")

Step 1: foreground

- fg = (label > 0).astype(np.float32)

Why: base segmentation signal.

Alternative:
- separate "body" vs "head" classes, etc., if you want multi-class.


Step 2: boundary

- find_boundaries(label, mode="outer") -> ~1 px boundary
- dilate by a disk of radius boundary_width//2
- intersect with fg

Why
- A thick-ish boundary target makes it easier for the network to learn "don't merge touching worms".
- Intersecting with fg avoids teaching boundaries in background.

Alternatives
- Use signed distance transform as boundary target.
- Use a per-instance "touching regions" map (more complex).


Step 3: seed

Two modes:

A) "skeleton"
- Make skeleton per instance (skeletonize(label==i))
- Trim ends (removes ~15% from each end)
- Dilate slightly (2x2)
- Remove overlap with boundary
- Drop tiny blobs (min_area=40)

Why
- Skeleton seeds encourage the network to place "confidence" down the middle of objects.
- Trimming ends avoids emphasizing thin noisy tips and helps avoid seeds leaking into touching regions at worm ends.
- Removing boundaries reduces the chance seeds appear right at contact points.

Alternatives
- Use eroded foreground as seeds (seed = erosion(fg) per instance).
- Use distance transform peaks ("distance" method below) but binarize.
- Use watershed markers from per-instance centerpoints if you have them.

B) "distance"
- distance transform inside fg
- zero out a rim around boundaries
- normalize to [0,1]

Why
- This creates a smooth interior "center-ness" map.
- Good when skeletons are messy or branched.

Alternative
- Use seed = (dist > threshold) if you want a binary seed.


trim_skeleton_ends and helpers

- _endpoints: endpoints = pixels with exactly one neighbor (degree 1)
- trim_skeleton_ends: iteratively deletes endpoints for k rounds, where k ~= frac * length / 2

Why
- Prunes away endpoints that can be noisy and merge in touching worms

Alternative
- Prune by geodesic distance along skeleton, not pixel count iterations.
- Use medial axis transform (MAT) instead of skeletonize.


keep_big_components

- Labels components and removes ones smaller than min_area.

Why
- Skeleton/seed generation can leave tiny fragments; these are "label noise" for training.

Alternative
- Keep only the largest component per instance or per image.


## 4) TrainFlexiUnet.py - dataset + training loop for the 3-head target

What this script does (big picture)

It trains UNet to predict 3 output channels (fg, boundary, seed) from an input image, using:
- weighted BCE-with-logits + weighted Dice
- optional mixed precision
- optional torch.compile
- optional multiple augmentations per sample per epoch

Many of the design choices here were done because of limited GPU memory. On-the-fly augmentations, deleting certain variables every few epochs, gradient accumulation, etc.


Dataset: MatSegDataset

__init__
- lists sample folders under root
- expects each sample folder to contain in.mat and out.mat

Why
- Matches MATLAB dataset organization.

Alternative
- Preconvert to .npz or .pt for faster loading.
- Use memory-mapped arrays or LMDB if I/O becomes the bottleneck.


_load_sample(path)

Step-by-step:
- Load image from in.mat
- Load label from out.mat
- Resize both to 512x512 (resize_image_512, resize_mask_512)
- Normalize image to [0,1]
- Build 3 targets from label:
  - t = make_targets(out, boundary_width=4, seed_method="skeleton")
  - target = stack([fg,boundary,seed])   # shape (3,H,W)
- Convert targets to torch
- Also binarizes out to 0/1 (but training doesn't really use it later)

Why 512x512
- Fixed-size simplifies batching and UNet shape constraints.
- 512 is a standard sweet spot: enough detail, manageable memory.

Alternative
- Train at native resolution and random crop to patches.
- Multi-scale training (harder but sometimes better).


Loss: multihead_loss + ClassWeightedLoss

Input:
- logits: (B,3,H,W)
- target: (B,3,H,W)

Components:
- BCEWithLogits with per-channel pos_weight (boost positives differently for fg/boundary/seed)
- Dice loss computed per-channel and weighted by chan_w
- Returns bce + 0.5*dice

Why:
- Boundary pixels are sparse and important -> higher weights.
- Seed might need special emphasis but not as much as boundary.
- Pos-weight helps when positives are rare (like boundary and seeds).

Simple alternatives
- Use focal loss for boundary/seed sparsity.
- Use Tversky loss instead of Dice.
- Use a single loss like BCE only (simpler but often worse on thin boundaries).


Metric: dice_coeff

- Applies sigmoid
- thresholds at 0.5
- computes a single dice over all channels combined

Why
- Quick sanity metric.

Alternative (often better here)
- Report dice per channel (fg dice is usually the meaningful one).
- Use soft dice without thresholding for smoother feedback.


Training loop: train_one_epoch

Key behavior:
- Creates an augmenter: aug = WormAugUniversal()
- For each minibatch, it can generate augs_per_sample different augmented versions.
- It runs K micro-passes and accumulates gradients, dividing loss by K so total gradient scale stays consistent.

Why do "augmentations per sample" inside the loop?
- It's like "virtual batch expansion" without increasing DataLoader size.
- Helps when dataset is small or when you want many random views per epoch.

Simple alternatives
- Put augmentation inside the Dataset / DataLoader transform and just iterate normally.
- Use larger batch size (if GPU allows) instead of repeated augmentation.
- Use gradient accumulation over different batches rather than repeated aug views.


Validation: validate
- sets eval()
- forward pass
- computes loss + dice


Main training orchestration: main

- Seeds RNG
- splits dataset into train/val via random_split
- builds model: UNet(depth=args.layers, base_filters=args.base_filters)
- optional torch.compile
- optimizer: AdamW
- StepLR decay
- logs to a CSV in out_dir

Why AdamW
- Good default; decoupled weight decay tends to behave well.

Alternatives
- Adam (simpler)
- SGD+momentum (sometimes better final generalization, needs tuning)
