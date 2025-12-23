# Build training data (CreateDataSet)

## Goal
Convert MATLAB annotation outputs into a training dataset: **UUID folders** each holding an image and
the masks/metadata needed for training.

## Inputs
- Folder containing `*_measures.mat` and `*_head_coords`

## Outputs
Per-image UUID folder contains (key items):
- `in` — original image
- `out` — instance-labeled worm masks
- `heads_mask`, binary image with all head masks
- `tails_mask`, binary image with all tail masks
- `ignore_mask`, binary image with all “neck” masks (adds flexibility to head predictions)
- `head_coords`, Nx2 list of points in heads. Used to orient mask head-to-tail
- `box`, Nx4 array of x y  of corners of bounding box for each worm.
- `strain`, extracted from image folder names (might be broken)


