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
- `heads_mask`, `tails_mask`, `ignore_mask` (optional / legacy depending on training setup)

## Tips
- Keep dataset generation deterministic (log params + source folders).
- Store the exact commit hash / version used to generate the dataset.
