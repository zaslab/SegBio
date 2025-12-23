# Outputs and file formats

## Inference outputs
- `*_labels.npy` — integer label image (0 background, 1..N worms)
- `*_boundary.png` — boundary edit layer
- `*_seed.png` — seed edit layer
- `*_sizes.csv` — per-worm metrics (length, width, area, etc.)

## Training dataset outputs
Each UUID folder holds:
- the original image (`in`)
- instance labels (`out`)
- optional masks/metadata used by older pipelines
