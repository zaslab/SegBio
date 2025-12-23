# Run inference (worm_editor_gui)

## Goal
Run the trained model on an image, optionally edit boundary/seed layers, and export final instance labels + measurements.

## Inputs
- Image (`.tif`)
- Model weights (`.pth`)

## Outputs
- `*_labels.npy`
- `*_boundary.png`
- `*_seed.png`
- `*_sizes.csv`

## Next
For details of post-processing + GUI logic, see **Developer Guide → GUI + postproc deep-dive**.
