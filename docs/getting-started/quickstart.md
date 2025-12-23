# Quickstart

## Goal
Run the inference GUI on a single image and export instance masks + measurements.

## Steps
1. Open the GUI (`worm_editor_gui.py`)
2. Load model weights (`.pth`)
3. Load an image (`.tif`)
4. (Optional) edit **boundary** / **seed** layers
5. Re-segment and export outputs:
   - `*_labels.npy`
   - `*_boundary.png`
   - `*_seed.png`
   - `*_sizes.csv`

If you have not trained weights yet, see **User Guide → Training**.
