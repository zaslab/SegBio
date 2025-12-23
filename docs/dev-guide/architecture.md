# Architecture overview

## Modules (high level)
- **MATLAB annotation + mask generation**
  - WormMarkerGUI
  - GenerateMask.m (derives masks from centerlines + head clicks)
  - assign_heads_to_worms (matches head clicks to worms robustly)

- **Python training**
  - dataset reading + augmentations (segmentor_utils)
  - targets (foreground/boundary/seed construction)
  - model (FlexiUnet)
  - training loop (TrainFlexiUnet)

- **Python inference + post-processing**
  - model inference to prob maps
  - post-processing to split instances (distance transform + watershed)
  - geometry filtering (length/width heuristics)
  - Napari GUI for edits + export
