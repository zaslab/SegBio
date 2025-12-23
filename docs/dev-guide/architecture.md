# Architecture overview

## Modules (high level)
- **MATLAB annotation + mask generation**
  - WormMarkerGUI
  - CreateDataSet
      - GenerateMask.m (derives masks from centerlines + head clicks)
      - assign_heads_to_worms (matches head clicks to worms robustly)

- **Python training**
  - FlexiUnet (construct model)
  - TrainFlexiUnet (training loop)
  - segmentor_utils (dataset reading + augmentations)
  - targets (foreground/boundary/seed construction)
  

- **Python inference + post-processing**
  - worm_editor_gui (Napari GUI for edits + export)
    - postproc (post-processing to split instances)
    
  

