#!/usr/bin/env python3
"""
Generate MkDocs-ready documentation files for the Whole-worm UNet pipeline.

Usage:
  python make_docs.py  # writes mkdocs.yml + docs/...

Optional:
  pip install mkdocs mkdocs-material
  mkdocs serve
"""

from pathlib import Path

FILES = {
    'mkdocs.yml': r"""site_name: Whole-worm UNet Docs
theme:
  name: material
markdown_extensions:
  - admonition
  - toc:
      permalink: true
  - pymdownx.superfences
  - pymdownx.details
plugins:
  - search
nav:
  - Home: index.md
  - Getting Started:
      - Installation: getting-started/install.md
      - Quickstart: getting-started/quickstart.md
  - User Guide:
      - Pipeline overview: user-guide/pipeline-overview.md
      - Annotation (WormMarkerGUI): user-guide/annotation-wormmarkergui.md
      - Build training dataset: user-guide/dataset.md
      - Training: user-guide/training.md
      - Inference GUI: user-guide/inference-gui.md
      - Outputs: user-guide/exports.md
      - Compile app: user-guide/compile-app.md
  - Developer Guide:
      - Architecture: dev-guide/architecture.md
      - Training deep-dive: dev-guide/training-deep-dive.md
      - GUI + postproc deep-dive: dev-guide/gui-postproc-deep-dive.md
      - assign_heads_to_worms deep-dive: dev-guide/assign-heads.md
      - GenerateMask.m deep-dive: dev-guide/generate-mask.md
  - Reference:
      - CLI: reference/cli.md
      - File formats: reference/file-formats.md
      - Troubleshooting: reference/troubleshooting.md
""",
    'docs/index.md': r"""# Whole-worm UNet segmentation pipeline

This documentation covers the end-to-end workflow for whole-worm instance segmentation:

1. **Annotate** worms in MATLAB (centerlines + widths + head clicks)
2. **Build a training dataset** (UUID folders with image + masks)
3. **Train** a UNet to predict foreground / boundary / seed maps
4. **Run inference** in an interactive Napari GUI (edit + re-segment + export)
5. (Optional) **Compile** the inference GUI as a standalone app

Start here:
- **User Guide → Pipeline overview**
- **Getting Started → Quickstart**
""",
    'docs/getting-started/install.md': r"""# Installation

> This is a practical checklist. Keep it short and concrete.

## Python environment
- Create a dedicated environment (conda/venv).
- Install PyTorch (GPU if available) + project requirements.

## GUI dependencies (Napari / Qt)
Napari requires a Qt backend. If the GUI fails to launch, install/verify:
- `napari[all]`
- `pyqt5` or `pyside2` (depending on your stack)

## MATLAB (for annotation + mask generation)
MATLAB is used for the **WormMarkerGUI** + the scripts that generate training masks.

## Project layout expectations
The pipeline expects:
- raw images in a folder
- MATLAB outputs saved alongside / in a known location
- training data generated into a root folder with UUID subfolders
""",
    'docs/getting-started/quickstart.md': r"""# Quickstart

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
""",
    'docs/user-guide/pipeline-overview.md': r"""# Pipeline overview

```mermaid
flowchart LR
  A[1. WormMarkerGUI<br/>manual centerlines + widths + heads] --> B[2. CreateDataSet<br/>UUID folders + masks]
  B --> C[3. TrainFlexiUnet<br/>UNet predicts fg/boundary/seed]
  C --> D[4. worm_editor_gui<br/>interactive inference + edits]
  D --> E[5. Exports<br/>labels + boundary/seed + sizes]
  D --> F[(Optional) Compile app<br/>PyInstaller]
