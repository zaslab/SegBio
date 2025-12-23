# Whole-worm UNet segmentation pipeline

This documentation covers the end-to-end workflow for whole-worm instance segmentation:

1. **Annotate** worms in MATLAB (centerlines + widths + head clicks)
2. **Build a training dataset** (UUID folders with image + masks)
3. **Train** a UNet to predict foreground / boundary / seed maps
4. **Run inference** in an interactive Napari GUI (edit + re-segment + export)
5. (Optional) **Compile** the inference GUI as a standalone app

Start here:
- **User Guide → Pipeline overview**
- **Getting Started → Quickstart**
