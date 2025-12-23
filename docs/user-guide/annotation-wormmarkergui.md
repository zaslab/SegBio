# Annotate images (WormMarkerGUI)

## Goal
Mark each worm’s **centerline**, **widest-point width**, and (optionally) **head click(s)**.

## Inputs
- Raw microscope images (`.tif`)

## Outputs
- `*_measures.mat` (centerlines + widths + lengths/widths + a worm mask image)
- `*_head_coords` (Nx2 head points)

## Notes
- Head clicks are used to orient masks head-to-tail (important for head/tail derived masks).
