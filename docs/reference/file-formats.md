# File formats (reference)

## `*_measures.mat` (from WormMarkerGUI)
- col 1: m×2 centerline (x,y)
- col 2: 2×2 widest-part points (x,y)
- col 3: length
- col 4: width
- col 5: worm mask image

## `*_head_coords`
- K×2 clicked head points used for orientation / head-tail masks

## UUID training folders
- `in` image
- `out` instance labels
- optional masks: heads/tails/ignore + metadata arrays
