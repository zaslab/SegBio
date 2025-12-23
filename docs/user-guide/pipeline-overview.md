# Pipeline overview

```mermaid
flowchart LR
  A["1. WormMarkerGUI<br>centerlines + widths + heads"] --> B["2. CreateDataSet<br>UUID folders + masks"]
  B --> C["3. TrainFlexiUnet<br>predict fg / boundary / seed"]
  C --> D["4. worm_editor_gui<br>interactive inference + edits"]
  D --> E["5. Exports<br>labels + boundary/seed + sizes"]
  D --> F["Optional: Compile app<br>PyInstaller"]
```

## Step 1 — WormMarkerGUI (MATLAB annotation)

**Input:** `.tif` microscope images  
**Outputs (per image):**
- `*_measures.mat` — Nx5 cell array (centerline, width points, length, width, worm mask)
- `*_head_coords` — Nx2 head clicks used to orient head→tail

## Step 2 — CreateDataSet (build training folders)

**Input:** folder containing step-1 outputs  
**Output:** **one UUID folder per image**, containing (not all are used later):
- `box` — Nx4 bounding boxes
- `head_coords`
- `heads_mask`, `tails_mask`
- `ignore_mask` (a “neck” band to allow flexibility near head boundary)
- `in` (original image)
- `out` (instance-labeled worm masks)
- `strain` (parsed from folder names; note: may be brittle)

## Step 3 — TrainFlexiUnet (PyTorch)

**Input:** `--root <path-to-step-2-output>`  
**Output:**
- `.pth` model weights
- `.csv` training metrics

## Step 4 — worm_editor_gui (Napari inference + edits)

**Inputs:**
- (optional) image path to open
- model weights path (defaults to weights in the same folder)

**Outputs:**
- `*_labels.npy` — per-worm instance mask labels
- `*_boundary.png` — boundary edit layer
- `*_seed.png` — seed edit layer
- `*_sizes.csv` — worm metrics (length, width, area, etc.)

## Optional — Compile inference app
See **User Guide → Compile app**.

