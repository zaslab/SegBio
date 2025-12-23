# GUI + post-processing (postproc.py and worm_editor_gui.py)

This is a "walk through the file from top to bottom" explanation for postproc.py and worm_editor_gui.py:
what each part does, how it works, why certain choices were made, and a few simple alternatives for the major design decisions.


# postproc.py - step by step (what it does + why)

## Imports / overall intent

This module converts model outputs (either UNet logits/prob maps or user-edited maps from the GUI) into instance labels using a marker-based watershed pipeline, then optionally filters instances by geometry (length/width/ratio).

It relies on:
- keep_big_components to remove small blobs
- watershed + distance_transform_edt to split instances
- medial_axis / regionprops to estimate worm-like length/width for filtering


## _get_skel_length(p) / _get_skel_width(p)

Uses medial_axis(p.image, return_distance=True) inside each region (from regionprops) to get:
- length ~= number of skeleton pixels
- width ~= max(2*distance) on skeleton pixels

Simple alternatives:
- Use geodesic length along a skeleton graph (more accurate, more code).


## _get_labels_from_logits(...)

This is a secondary segmentation routine that does a watershed directly on -fg_prob using markers derived from seed.

Key steps:
- core = seed > 0.2 and remove_small_objects(core, min_size=200)
- markers = label(core)
- fg = fg_prob > fg_thr
- labels = watershed(-fg_prob, markers=markers, mask=fg)

If initial == True:
- it computes a "boundary ring" for each label by eroding and taking mask & ~eroded
- then sets those ring pixels to 0 (deletes the rim from the label)
- returns that ring as bnd

Why these choices:
- Using -fg_prob as the watershed "elevation" makes basins form at high-probability zones.
- Making markers from a stricter "core" prevents over-fragmentation.
- The "delete the rim pixels" trick tries to prevent touching objects from sticking together and prevents the label map from including ambiguous boundary pixels.


## _sizes_from_labels(labels) and calculate_sizes_in_image(labels)

_sizes_from_labels:
- regionprops -> compute skeleton length and width per label

calculate_sizes_in_image:
- computes Area by iterating label IDs and counting pixels
- adds Lengths/Widths/Ratios into a DataFrame


## filter_labels_by_axes(...)

Filters labels by:
- min/max length
- min/max width
- min/max ratio
- optionally removes objects touching border

Then:
- it zeroes removed IDs
- relabel_sequential to pack IDs again

Why this exists:
- in worm segmentation you often get tiny debris blobs, fat merges, or partial worms on borders
- length/width/ratio filters are a cheap quality gate

Simple alternatives:
- Filter only by area (simpler but less worm-specific).
- Filter by bbox aspect ratio (fast proxy).


## _instances_from_maps(fg_bool, bnd_bool, seed_bool, ...)

This is the main instance extraction primitive:

- rim = find_boundaries(fg_bool, mode="outer")
- mask = fg_bool & ~(bnd_bool | rim)
- markers = consolidate_markers(seed_bool, mask)
- dist = distance_transform_edt(mask)
- labels = watershed(-dist, markers, mask, compactness=...)
- remove_small_objects(labels, min_size=...)

Why this choice:
- marker-controlled watershed on distance transform is a standard way to turn a semantic mask into instances
- boundary strokes act as "do not cross" constraints

Simple alternatives:
- Watershed on -fg_prob (like _get_labels_from_logits) instead of distance.
- Use skimage.feature.peak_local_max on dist for markers (instead of consolidate_markers).


## split_instances_v2(...)

This is the public API used by the GUI.

It supports two modes:

Model mode:
- pred_logits (3,H,W) -> sigmoid -> threshold into fg/seed/bnd

GUI mode:
- uses fg_prob plus edited bnd_edit, seed_edit

Then it does:
- keep_big_components on fg/seed/bnd with fairly large min_areas (400/200/600)
- bnd = bnd & ~seed (don't let boundary block marker pixels)
- bnd = thin(bnd, max_num_iter=20) (sharpen lines)
- watershed via _instances_from_maps
- optional geometric filtering via filter_labels_by_axes

One notable oddity:
- if bnd_edit is already boolean, it assumes seed_edit is already boolean too and doesn't threshold it
- this can behave inconsistently if you pass uint8 0/1 labels

Note:
- this differentiates the original segmentation from a later one that uses the GUI edits
- can probably be done more elegantly

Simple alternative:
- normalize explicitly every time:
  - bnd = np.asarray(bnd_edit) > 0
  - seed = np.asarray(seed_edit) > 0
  regardless of dtype


## consolidate_markers(seed_bool, mask_bool)

For each connected component of the allowed region (mask):
- prefer pixels that are inside seed_bool; otherwise allow any pixel
- pick the pixel with maximum distance transform value (most centered)
- place exactly one marker per component

Why this choice:
- prevents multiple seeds inside one worm from creating over-splitting
- still guarantees every allowed FG component gets a marker

Simple alternatives:
- Use connected components of seed_bool as markers directly (simpler, but more splitting/fragility).
- Use peak_local_max(dist) with a minimum distance.


# worm_editor_gui.py - step by step (what it does + why)

This is a Napari + magicgui desktop tool for:
- loading an image
- running the UNet
- showing editable "Boundary" and "Seed" layers
- re-running watershed instance splitting using split_instances_v2
- exporting results and per-instance size CSV


## Imports + configuration

Pulls in napari, magicgui, torch, skimage, scipy, imageio.

Imports from postproc:
- split_instances_v2
- _get_labels_from_logits
- calculate_sizes_in_image

DEFAULTS defines brush sizes, thresholds, compactness, and visibility toggles.

Why:
- centralizes GUI "feel" knobs

Alternative:
- store defaults in a small YAML/JSON config so you don't edit code to tune


## _exe_dir() and _find_default_ckpt()

These let the same script work as:
- normal Python script, or
- PyInstaller-frozen .exe

It searches next to the exe for .pth weights (also in common subfolders).

Why:
- usability: double-click app without passing a checkpoint path

Alternative:
- store last-used ckpt path in a small settings file in user home


## _remove_label_at_click(layer, event)

Mouse callback:
- click on "Instances" layer
- find the label under cursor
- set that entire label to 0
- relabel sequentially

Why:
- fast manual cleanup without painting

Alternative:
- use napari's built-in label eraser brush (slower for whole-object delete)


## load_image_any(path)

Loads:
- .mat (prefers key "in", otherwise first non-__ key; if 3D takes BF channel 1 if exists)
- image formats (png/jpg/tif), converting to grayscale if RGB
- normalizes to [0,1]


## resize_512(img)

Resizes to 512x512 using OpenCV:
- INTER_AREA when downsampling
- INTER_LINEAR when upsampling

Why:
- model was trained at fixed 512

Alternative:
- pad/crop to 512 without resampling (preserves scale)
- run fully-convolutional at native size (requires training/inference changes)


## rasterize_paths(paths, shape, width)

Converts napari Shapes paths into a thick boolean mask using:
- skimage.draw.line
- dilation

Why:
- user can draw crisp strokes as shapes, then project them into the boundary mask

Alternative:
- skip shapes and just paint boundary in a label layer


## main(argv) - GUI execution flow

### 1) Parse args, find ckpt, pick device, infer model params from filename

- --ckpt optional; if omitted find weights in the same folder
- --device auto/cpu/cuda chooses CUDA if available

Regex parses _bf(\d+) and _depth(\d+) from ckpt filename, defaulting to 32 and 5.

Why:
- avoids mismatches between checkpoint architecture and loader args without requiring a config file

Alternative:
- store architecture metadata inside checkpoint (recommended long-term)


### 2) Load the model

- load_unet(... n_channels=1, n_classes=3, base_filters=bf, depth=depth)
- move to device


### 3) Create the Napari viewer and define helpers

Defines:
- run_unet_on(img) -> runs model -> logits -> sigmoid -> fg_prob, bnd_prob, seed_prob

Defines keyboard shortcuts to speed editing (examples):
- R, ], [, v, Shift-V, d/s/a/g, P/O


### 4) load_and_predict(path)

Main "load image -> predict -> create layers" routine:

- Load + resize to 512
- Run UNet to get prob maps
- viewer.layers.clear()

Add:
- "Draw lines" shapes layer
- "BF" image layer

Initialize editable masks:
- boundary_init = (bnd > 0.25)
- seed_init = (seed > 0.35)

Try an initial segmentation via:
- _get_labels_from_logits(fg_prob=fg, seed=seed_init, initial=True)

Store:
- bnd_history = boundary_init.copy()

Add label layers:
- "Boundary edit"
- "Seed edit"
- "Instances"

Set brush sizes and select boundary layer as active.
Calls resegment() at the end.

Why these choices:
- start from thresholded predicted boundary/seed as editable starting point
- show an initial instance result immediately

Simple alternatives:
- skip _get_labels_from_logits and always use split_instances_v2 for initial too
- auto-tune thresholds instead of hard-coded 0.25/0.35


### 5) resegment() (magicgui button + key "g")

Core editing loop:

- Decide fg_thr: use slider thr if set, else default 0.1
- Read current boundary/seed edits from napari label layers
- If any paths in "Draw lines": rasterize and OR into boundary; then clear shapes
- Call split_instances_v2(... fg_prob=fg_prob, bnd_edit=bnd_edit, seed_edit=seed_edit,
  fg_thr=..., bnd_thr=0.6, min_size=32, compactness=..., plus geometry filters)
- Update/create "Instances" label layer
- If remove-mode is on, ensure click-delete callback is attached

Simple alternatives:
- live preview (auto-call on paint events) instead of manual "Re-segment"
- use only boundary edits (no seeds) and auto-place markers via distance peaks


### 6) Threshold slider (threshold_widget)

A slider (0.0-0.2) that calls run_with_threshold(thresh) on change:
- sets global thr
- calls _get_labels_from_logits(...) (but doesn't update any displayed bnd layer)
- calls resegment()

Why:
- lets you tune foreground threshold interactively


### 7) Remove mode toggle (toggle_remove)

Adds/removes the click-delete callback and changes button text accordingly.

Why:
- avoids accidental deletes unless you intentionally enable it


### 8) Export

Exports:
- labels as .npy (exact) or .png (uint16)
- boundary/seed edit layers as _boundary.png and _seed.png
- sizes CSV via calculate_sizes_in_image(labels) saved as *_sizes.csv
