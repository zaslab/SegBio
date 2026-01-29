# worm_editor_gui.py
# Minimal interactive GUI for the worm UNet + watershed post-proc, with manual fixes.
# Dependencies: napari, magicgui, numpy, torch, skimage, scipy, imageio, (optional) tifffile, opencv-python
#
# Usage:
#   python -m pip install "napari[all]" magicgui numpy torch torchvision torchaudio \
#       scikit-image scipy imageio tifffile opencv-python
#   python worm_editor_gui.py --ckpt /path/to/unet_out_bf32_epXXX.pth  [--image some.png]
#
# What you can do:
#   • Load image (menu or CLI)  • Run UNet  • See FG/Boundary/Seed maps
#   • Paint to add/delete boundary lines and seeds  • Re-segment (watershed)
#   • Export labels / edited maps

import sys, argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio.v2 as imageio
from scipy.io import loadmat
from scipy import ndimage as ndi
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import binary_dilation, remove_small_objects, thin
from skimage.draw import line as draw_line
import warnings
from postproc import split_instances_v2, _get_labels_from_logits
from postproc import calculate_sizes_in_image
# ---- local project imports (adjust if needed) -------------------------------
# We try relative imports first; if they fail, add this script's folder to sys.path
try:
    from FlexiUnet import load_unet  # your loader returns a model set to eval()
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent))
    from FlexiUnet import load_unet

try:
    # We import helpers only to *reuse* consolidate_markers, but we keep our
    # own resegment() here so it respects user edits.
    from postproc import consolidate_markers
except Exception:
    consolidate_markers = None

# ---------------- small I/O helpers ----------------
DEFAULTS = dict(
    boundary_brush_size=4,   # px
    seed_brush_size=4,       # px
    boundary_label=1,        # paint adds lines
    seed_label=1,            # paint adds seeds
    instances_opacity=0.5,
    fg_thr=0.1,
    compactness=0.001,
    line_width=1,            # for rasterizing "Draw lines"
    show_fg=False,
    show_bnd=False,
    show_seed=False,
)
from skimage.segmentation import relabel_sequential

def _exe_dir() -> Path:
    """Folder that contains the running app.
    - onefile/onedir build: folder of the .exe
    - normal Python: folder of this .py file
    """
    if getattr(sys, "frozen", False):        # PyInstaller
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent

def _find_default_ckpt() -> Path:
    """Look next to the exe (and common subfolders) for a .pth file."""
    base = _exe_dir()
    candidates = []
    for sub in ("", "weights", "models", "checkpoints"):
        d = base / sub if sub else base
        if d.is_dir():
            candidates += sorted(d.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(
            "No .pth checkpoint found next to the app. "
            "Place your weights file next to the .exe (or in weights/models/checkpoints)."
        )
    # pick the first (or implement your own selection rule)
    return candidates[0]
def _remove_label_at_click(layer, event):
    """Mouse callback for the Instances layer: delete clicked label."""
    if event.type != "mouse_press" or event.button != 1:
        return
    # Which label is under the cursor?
    try:
        lab = layer.get_value(event.position, world=True)
    except Exception:
        # Fallback for older napari: map coords manually
        data_coords = layer.world_to_data(event.position)
        rc = tuple(int(round(x)) for x in data_coords[-2:])
        H, W = layer.data.shape[-2:]
        if not (0 <= rc[0] < H and 0 <= rc[1] < W):
            return
        lab = layer.data[rc]
    if not lab or lab == 0:
        return
    # Delete that label and relabel sequentially
    data = layer.data.copy()
    data[data == lab] = 0
    new, _, _ = relabel_sequential(data)
    layer.data = new.astype(np.int32)

def _first_key(mat: dict) -> str:
    return next(k for k in mat.keys() if not k.startswith("__"))



def load_image_any(path: Path) -> np.ndarray:
    """Return a float32 HxW array in [0,1].
       Supports: PNG/JPG/TIF (grayscale or RGB), MAT (expects 'in')."""
    path = Path(path)
    if path.is_dir():
        # try typical raw sample layout with in.mat
        cand = path / "in.mat"
        if not cand.exists():
            raise FileNotFoundError(f"Folder {path} has no in.mat")
        path = cand

    if path.suffix.lower() == ".mat":
        mat = loadmat(path, simplify_cells=True)
        arr = mat.get("in", None)
        if arr is None:
            arr = mat[_first_key(mat)]
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            img = arr
        elif arr.ndim == 3:
            # by convention in your repo, channel 1 is BF
            if arr.shape[-1] >= 2:
                img = arr[..., 1]
            else:
                img = arr[..., 0]
        else:
            raise ValueError(f"Unsupported .mat shape: {arr.shape}")
        # Normalize if not already in 0-1
        if img.max() > 1.0:
            # Often raw BF is 0..255
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        return img
    else:
        img = imageio.imread(path)
        if img.ndim == 3:
            # convert to grayscale (simple mean; you can swap to BF channel if needed)
            img = img.mean(axis=-1)
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        return np.clip(img, 0.0, 1.0).astype(np.float32)

def resize_512(img: np.ndarray) -> np.ndarray:
    """Resize to 512x512 with area/linear depending on scale direction (OpenCV-style)."""
    import cv2
    H, W = img.shape[:2]
    interp = cv2.INTER_AREA if (512 < H or 512 < W) else cv2.INTER_LINEAR
    return cv2.resize(img, (512, 512), interpolation=interp)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _get_layer(viewer, name):
    try:
        return viewer.layers[name]   # LayerList supports name indexing
    except KeyError:
        return None


# ---------------- rasterize 'line' shapes into a mask ----------------
def rasterize_paths(paths, shape, width=3):
    """paths: iterable of arrays with shape (N_points, 2) in (row, col) coords.
       returns a boolean mask with thick polyline strokes.
    """
    H, W = shape
    out = np.zeros((H, W), dtype=bool)
    for pts in paths:
        pts = np.asarray(pts, dtype=np.float32)
        pts = np.round(pts).astype(int)
        for i in range(len(pts) - 1):
            r0, c0 = pts[i]
            r1, c1 = pts[i+1]
            rr, cc = draw_line(r0, c0, r1, c1)
            rr = np.clip(rr, 0, H-1); cc = np.clip(cc, 0, W-1)
            out[rr, cc] = True
    if width > 1:
        from skimage.morphology import disk, binary_dilation
        out = binary_dilation(out, footprint=disk(width // 2))
    return out

# ---------------- GUI (napari + magicgui) ----------------
def main(argv):
    ap = argparse.ArgumentParser(description="Interactive worm UNet editor")
    ap.add_argument("--ckpt", type=Path, default=None,
                help="Path to UNet .pth (if omitted, search next to the .exe)")
    ap.add_argument("--image", type=Path, default=None, help="Optional image (PNG/TIF/JPG/MAT or folder with in.mat)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args(argv)
    ckpt_path = args.ckpt if args.ckpt is not None else _find_default_ckpt()
    args.ckpt = ckpt_path
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    # infer base_filters from ckpt name, fallback to 32
    import re
    m = re.search(r"_bf(\d+)", args.ckpt.stem)
    bf = int(m.group(1)) if m else 32
    
    d = re.search(r"_depth(\d+)", args.ckpt.stem)
    depth = int(d.group(1)) if d else 5    
    # load model
    model = load_unet(args.ckpt, device=device, n_channels=1, n_classes=3, 
                      base_filters=bf,depth=depth)
    model_device = torch.device(device)

    # lazy imports (GUI libs)
    import napari
    from magicgui import magicgui

    # state
    cur_image = None
    fg_prob = None
    bnd_prob = None
    seed_prob = None
    bnd_history = None
    thr = None
    # viewer and layers
    viewer = napari.Viewer(title="Worm UNet Editor")

    # helper to (re)run UNet on current image
    def run_unet_on(img: np.ndarray):
        nonlocal fg_prob, bnd_prob, seed_prob
        model.eval()
        with torch.no_grad():
            t = torch.from_numpy(img)[None, None].to(model_device)
            logits = model(t)[0].detach().cpu().numpy()  # (3,H,W) LOGITS
            probs = sigmoid(logits)
        fg_prob, bnd_prob, seed_prob = probs[0], probs[1], probs[2]
        return fg_prob, bnd_prob, seed_prob

    def _active_index():
        sel = list(viewer.layers.selection)
        if not sel:
            return None
        active = sel[-1]
        return list(viewer.layers).index(active)
    
    def _set_active_by_index(i):
        if i is None or len(viewer.layers) == 0:
            return
        i = max(0, min(i, len(viewer.layers) - 1))
        layer = viewer.layers[i]
        viewer.layers.selection = [layer]
    

    @viewer.bind_key("R")
    def _toggle_remove_key(v):
        toggle_remove()  # same as pressing the button
        
    # Next / previous layer
    @viewer.bind_key("]")
    def _next_layer(v):
        i = _active_index()
        if i is None: return
        _set_active_by_index(i + 1)
    
    @viewer.bind_key("[")
    def _prev_layer(v):
        i = _active_index()
        if i is None: return
        _set_active_by_index(i - 1)
    
    # Toggle visibility of the active layer
    @viewer.bind_key("v")
    def _toggle_active_visibility(v):
        sel = list(viewer.layers.selection)
        if not sel: return
        L = sel[-1]
        L.visible = not L.visible
    
    # SOLO active layer (hide others), press again to restore
    @viewer.bind_key("Shift-V")
    def _solo_active(v):
        global _last_vis
        sel = list(viewer.layers.selection)
        if not sel: return
        active = sel[-1]
        if not _solo_on["v"]:
            # save current visibilities & hide all others
            _last_vis = {ly.name: ly.visible for ly in viewer.layers}
            for ly in viewer.layers:
                ly.visible = (ly is active)
            _solo_on["v"] = True
        else:
            # restore
            for ly in viewer.layers:
                if ly.name in _last_vis:
                    ly.visible = _last_vis[ly.name]
            _solo_on["v"] = False
    
    # Quick toggles for specific layers (if they exist)
    def _toggle(name):
        ly = viewer.layers[name] if name in viewer.layers else None
        if ly is not None:
            ly.visible = not ly.visible
    
    
    @viewer.bind_key("d")   # BND prob
    def _toggle_bnd(v):  _toggle("Boundary edit")
    
    @viewer.bind_key("s")   # SEED prob
    def _toggle_seed(v): _toggle("Seed edit")
    
    @viewer.bind_key("a")   # Instances on/off
    def _toggle_instances(v): _toggle("Instances")
    
    @viewer.bind_key("g")   # Instances on/off
    def _toggle_resegment(v): resegment()
    
    
    # Make Boundary edit the active paint layer quickly
    @viewer.bind_key("P")   # 'P' for (boundary) Paint
    def _focus_boundary_edit(v):
        if "Boundary edit" in viewer.layers:
            viewer.layers.selection = [viewer.layers["Boundary edit"]]
    
    @viewer.bind_key("O")   # 'O' for seed edit
    def _focus_seed_edit(v):
        if "Seed edit" in viewer.layers:
            viewer.layers.selection = [viewer.layers["Seed edit"]]
    # load image, run unet, setup layers
    def load_and_predict(path: Path):
        nonlocal cur_image, bnd_history, thr
        img = load_image_any(path)
        if img.shape != (512, 512):
            img = resize_512(img)
        cur_image = img

        fg, bnd, seed = run_unet_on(cur_image)
        
        # clear any existing layers
        viewer.layers.clear()
        # base
        # shapes layer for crisp lines (optional)
        viewer.add_shapes(name="Draw lines", shape_type="path", edge_width=2)
        viewer.add_image(cur_image, name="BF", colormap="gray")
        
        # editable layers (Labels: 0=off, 1=on)
        boundary_init = (bnd > 0.25).astype(np.uint8)
        seed_init     = (seed > 0.35).astype(np.uint8)

        try:
            thr = None
            labels, boundary_init = _get_labels_from_logits(fg_prob=fg,
                                                            seed=seed_init, 
                                                            initial=True) 
            bnd_history = boundary_init.copy()
        except Exception as e:
            warnings.warn(f"Initial watershed failed: {e}")
            labels = np.zeros_like(boundary_init, dtype=np.int32)
            
        b_layer = viewer.add_labels(boundary_init, name="Boundary edit")
        s_layer = viewer.add_labels(seed_init,     name="Seed edit")

        # defaults
        b_layer.brush_size = DEFAULTS["boundary_brush_size"]
        b_layer.selected_label = DEFAULTS["boundary_label"]
        s_layer.brush_size = DEFAULTS["seed_brush_size"]
        s_layer.selected_label = DEFAULTS["seed_label"]

        # (optional) make boundary layer active so the brush is ready
        viewer.layers.selection = [b_layer]
    
        
        viewer.add_labels(labels, name="Instances", opacity=0.5)
        # if remove mode is on, ensure callback is attached
        inst_layer = _get_layer(viewer, "Instances")
        if _remove_mode.get("on") and _remove_label_at_click not in inst_layer.mouse_drag_callbacks:
            inst_layer.mouse_drag_callbacks.append(_remove_label_at_click)
        resegment()
    # ---- UI widgets ------------------------------------------------
    @magicgui(call_button="Open image…")
    def open_image():
        from qtpy.QtWidgets import QFileDialog
        fname, _ = QFileDialog.getOpenFileName(caption="Choose image or in.mat",
                                               filter="Images (*.png *.tif *.tiff *.jpg *.jpeg *.mat);;All files (*)")
        if fname:
            load_and_predict(Path(fname))

    @magicgui(call_button="Re-segment (use edits)")
    def resegment():
    # use defaults (or read line width from the shapes layer)
       
        if thr is not None: 
            fg_thr = thr
        else:
            fg_thr = DEFAULTS["fg_thr"]
        compactness = DEFAULTS["compactness"]
        line_width = DEFAULTS["line_width"]
        if "Draw lines" in viewer.layers:
        # use the on-screen edge width if you want it to control raster thickness
            try:
                line_width = int(round(viewer.layers["Draw lines"].edge_width))
            except Exception:
                    pass
        if cur_image is None:
            return
        # gather edited maps
    
        bnd_edit_layer  = _get_layer(viewer, "Boundary edit")
        seed_edit_layer = _get_layer(viewer, "Seed edit")        
        inst_layer      = _get_layer(viewer, "Instances")
        lines_layer     = _get_layer(viewer, "Draw lines")
        
        if bnd_edit_layer is None or seed_edit_layer is None:
            warnings.warn("Editable layers missing; run inference first.")
            return

        bnd_edit = bnd_edit_layer.data.astype(bool)
        seed_edit = seed_edit_layer.data.astype(bool)

        # union with drawn paths (if any)
        if lines_layer is not None and len(lines_layer.data) > 0:
      
            mask_lines = rasterize_paths(lines_layer.data, bnd_edit.shape, width=line_width)
            bnd_edit = bnd_edit | mask_lines
            # optional: clear shapes after projecting
            lines_layer.data = []


        labels = split_instances_v2(
            pred_logits=None,                 # use edited-maps path
            fg_prob=fg_prob,                  # UNet FG prob (float)
            bnd_edit=bnd_edit,                # edited boundary (bool/0-1)
            seed_edit=seed_edit,              # edited seeds (bool/0-1)
            fg_thr=fg_thr,
            bnd_thr=0.6,
            min_size=32,
            compactness=compactness,
            min_length=10, max_width=60,min_ratio=5,
            remove_touching_border=False,
            )   
        

        if inst_layer is None:
            inst_layer = viewer.add_labels(labels, name="Instances", opacity=0.5)
        else:
            inst_layer.data = labels
        
        # keep the remove-click active if the mode is on
        if _remove_mode.get("on") and _remove_label_at_click not in inst_layer.mouse_drag_callbacks:
            inst_layer.mouse_drag_callbacks.append(_remove_label_at_click)

    # state toggle
    _remove_mode = {"on": False}
    
    def run_with_threshold(thresh: float):
        
        """Whatever you want to recompute when slider moves."""
        nonlocal thr
        thr =  thresh
        bnd_edit_layer = _get_layer(viewer, "Boundary edit")     
        bnd_edit = bnd_edit_layer.data.astype(bool)
        seed_edit_layer = _get_layer(viewer, "Seed edit")     
        seed_edit = seed_edit_layer.data.astype(bool)
        labels, bnd = _get_labels_from_logits(fg_prob=fg_prob,
                                                        seed=seed_edit, 
                                                        fg_thr=thr, 
                                                        bnd=bnd_edit,
                                                        outer_outline=True,
                                                        initial=False) 
        #viewer.layers["BND prob"].data   = bnd
        resegment()
        
    # e.g. recompute segmentation here

    @magicgui(
        threshold={"label": "Threshold", "min": 0.0, "max": 0.2, "step": 0.01},
        auto_call=True,   # <- calls the function whenever the slider moves
        )
    def threshold_widget(threshold: float = DEFAULTS["fg_thr"]):
        #print(thr)
        run_with_threshold(threshold)

    viewer.window.add_dock_widget(threshold_widget, area="right")
    
    @magicgui(call_button="Remove labels (click)")
    def toggle_remove():
        _remove_mode["on"] = not _remove_mode["on"]
        inst = _get_layer(viewer, "Instances")
        if inst is None:
            return
        # attach/detach the mouse callback
        if _remove_mode["on"]:
            if _remove_label_at_click not in inst.mouse_drag_callbacks:
                inst.mouse_drag_callbacks.append(_remove_label_at_click)
            toggle_remove.call_button.text = "✅ Removing… (click to delete)"
        else:
            # remove if present
            try:
                inst.mouse_drag_callbacks.remove(_remove_label_at_click)
            except ValueError:
                pass
            toggle_remove.call_button.text = "Remove labels (click)"
    
    # after you create/show Instances for the first time:
    # viewer.add_labels(labels, name="Instances", opacity=0.5)
    
    # add the toggle to the right dock
    viewer.window.add_dock_widget(toggle_remove, area="right")

    

    @magicgui(call_button="Export…")
    def export():
        if "Instances" not in viewer.layers:
            warnings.warn("Nothing to export (no Instances layer).")
            return
        from qtpy.QtWidgets import QFileDialog
        base, _ = QFileDialog.getSaveFileName(caption="Save segmentation (basename)",
                                              filter="NumPy (*.npy);;PNG (*.png)")
        if not base:
            return
        base = Path(base)
        labels = viewer.layers["Instances"].data.astype(np.int32)
        bnd_edit = viewer.layers["Boundary edit"].data.astype(np.uint8) * 255
        seed_edit = viewer.layers["Seed edit"].data.astype(np.uint8) * 255
        sizes = calculate_sizes_in_image(labels)
        try:
            # save labels as .npy (exact) if chosen, else as PNG
            if base.suffix.lower() == ".npy":
                np.save(base, labels)
                print(base)
                print(str(Path(base).stem)+'sizes')
                csv_path = (Path(str(base)[0:-4]+'_sizes.csv'))
                
                sizes.to_csv(csv_path, index=False)
               
                imageio.imwrite(base.with_name(base.stem + "_boundary.png"), bnd_edit)
                imageio.imwrite(base.with_name(base.stem + "_seed.png"), seed_edit)
            else:
                # default: PNGs
                imageio.imwrite(base, labels.astype(np.uint16))  # preserve IDs
                imageio.imwrite(base.with_name(base.stem + "_boundary.png"), bnd_edit)
                imageio.imwrite(base.with_name(base.stem + "_seed.png"), seed_edit)
        except Exception as e:
            warnings.warn(f"Export failed: {e}")

    # dock widgets
    viewer.window.add_dock_widget(open_image, area="right")
    #viewer.window.add_dock_widget(rerun_unet, area="left")
    viewer.window.add_dock_widget(resegment, area="right")
    viewer.window.add_dock_widget(export, area="right")

    # If image was passed via CLI, load it now
    if args.image is not None:
        try:
            load_and_predict(args.image)
            print('Opened via CLI')
        except Exception as e:
            warnings.warn(f"Failed to load {args.image}: {e}")

    napari.run()
    # ---------------- Keyboard shortcuts for layer navigation & visibility --------

    # Track a "solo" state to hide all-but-active (press again to restore)
    _last_vis = {}   # name -> bool
    _solo_on = {"v": False}
    


if __name__ == "__main__":
    main([
        #"--ckpt", path/to/ckpt,
        #"--image", path/to/img
        ])



# Currently broken
#@magicgui(call_button="Re-run UNet (reset edits)")
#def rerun_unet():
#    if cur_image is None:
#        return
#    fg, bnd, seed = run_unet_on(cur_image)
#    # reset prob layers
#   # clear shapes
#    if "Draw lines" in viewer.layers:
#        viewer.layers["Draw lines"].data = []
    #viewer.layers["FG prob"].data    = fg
    #viewer.layers["Boundary edit"].data   = bnd
    #viewer.layers["Seed edit"].data  = seed
    # reset editable layers from probs
#    viewer.layers["Boundary edit"].data = (bnd > 0.25).astype(np.uint8)
#    viewer.layers["Seed edit"].data     = (seed > 0.35).astype(np.uint8)
    
    # update instances
#    labels = resegment_from_layers(fg, viewer.layers["Boundary edit"].data,
#                                   viewer.layers["Seed edit"].data, fg_thr=0.25)
#    viewer.layers["Instances"].data = labels
#    inst_layer = _get_layer(viewer, "Instances")
#    if _remove_mode.get("on") and _remove_label_at_click not in inst_layer.mouse_drag_callbacks:
#        inst_layer.mouse_drag_callbacks.append(_remove_label_at_click)

# ---------------- watershed from edited layers ----------------
#def resegment_from_layers(fg_prob: np.ndarray,
#                          bnd_edit: np.ndarray,
#                          seed_edit: np.ndarray,
#                          *, fg_thr=0.25, compactness=0.001,
#                          min_size=32) -> np.ndarray:
#    """Run watershed using edited boundary + seed layers.
#       bnd_edit, seed_edit are boolean arrays.
#    """
#    fg = fg_prob > fg_thr
    # a gentle rim around connected FG helps discourage merges
#    rim = find_boundaries(fg.astype(int), mode="outer")
    # final boundary = user-edited boundary ∪ rim
#    bnd = (bnd_edit > 0) | rim

    # allowed growth region
#    mask = fg & (~bnd)

    # synthesize or consolidate markers
#    seed = (seed_edit > 0)
#    if consolidate_markers is not None:
#        markers = consolidate_markers(seed, mask)
#    else:
        # fallback: label seeds clipped to mask
#        markers, _ = ndi.label(seed & mask)

#    dist = ndi.distance_transform_edt(mask)
#    labels = watershed(-dist, markers=markers, mask=mask, compactness=compactness)
#    labels = remove_small_objects(labels, min_size=min_size)
#    
#    return labels.astype(np.int32)
