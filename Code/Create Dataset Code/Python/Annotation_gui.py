import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional
import matplotlib
matplotlib.use("QtAgg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib import colormaps
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage as ndi
from skimage.draw import polygon
from skimage.morphology import binary_closing, disk
from geometry_utils import (euclidean_length, pair_distance, resample_centerline,
                            default_width_profile_rel, symmetric_width_profile_rel, centerline_to_polygon,
                            polygon_to_mask, postprocess_mask, generate_preview_mask)
from io_utils import ensure_grayscale_image, load_image_any
try:
    import tifffile
except Exception:
    tifffile = None

try:
    import imageio.v3 as iio
except Exception:
    iio = None

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None


@dataclass
class WormAnnotation:
    worm_id: int
    centerline_points: List[List[float]]
    width_points: List[List[float]]
    head_point: Optional[List[float]]
    centerline_length_px: float
    max_width_px: float
    image_name: str
    notes: str = ""
    width_multiplier: float = 1.0

    def to_json_dict(self):
        return asdict(self)


class ImageCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 8), constrained_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        self.image_artist = None
        self.annotations: List[dict] = []
        self.selected_index: Optional[int] = None
        self.current_centerline: List[List[float]] = []
        self.current_width: List[List[float]] = []
        self.centerline_preview_line: Optional[Line2D] = None
        self.centerline_preview_scatter: Optional[PathCollection] = None
        self.width_preview_line: Optional[Line2D] = None
        self.width_preview_scatter: Optional[PathCollection] = None
        self.mode = "idle"
        self.dragging = None
        self.image = None
        self.image_path = None
        self.cmap_name = "gray"
        self.vertices_transparent = False
        self.vertex_size_selected = 20
        self.vertex_size_normal = 14
        self.vertex_size_current = 16
        self.head_size = 48
        self.vertex_alpha_visible = 1.0
        self.vertex_alpha_faded = 0.28
        self.pan_active = False
        self.pan_start = None
        self.pan_xlim = None
        self.pan_ylim = None
        self.mask_preview = None
        self.mask_visible = True

        self._cid_press = self.mpl_connect("button_press_event", self.on_press)
        self._cid_release = self.mpl_connect("button_release_event", self.on_release)
        self._cid_motion = self.mpl_connect("motion_notify_event", self.on_motion)
        self._cid_key = self.mpl_connect("key_press_event", self.on_key)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor("black")
        self.on_state_change = None

    def set_state_callback(self, fn):
        self.on_state_change = fn

    def emit_state(self):
        if callable(self.on_state_change):
            self.on_state_change()

    def load_image(self, image: np.ndarray, image_path: str):
        self.image = np.asarray(image)
        self.image_path = image_path
        self.annotations = []
        self.selected_index = None
        self.current_centerline = []
        self.current_width = []
        self.mask_preview = None
        self.mode = "idle"
        self.dragging = None
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor("black")
        self.image_artist = self.ax.imshow(self.image, cmap=self.cmap_name, origin="upper")
        self.ax.set_title(os.path.basename(image_path), color="white")
        self.redraw_all()
        self.emit_state()

    def cycle_colormap(self):
        maps = ["gray", "viridis", "magma", "inferno", "cividis", "plasma", "bone"]
        idx = maps.index(self.cmap_name) if self.cmap_name in maps else 0
        self.cmap_name = maps[(idx + 1) % len(maps)]
        if self.image_artist is not None:
            self.image_artist.set_cmap(colormaps[self.cmap_name])
        self.draw_idle()
        self.emit_state()

    def toggle_vertices(self):
        self.vertices_transparent = not self.vertices_transparent
        self.redraw_all()
        self.emit_state()

    def set_selected_width_multiplier(self, value: float):
        if self.selected_index is None or self.selected_index < 0 or self.selected_index >= len(self.annotations):
            return
        self.annotations[self.selected_index]["width_multiplier"] = float(value)
        if self.mask_preview is not None:
            self.update_mask_preview()
        self.emit_state()

    def toggle_mask_visibility(self):
        self.mask_visible = not self.mask_visible
        self.redraw_all()
        self.emit_state()

    def update_mask_preview(self):
        if self.image is None:
            self.mask_preview = None
            self.redraw_all()
            return
        h, w = self.image.shape[:2]
        preview = np.zeros((h, w), dtype=bool)
        for ann in self.annotations:
            try:
                preview |= generate_preview_mask(ann, (h, w), width_multiplier=None)
            except Exception:
                pass
        self.mask_preview = preview
        self.redraw_all()

    def zoom(self, factor: float, x: Optional[float] = None, y: Optional[float] = None):
        if self.image is None:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        cx = (xlim[0] + xlim[1]) / 2 if x is None else x
        cy = (ylim[0] + ylim[1]) / 2 if y is None else y
        width = abs(xlim[1] - xlim[0]) * factor
        height = abs(ylim[1] - ylim[0]) * factor
        self.ax.set_xlim(cx - width / 2, cx + width / 2)
        if ylim[0] > ylim[1]:
            self.ax.set_ylim(cy + height / 2, cy - height / 2)
        else:
            self.ax.set_ylim(cy - height / 2, cy + height / 2)
        self.draw_idle()

    def reset_view(self):
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        self.ax.set_xlim(-0.5, w - 0.5)
        self.ax.set_ylim(h - 0.5, -0.5)
        self.draw_idle()

    def start_new_worm(self):
        self.current_centerline = []
        self.current_width = []
        self.selected_index = None
        self.mode = "centerline"
        self.emit_state()

    def finish_centerline(self):
        if len(self.current_centerline) < 2:
            raise ValueError("Centerline must contain at least 2 points.")
        self.mode = "width"
        self.emit_state()

    def finish_width(self):
        if len(self.current_width) != 2:
            raise ValueError("Width must contain exactly 2 points.")
        worm_id = len(self.annotations) + 1
        image_name = os.path.basename(self.image_path) if self.image_path else ""
        ann = {
            "worm_id": worm_id,
            "centerline_points": [p[:] for p in self.current_centerline],
            "width_points": [p[:] for p in self.current_width],
            "head_point": None,
            "centerline_length_px": euclidean_length(self.current_centerline),
            "max_width_px": pair_distance(self.current_width),
            "image_name": image_name,
            "notes": "",
            "width_multiplier": 1.0,
        }
        self.annotations.append(ann)
        self.selected_index = len(self.annotations) - 1
        self.current_centerline = []
        self.current_width = []
        self.mode = "edit"
        self.redraw_all()
        self.emit_state()

    def start_mark_head(self):
        if self.selected_index is None:
            raise ValueError("Select a worm first.")
        self.mode = "head"
        self.emit_state()

    def set_head_for_selected(self, x: float, y: float):
        if self.selected_index is None:
            raise ValueError("Select a worm first.")
        self.annotations[self.selected_index]["head_point"] = [float(x), float(y)]
        self.mode = "edit"
        if self.mask_preview is not None:
            self.update_mask_preview()
        self.redraw_all()
        self.emit_state()

    def select_annotation(self, index: Optional[int]):
        if index is None or index < 0 or index >= len(self.annotations):
            self.selected_index = None
        else:
            self.selected_index = index
        self.redraw_all()
        self.emit_state()

    def delete_selected(self):
        if self.selected_index is None:
            return
        del self.annotations[self.selected_index]
        for i, ann in enumerate(self.annotations, start=1):
            ann["worm_id"] = i
        if len(self.annotations) == 0:
            self.selected_index = None
        else:
            self.selected_index = min(self.selected_index, len(self.annotations) - 1)
        if self.mask_preview is not None:
            self.update_mask_preview()
        self.redraw_all()
        self.emit_state()

    def undo_last(self):
        if self.mode == "centerline" and len(self.current_centerline) > 0:
            self.current_centerline.pop()
        elif self.mode == "width" and len(self.current_width) > 0:
            self.current_width.pop()
        elif self.mode == "head" and self.selected_index is not None:
            self.annotations[self.selected_index]["head_point"] = None
            self.mode = "edit"
        elif self.annotations:
            self.annotations.pop()
            if self.selected_index is not None:
                self.selected_index = min(self.selected_index, len(self.annotations) - 1) if self.annotations else None
        if self.mask_preview is not None:
            self.update_mask_preview()
        self.redraw_all()
        self.emit_state()

    def clear_current(self):
        if self.mode == "centerline":
            self.current_centerline = []
        elif self.mode == "width":
            self.current_width = []
        elif self.mode == "head" and self.selected_index is not None:
            self.annotations[self.selected_index]["head_point"] = None
            self.mode = "edit"
        if self.mask_preview is not None:
            self.update_mask_preview()
        self.redraw_all()
        self.emit_state()

    def update_selected_notes(self, text: str):
        if self.selected_index is not None:
            self.annotations[self.selected_index]["notes"] = text
            self.emit_state()

    def recompute_selected_metrics(self):
        if self.selected_index is None:
            return
        ann = self.annotations[self.selected_index]
        ann["centerline_length_px"] = euclidean_length(ann["centerline_points"])
        ann["max_width_px"] = pair_distance(ann["width_points"])
        self.emit_state()

    def get_export_objects(self) -> List[WormAnnotation]:
        return [WormAnnotation(**ann) for ann in self.annotations]

    def find_hit(self, x: float, y: float, radius: float = 8.0):
        if x is None or y is None:
            return None
        candidates = []
        for ai, ann in enumerate(self.annotations):
            for pi, pt in enumerate(ann["centerline_points"]):
                d = math.hypot(pt[0] - x, pt[1] - y)
                candidates.append((d, ai, "centerline", pi))
            for pi, pt in enumerate(ann["width_points"]):
                d = math.hypot(pt[0] - x, pt[1] - y)
                candidates.append((d, ai, "width", pi))
            hp = ann.get("head_point", None)
            if hp is not None:
                d = math.hypot(hp[0] - x, hp[1] - y)
                candidates.append((d, ai, "head", 0))
        if not candidates:
            return None
        d, ai, kind, pi = min(candidates, key=lambda t: t[0])
        if d <= radius:
            return ai, kind, pi
        return None

    def on_press(self, event):
        self.setFocus()
        if event.inaxes != self.ax or self.image is None:
            return

        if event.button == MouseButton.MIDDLE:
            self.pan_active = True
            self.pan_start = (event.xdata, event.ydata)
            self.pan_xlim = self.ax.get_xlim()
            self.pan_ylim = self.ax.get_ylim()
            return

        if event.button == 1:
            x, y = float(event.xdata), float(event.ydata)
            if self.mode == "centerline":
                self.current_centerline.append([x, y])
                self.redraw_all()
                self.emit_state()
                return
            if self.mode == "width":
                if len(self.current_width) < 2:
                    self.current_width.append([x, y])
                else:
                    self.current_width[1] = [x, y]
                self.redraw_all()
                self.emit_state()
                return
            if self.mode == "head":
                self.set_head_for_selected(x, y)
                return

            hit = self.find_hit(x, y)
            if hit is not None:
                ai, kind, pi = hit
                self.selected_index = ai
                self.dragging = (ai, kind, pi)
                self.redraw_all()
                self.emit_state()
                return

            nearest = self.find_hit(x, y, radius=25.0)
            if nearest is not None:
                self.selected_index = nearest[0]
                self.redraw_all()
                self.emit_state()

        elif event.button == 3:
            if self.mode == "centerline" and self.current_centerline:
                self.current_centerline.pop()
            elif self.mode == "width" and self.current_width:
                self.current_width.pop()
            elif self.mode == "head" and self.selected_index is not None:
                self.annotations[self.selected_index]["head_point"] = None
            if self.mask_preview is not None:
                self.update_mask_preview()
            self.redraw_all()
            self.emit_state()

    def on_motion(self, event):
        if event.inaxes != self.ax or self.image is None:
            return
        if self.pan_active and self.pan_start is not None:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            dx = x - self.pan_start[0]
            dy = y - self.pan_start[1]
            self.ax.set_xlim(self.pan_xlim[0] - dx, self.pan_xlim[1] - dx)
            self.ax.set_ylim(self.pan_ylim[0] - dy, self.pan_ylim[1] - dy)
            self.draw_idle()
            return
        if self.dragging is None:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        ai, kind, pi = self.dragging
        if kind == "head":
            self.annotations[ai]["head_point"] = [float(x), float(y)]
        else:
            self.annotations[ai][f"{kind}_points"][pi] = [float(x), float(y)]
            self.recompute_selected_metrics()
        if self.mask_preview is not None:
            self.update_mask_preview()
        self.redraw_all()

    def on_release(self, event):
        self.dragging = None
        self.pan_active = False
        self.pan_start = None
        self.pan_xlim = None
        self.pan_ylim = None

    def on_key(self, event):
        self.setFocus()
        if event.key in ["enter", "return", "f"]:
            try:
                if self.mode == "centerline":
                    self.finish_centerline()
                elif self.mode == "width":
                    self.finish_width()
                    if self.mask_preview is not None:
                        self.update_mask_preview()
            except ValueError:
                pass
        elif event.key == "h":
            try:
                self.start_mark_head()
            except ValueError:
                pass
        elif event.key == "escape":
            self.clear_current()
        elif event.key == "backspace":
            self.undo_last()
        elif event.key == "n":
            self.start_new_worm()
        elif event.key in ["+", "="]:
            self.zoom(0.8)
        elif event.key in ["-", "_"]:
            self.zoom(1.25)
        elif event.key == "0":
            self.reset_view()

    def redraw_all(self):
        if self.image is None:
            self.draw_idle()
            return
        old_xlim = self.ax.get_xlim()
        old_ylim = self.ax.get_ylim()
        preserve_view = len(self.ax.images) > 0
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor("black")
        self.image_artist = self.ax.imshow(self.image, cmap=self.cmap_name, origin="upper")
        self.ax.set_title(os.path.basename(self.image_path) if self.image_path else "", color="white")

        if self.mask_visible and self.mask_preview is not None and np.any(self.mask_preview):
            overlay = np.zeros((*self.mask_preview.shape, 4), dtype=float)
            overlay[..., 1] = 1.0
            overlay[..., 3] = self.mask_preview.astype(float) * 0.28
            self.ax.imshow(overlay, origin="upper")

        for i, ann in enumerate(self.annotations):
            selected = i == self.selected_index
            line_color = "yellow" if selected else "red"
            width_color = "cyan" if selected else "deepskyblue"
            head_color = "magenta" if selected else "hotpink"
            point_edge = "black"
            point_size = self.vertex_size_selected if selected else self.vertex_size_normal
            point_alpha = self.vertex_alpha_faded if self.vertices_transparent else self.vertex_alpha_visible
            line_alpha = self.vertex_alpha_faded if self.vertices_transparent else self.vertex_alpha_visible
            lw = 2.8 if selected else 2.0

            cpts = np.asarray(ann["centerline_points"], dtype=float)
            if len(cpts) > 0:
                self.ax.plot(cpts[:, 0], cpts[:, 1], color=line_color, linewidth=lw, alpha=line_alpha)
                self.ax.scatter(cpts[:, 0], cpts[:, 1], s=point_size, color=line_color, edgecolors=point_edge, linewidths=0.5, alpha=point_alpha, zorder=3)
                label_x, label_y = cpts[0, 0], cpts[0, 1]
                self.ax.text(label_x, label_y, str(ann["worm_id"]), color="white", fontsize=10, weight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, edgecolor="none"))

            wpts = np.asarray(ann["width_points"], dtype=float)
            if len(wpts) == 2:
                self.ax.plot(wpts[:, 0], wpts[:, 1], color=width_color, linewidth=lw, alpha=line_alpha)
                self.ax.scatter(wpts[:, 0], wpts[:, 1], s=point_size, color=width_color, edgecolors=point_edge, linewidths=0.5, alpha=point_alpha, zorder=3)

            hp = ann.get("head_point", None)
            if hp is not None:
                self.ax.scatter([hp[0]], [hp[1]], s=self.head_size, color=head_color, edgecolors="white", linewidths=1.0, alpha=point_alpha, marker="o", zorder=5)
                self.ax.text(hp[0] + 4, hp[1] - 4, "H", color="white", fontsize=9, weight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor=head_color, alpha=0.6, edgecolor="none"), zorder=6)

        if self.current_centerline:
            cpts = np.asarray(self.current_centerline, dtype=float)
            current_alpha = self.vertex_alpha_faded if self.vertices_transparent else self.vertex_alpha_visible
            self.ax.plot(cpts[:, 0], cpts[:, 1], color="lime", linewidth=2.2, linestyle="-", alpha=current_alpha)
            self.ax.scatter(cpts[:, 0], cpts[:, 1], s=self.vertex_size_current, color="lime", edgecolors="black", linewidths=0.5, alpha=current_alpha, zorder=4)

        if len(self.current_width) >= 1:
            wpts = np.asarray(self.current_width, dtype=float)
            current_alpha = self.vertex_alpha_faded if self.vertices_transparent else self.vertex_alpha_visible
            if len(wpts) == 1:
                self.ax.scatter(wpts[:, 0], wpts[:, 1], s=self.vertex_size_current, color="orange", edgecolors="black", linewidths=0.5, alpha=current_alpha, zorder=4)
            elif len(wpts) == 2:
                self.ax.plot(wpts[:, 0], wpts[:, 1], color="orange", linewidth=2.2, alpha=current_alpha)
                self.ax.scatter(wpts[:, 0], wpts[:, 1], s=self.vertex_size_current, color="orange", edgecolors="black", linewidths=0.5, alpha=current_alpha, zorder=4)

        if preserve_view:
            self.ax.set_xlim(old_xlim)
            self.ax.set_ylim(old_ylim)
        self.draw_idle()


class WormMarkerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Worm Marker App")
        self.resize(1400, 900)

        self.canvas = ImageCanvas(self)
        self.canvas.set_state_callback(self.refresh_ui_from_state)

        self.load_btn = QPushButton("Load Image")
        self.new_worm_btn = QPushButton("New Worm")
        self.finish_centerline_btn = QPushButton("Finish Centerline")
        self.finish_width_btn = QPushButton("Finish Width")
        self.mark_head_btn = QPushButton("Mark Head")
        self.undo_btn = QPushButton("Undo")
        self.delete_btn = QPushButton("Delete Selected")
        self.cmap_btn = QPushButton("Toggle Colormap")
        self.toggle_vertices_btn = QPushButton("Fade Vertices")
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.reset_view_btn = QPushButton("Reset View")
        self.generate_mask_btn = QPushButton("Generate Masks")
        self.toggle_masks_btn = QPushButton("Hide Masks")
        self.save_json_btn = QPushButton("Save JSON")
        self.save_csv_btn = QPushButton("Save CSV")
        self.save_both_btn = QPushButton("Save All")

        self.status_label = QLabel("Load an image to begin.")
        self.status_label.setWordWrap(True)
        self.mode_label = QLabel("Mode: idle")
        self.file_label = QLabel("Image: none")
        self.cmap_label = QLabel("Colormap: gray")

        self.annotation_list = QListWidget()
        self.annotation_list.setSelectionMode(QListWidget.SingleSelection)

        self.notes_edit = QLineEdit()
        self.length_label = QLabel("Length: -")
        self.width_label = QLabel("Width: -")
        self.head_label = QLabel("Head: -")
        self.width_mult_label = QLabel("Selected worm width multiplier: 1.00")
        self.width_mult_slider = QSlider(Qt.Horizontal)
        self.width_mult_slider.setMinimum(50)
        self.width_mult_slider.setMaximum(300)
        self.width_mult_slider.setValue(100)
        self.width_mult_slider.setSingleStep(5)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.new_worm_btn)
        control_layout.addWidget(self.finish_centerline_btn)
        control_layout.addWidget(self.finish_width_btn)
        control_layout.addWidget(self.mark_head_btn)
        control_layout.addWidget(self.undo_btn)
        control_layout.addWidget(self.delete_btn)
        control_layout.addWidget(self.cmap_btn)
        control_layout.addWidget(self.toggle_vertices_btn)
        control_layout.addWidget(self.zoom_in_btn)
        control_layout.addWidget(self.zoom_out_btn)
        control_layout.addWidget(self.reset_view_btn)
        control_layout.addWidget(self.generate_mask_btn)
        control_layout.addWidget(self.toggle_masks_btn)
        control_layout.addWidget(self.width_mult_label)
        control_layout.addWidget(self.width_mult_slider)
        control_layout.addSpacing(12)
        control_layout.addWidget(QLabel("Annotations"))
        control_layout.addWidget(self.annotation_list, stretch=1)

        form = QFormLayout()
        form.addRow("Notes", self.notes_edit)
        form.addRow("Centerline", self.length_label)
        form.addRow("Max width", self.width_label)
        form.addRow("Head", self.head_label)
        control_layout.addLayout(form)

        control_layout.addSpacing(12)
        control_layout.addWidget(self.save_json_btn)
        control_layout.addWidget(self.save_csv_btn)
        control_layout.addWidget(self.save_both_btn)
        control_layout.addSpacing(12)
        control_layout.addWidget(self.mode_label)
        control_layout.addWidget(self.cmap_label)
        control_layout.addWidget(self.file_label)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch(0)

        splitter = QSplitter()
        splitter.addWidget(self.canvas)
        splitter.addWidget(control_widget)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        self.load_btn.clicked.connect(self.load_image)
        self.new_worm_btn.clicked.connect(self.start_new_worm)
        self.finish_centerline_btn.clicked.connect(self.finish_centerline)
        self.finish_width_btn.clicked.connect(self.finish_width)
        self.mark_head_btn.clicked.connect(self.start_mark_head)
        self.undo_btn.clicked.connect(self.canvas.undo_last)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.cmap_btn.clicked.connect(self.toggle_cmap)
        self.toggle_vertices_btn.clicked.connect(self.toggle_vertices)
        self.zoom_in_btn.clicked.connect(lambda: self.canvas.zoom(0.8))
        self.zoom_out_btn.clicked.connect(lambda: self.canvas.zoom(1.25))
        self.reset_view_btn.clicked.connect(self.canvas.reset_view)
        self.generate_mask_btn.clicked.connect(self.generate_masks)
        self.toggle_masks_btn.clicked.connect(self.toggle_masks)
        self.width_mult_slider.valueChanged.connect(self.on_width_multiplier_changed)
        self.save_json_btn.clicked.connect(self.save_json)
        self.save_csv_btn.clicked.connect(self.save_csv)
        self.save_both_btn.clicked.connect(self.save_all)
        self.annotation_list.currentRowChanged.connect(self.on_list_selection_changed)
        self.notes_edit.editingFinished.connect(self.save_notes_from_box)

        self.refresh_ui_from_state()

    def show_error(self, message: str):
        QMessageBox.critical(self, "Error", message)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.mat);;All files (*)")
        if not path:
            return
        try:
            image = load_image_any(path)
            self.canvas.load_image(image, path)
            self.canvas.setFocus()
            self.status_label.setText("Image loaded. Click 'New Worm' to start placing centerline points.")
        except Exception as e:
            self.show_error(str(e))

    def start_new_worm(self):
        if self.canvas.image is None:
            self.show_error("Load an image first.")
            return
        self.canvas.start_new_worm()
        self.canvas.setFocus()
        self.status_label.setText("Centerline mode: left-click to add ordered vertices. Right-click removes the last current point. Press Enter or click 'Finish Centerline' when done.")

    def finish_centerline(self):
        try:
            self.canvas.finish_centerline()
            self.canvas.setFocus()
            self.status_label.setText("Width mode: click exactly 2 points for the maximum width line. Press Enter or click 'Finish Width' when done.")
        except Exception as e:
            self.show_error(str(e))

    def finish_width(self):
        try:
            self.canvas.finish_width()
            if self.canvas.mask_preview is not None:
                self.canvas.update_mask_preview()
            self.canvas.setFocus()
            self.status_label.setText("Worm added. You can now drag vertices to edit, select worms in the list, mark a head, or start a new worm.")
        except Exception as e:
            self.show_error(str(e))

    def start_mark_head(self):
        if self.canvas.image is None:
            self.show_error("Load an image first.")
            return
        if self.canvas.selected_index is None:
            self.show_error("Select a worm first.")
            return
        try:
            self.canvas.start_mark_head()
            self.canvas.setFocus()
            self.status_label.setText("Head mode: left-click once to place the head point for the selected worm. Press H as a shortcut.")
        except Exception as e:
            self.show_error(str(e))

    def generate_masks(self):
        if self.canvas.image is None or not self.canvas.annotations:
            self.show_error("Load an image and mark at least one worm first.")
            return
        self.canvas.update_mask_preview()
        self.status_label.setText("Preview masks generated for current worms.")

    def on_width_multiplier_changed(self, value: int):
        mult = value / 100.0
        self.width_mult_label.setText(f"Selected worm width multiplier: {mult:.2f}")
        self.canvas.set_selected_width_multiplier(mult)
        if self.canvas.mask_preview is not None:
            self.status_label.setText("Updated mask preview.")
        self.refresh_ui_from_state()

    def toggle_masks(self):
        self.canvas.toggle_mask_visibility()
        self.toggle_masks_btn.setText("Hide Masks" if self.canvas.mask_visible else "Show Masks")

    def delete_selected(self):
        self.canvas.delete_selected()
        self.status_label.setText("Selected worm deleted.")

    def toggle_cmap(self):
        self.canvas.cycle_colormap()
        self.status_label.setText("Display colormap changed.")

    def toggle_vertices(self):
        self.canvas.toggle_vertices()
        state = "faded" if self.canvas.vertices_transparent else "fully visible"
        self.status_label.setText(f"Vertex markers are now {state}.")

    def on_list_selection_changed(self, row: int):
        self.canvas.select_annotation(None if row < 0 else row)
        self.refresh_ui_from_state()

    def save_notes_from_box(self):
        self.canvas.update_selected_notes(self.notes_edit.text())
        self.refresh_ui_from_state()

    def default_base_path(self) -> str:
        if self.canvas.image_path:
            base, _ = os.path.splitext(self.canvas.image_path)
            return base
        return os.path.join(os.getcwd(), "annotations")

    def save_json(self):
        anns = self.canvas.get_export_objects()
        if not anns:
            self.show_error("There are no annotations to save.")
            return
        default_path = self.default_base_path() + "_annotations.json"
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", default_path, "JSON (*.json)")
        if not path:
            return
        payload = {
            "image_path": self.canvas.image_path,
            "image_shape": list(self.canvas.image.shape) if self.canvas.image is not None else None,
            "annotations": [a.to_json_dict() for a in anns],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.status_label.setText(f"Saved JSON to {path}")

    def save_csv(self):
        anns = self.canvas.get_export_objects()
        if not anns:
            self.show_error("There are no annotations to save.")
            return
        default_path = self.default_base_path() + "_annotations.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", default_path, "CSV (*.csv)")
        if not path:
            return
        rows = []
        for a in anns:
            rows.append({
                "worm_id": a.worm_id,
                "image_name": a.image_name,
                "centerline_length_px": a.centerline_length_px,
                "max_width_px": a.max_width_px,
                "n_centerline_points": len(a.centerline_points),
                "width_x1": a.width_points[0][0] if len(a.width_points) > 0 else np.nan,
                "width_y1": a.width_points[0][1] if len(a.width_points) > 0 else np.nan,
                "width_x2": a.width_points[1][0] if len(a.width_points) > 1 else np.nan,
                "width_y2": a.width_points[1][1] if len(a.width_points) > 1 else np.nan,
                "head_x": a.head_point[0] if a.head_point is not None else np.nan,
                "head_y": a.head_point[1] if a.head_point is not None else np.nan,
                "notes": a.notes,
            })
        pd.DataFrame(rows).to_csv(path, index=False)
        self.status_label.setText(f"Saved CSV to {path}")

    def save_all(self):
        anns = self.canvas.get_export_objects()
        if not anns:
            self.show_error("There are no annotations to save.")
            return
        base_default = self.default_base_path()
        json_path, _ = QFileDialog.getSaveFileName(self, "Save annotation JSON", base_default + "_annotations.json", "JSON (*.json)")
        if not json_path:
            return
        csv_path = os.path.splitext(json_path)[0] + ".csv"
        payload = {
            "image_path": self.canvas.image_path,
            "image_shape": list(self.canvas.image.shape) if self.canvas.image is not None else None,
            "annotations": [a.to_json_dict() for a in anns],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        rows = []
        for a in anns:
            rows.append({
                "worm_id": a.worm_id,
                "image_name": a.image_name,
                "centerline_length_px": a.centerline_length_px,
                "max_width_px": a.max_width_px,
                "n_centerline_points": len(a.centerline_points),
                "width_x1": a.width_points[0][0] if len(a.width_points) > 0 else np.nan,
                "width_y1": a.width_points[0][1] if len(a.width_points) > 0 else np.nan,
                "width_x2": a.width_points[1][0] if len(a.width_points) > 1 else np.nan,
                "width_y2": a.width_points[1][1] if len(a.width_points) > 1 else np.nan,
                "head_x": a.head_point[0] if a.head_point is not None else np.nan,
                "head_y": a.head_point[1] if a.head_point is not None else np.nan,
                "notes": a.notes,
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        self.status_label.setText(f"Saved JSON and CSV to {os.path.dirname(json_path)}")

    def refresh_ui_from_state(self):
        self.mode_label.setText(f"Mode: {self.canvas.mode}")
        self.cmap_label.setText(f"Colormap: {self.canvas.cmap_name}")
        image_name = os.path.basename(self.canvas.image_path) if self.canvas.image_path else "none"
        self.file_label.setText(f"Image: {image_name}")
        selected_mult = 1.0
        if self.canvas.selected_index is not None and 0 <= self.canvas.selected_index < len(self.canvas.annotations):
            selected_mult = float(self.canvas.annotations[self.canvas.selected_index].get("width_multiplier", 1.0))
        self.width_mult_label.setText(f"Selected worm width multiplier: {selected_mult:.2f}")
        self.toggle_masks_btn.setText("Hide Masks" if self.canvas.mask_visible else "Show Masks")

        self.annotation_list.blockSignals(True)
        self.annotation_list.clear()
        for ann in self.canvas.annotations:
            head_txt = "H=yes" if ann.get("head_point") is not None else "H=no"
            item = QListWidgetItem(f"Worm {ann['worm_id']} | L={ann['centerline_length_px']:.1f}px | W={ann['max_width_px']:.1f}px | {head_txt}")
            self.annotation_list.addItem(item)
        if self.canvas.selected_index is not None and 0 <= self.canvas.selected_index < self.annotation_list.count():
            self.annotation_list.setCurrentRow(self.canvas.selected_index)
        self.annotation_list.blockSignals(False)

        if self.canvas.selected_index is not None and 0 <= self.canvas.selected_index < len(self.canvas.annotations):
            ann = self.canvas.annotations[self.canvas.selected_index]
            self.notes_edit.setText(ann.get("notes", ""))
            self.length_label.setText(f"{ann['centerline_length_px']:.2f} px")
            self.width_label.setText(f"{ann['max_width_px']:.2f} px")
            hp = ann.get("head_point")
            self.head_label.setText(f"({hp[0]:.1f}, {hp[1]:.1f})" if hp is not None else "-")
            self.width_mult_slider.blockSignals(True)
            self.width_mult_slider.setValue(int(round(100 * float(ann.get("width_multiplier", 1.0)))))
            self.width_mult_slider.blockSignals(False)
        else:
            self.notes_edit.setText("")
            self.length_label.setText("-")
            self.width_label.setText("-")
            self.head_label.setText("-")
            self.width_mult_slider.blockSignals(True)
            self.width_mult_slider.setValue(100)
            self.width_mult_slider.blockSignals(False)

        has_image = self.canvas.image is not None
        has_annotations = len(self.canvas.annotations) > 0
        self.new_worm_btn.setEnabled(has_image)
        self.finish_centerline_btn.setEnabled(has_image and self.canvas.mode == "centerline")
        self.finish_width_btn.setEnabled(has_image and self.canvas.mode == "width")
        self.mark_head_btn.setEnabled(has_image and self.canvas.selected_index is not None)
        self.undo_btn.setEnabled(has_image)
        self.delete_btn.setEnabled(self.canvas.selected_index is not None)
        self.cmap_btn.setEnabled(has_image)
        self.toggle_vertices_btn.setEnabled(has_image)
        self.zoom_in_btn.setEnabled(has_image)
        self.zoom_out_btn.setEnabled(has_image)
        self.reset_view_btn.setEnabled(has_image)
        self.generate_mask_btn.setEnabled(has_image and has_annotations)
        self.width_mult_slider.setEnabled(has_image and self.canvas.selected_index is not None)
        self.toggle_masks_btn.setEnabled(has_image and self.canvas.mask_preview is not None)
        self.save_json_btn.setEnabled(has_annotations)
        self.save_csv_btn.setEnabled(has_annotations)
        self.save_both_btn.setEnabled(has_annotations)
        self.notes_edit.setEnabled(self.canvas.selected_index is not None)


_qt_window_ref = None

def main():
    global _qt_window_ref

    if _qt_window_ref is not None:
        try:
            _qt_window_ref.close()
        except Exception:
            pass
        _qt_window_ref = None

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        win = WormMarkerApp()
        _qt_window_ref = win
        win.show()
        app.exec()
        return win
    else:
        win = WormMarkerApp()
        _qt_window_ref = win
        win.show()
        win.raise_()
        win.activateWindow()
        return win


if __name__ == "__main__":
    main()