# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "shiboken6",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.qt_compat",
    "scipy.ndimage",
    "scipy.io",
    "skimage.draw",
    "skimage.morphology",
    "imageio.v3",
    "tifffile",
]

datas = collect_data_files("matplotlib")

a = Analysis(
    ["Annotation_gui.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "napari",
        "vispy",
        "magicgui",
        "qtpy",
        "torch",
        "tensorflow",
        "PyQt5",
        "PyQt6",
        "PySide2",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="AnnotationGUI",
    console=False,
    debug=False,
)