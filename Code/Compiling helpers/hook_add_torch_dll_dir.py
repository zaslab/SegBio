# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 08:29:28 2025

@author: eduar
"""

# hook_add_torch_dll_dir.py
import os, sys, site, pathlib

def add(path: pathlib.Path):
    try:
        if path.is_dir():
            os.add_dll_directory(str(path))
    except Exception:
        pass

# a) In a frozen app, Torch DLLs live inside the bundle
if getattr(sys, "frozen", False):
    exe_dir = pathlib.Path(sys.executable).parent
    # PyInstaller onedir layout
    add(exe_dir / "_internal" / "torch" / "lib")
    add(exe_dir / "torch" / "lib")
    # _MEIPASS (safety for some layouts/onefile)
    mp = getattr(sys, "_MEIPASS", None)
    if mp:
        base = pathlib.Path(mp)
        add(base / "torch" / "lib")
        add(base / "_internal" / "torch" / "lib")

# b) Also add site-packages locations when running unfrozen
for p in site.getsitepackages() + [site.getusersitepackages()]:
    add(pathlib.Path(p) / "torch" / "lib")
