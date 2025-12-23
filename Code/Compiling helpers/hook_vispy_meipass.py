# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 15:32:26 2025

@author: eduar
"""

# hook_vispy_meipass.py
import os, sys
# ensure vispy uses the bundled data when frozen
if hasattr(sys, "_MEIPASS"):
    base = os.path.join(sys._MEIPASS, "vispy")
    os.environ.setdefault("VISPY_DATA_PATH", os.path.join(base, "resources"))
    # PyQt5 backend to match your build
    os.environ.setdefault("VISPY_APP", "PyQt5")
    os.environ.setdefault("QT_API", "pyqt5")
