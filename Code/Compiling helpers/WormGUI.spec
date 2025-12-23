# -*- mode: python ; coding: utf-8 -*-

# wormgui.spec
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs, collect_all

hidden = []
hidden += collect_submodules('qtpy')
hidden += collect_submodules('magicgui')
hidden += collect_submodules('napari')
hidden += collect_submodules('skimage')
hidden += collect_submodules('imageio')
hidden += collect_submodules('cv2')

datas = []
datas += collect_data_files('PyQt5', include_py_files=True)   # Qt plugins
datas += collect_data_files('napari',  include_py_files=True)   # napari icons/themes

vispy_datas, vispy_bins, vispy_hidden = collect_all('vispy')
datas    += vispy_datas

binaries = []
binaries += vispy_bins
hidden   += vispy_hidden
binaries += collect_dynamic_libs('torch')                       # torch\lib\*.dll

#a = Analysis(
#    ['worm_editor_gui.py'],
#    pathex=[],
#    binaries=binaries,
#    datas=datas,
#    hiddenimports=hidden,
#    hookspath=[],
#    runtime_hooks=['hook_set_qt_api.py','hook_add_torch_dll_dir.py'],
#    noarchive=True,   # <— add this
#    optimize=0,
#    excludes=['tensorflow','tensorflow_intel','tensorboard','nvidia','tensorrt','cudnn',  #'PySide6','PySide6.QtCore','PySide6.QtGui','PySide6.QtWidgets','shiboken6',
#        'PySide2','shiboken2']
#)

#exe = EXE(a.pure, a.scripts, a.binaries, a.zipfiles, a.datas,
#          name='WormGUIApp', console=False)

#coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name='WormGui')

a = Analysis(
	['worm_editor_gui.py'],
	pathex=[],
	binaries=binaries,
	datas=datas,
	hiddenimports=hidden,
	hookspath=[],
	runtime_hooks=['hook_set_qt_api.py','hook_add_torch_dll_dir.py','hook_vispy_meipass.py'],
	excludes=['tensorflow','tensorflow_intel','tensorboard','nvidia',
	'tensorrt','cudnn','PySide6','PySide6.QtCore','PySide6.QtGui',
	'PySide6.QtWidgets','shiboken6','PySide2','shiboken2'],
	noarchive=False,          # <-- important
	optimize=0
)

pyz = PYZ(a.pure)             # <-- create PYZ

exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, name='WormGUIApp', console=True, onefile=True, debug=True)