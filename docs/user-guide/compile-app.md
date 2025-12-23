# Compile the inference app

## Goal
Build a standalone executable for the inference GUI using PyInstaller.

## Basic steps
1) Install PyInstaller in the same environment:

```bash
pip install pyinstaller
```

2) Run PyInstaller on the GUI entry script (example):

```bash
pyinstaller --noconfirm --onefile worm_editor_gui.py
```

## Notes
- Napari + Qt often requires extra hidden imports or collecting Qt plugins.
- Test the built app on a clean machine / clean environment.
