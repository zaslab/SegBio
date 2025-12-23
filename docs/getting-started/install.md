# Installation
bla bla bla

## Python environment
- Create a dedicated environment (conda/venv).
- Install PyTorch (GPU if available) + project requirements.

## GUI dependencies (Napari / Qt)
Napari requires a Qt backend. If the GUI fails to launch, install/verify:
- `napari[all]`
- `pyqt5` or `pyside2` (depending on your stack)

## MATLAB (annotation + mask generation)
MATLAB is used for the **WormMarkerGUI** and scripts that generate training masks.

## Project layout expectations
The pipeline expects raw images in a folder, MATLAB outputs saved alongside / in a known location,
and training data generated into a root folder with UUID subfolders.
