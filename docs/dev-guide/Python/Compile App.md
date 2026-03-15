# Compile the inference app

## Goal
Build a standalone executable for the inference GUI using PyInstaller.

## Basic steps
1) Put in a single folder: <br>
      &nbsp;•	Worm_gui_editor.py <br>
      &nbsp;•	postproc.py <br>
      &nbsp;•	segmentor_utils.py <br>
      &nbsp;•	FlexiUnet.py <br>
      &nbsp;•	All contents of compiling helpers <br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; o	WormGUI.spec <br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; o	hook_add_torch_dll_dir <br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; o	hook_set_qt_api <br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; o	hook_vispy_meipass <br>

3) Activate WormGuiVenv

4) From the folder that contains all scripts run:

  ```bash
  "path\to\venv\Scripts\python.exe" -m PyInstaller WormGui.spec path\to\output\filename.exe 
  ```
5) Repeat for annotation GUI using Annotationvenv, io_utils.py, geometryutils.py and AnnotationGUI.spec


