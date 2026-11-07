# Voidcraft

A Minecraft-style voxel game in Python and OpenGL with destruction physics.
Built with pygame, PyOpenGL, numpy and scipy.

Terrain is made of small cubes.
Blast a crater and the cubes become debris.

A phone scanner turns photos of real objects into blocks you can place and blow up.
It uses `scanner.html` and `scanner_server.py` (Flask, OpenCV).

## Run

    pip install pygame numpy PyOpenGL scipy
    python engineui.py

## Scanner (optional)

    pip install flask opencv-python numpy
    python scanner_server.py

Work in progress.
