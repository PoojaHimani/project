# Air Writing System (Python + OpenCV + MediaPipe)

This project allows writing in the air using hand gestures. It tracks the index fingertip using MediaPipe Hands and draws on a canvas overlay.

Features:
- Index finger: write
- Two fingers (index + middle): erase
- Stroke smoothing using a small moving average window
- Gesture stability counter to avoid accidental drawing

Requirements
```
python 3.8+
pip install -r requirements.txt
```

Run
```
python main.py
```

Controls
- 'c' : clear canvas
- 'q' : quit

Notes
- For best results, use a plain background and good lighting.
- Tweak `SMOOTHING_WINDOW` and `STABLE_THRESHOLD` in `main.py` to adjust responsiveness.
Compatibility note

- If you have older MediaPipe that exposes `mp.solutions` (classic API), this
	code will use it for accurate hand landmarks and landmark drawing.
- If you have newer MediaPipe 0.10+ (Tasks API) or no MediaPipe installed,
	the project will fall back to a lightweight OpenCV-based fingertip heuristic.

Using the Tasks API (0.10+) directly for the same landmark output requires a
MediaPipe hand-landmarker model file; that's optional and not bundled here.

If you want to use the fallback (no extra steps):

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

If you prefer to install MediaPipe (you already did):

```powershell
venv\Scripts\activate
pip install mediapipe
python main.py
```

If you'd like full MediaPipe Tasks-model landmarking (closer to the screenshot),
I can add code to load a hand-landmarker model from a local `models/` folder and
show how to download the proper `.tflite` model. Tell me if you want that
option and I will add the loader and instructions.
