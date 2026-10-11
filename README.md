# Make Me Disappear

Hold a peace sign up to your webcam and you vanish from the video. The room behind you
stays visible, even after you lower your hand. Hold the peace sign up again to come back.

Everything runs locally and in real time (~30 FPS on a laptop CPU). It uses OpenCV and
MediaPipe and makes no cloud calls.

## Quick start

Python 3.11+ is required.

```bash
git clone <this repo> && cd make-me-disappear
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

On first run the app downloads two small MediaPipe models (~8 MB) into `models/`.
On macOS, allow camera access for your terminal when prompted.

### How to do the trick

1. Start the app. **Step out of the frame for a second or two**, or at least move
   around, so it can learn what the room behind you looks like. The HUD line
   `background behind you: NN%` shows how much of the area you cover has been seen.
2. Come back and raise a **peace sign** (index and middle fingers up, ring and pinky curled) toward the camera.
3. Hold it **steady for 1 second**. A ring fills around your hand, then you fade out.
4. Lower your hand. You stay invisible and can move around.
5. Raise a peace sign and hold it steady for 1 second again to fade back in. Walking out
   of the frame for 1.5 s also resets you to normal.

Only one hand is tracked at a time. If both hands are up, the app follows one of them.

### Controls

| Key | Action |
|-----|--------|
| `q` / `Esc` | quit |
| `m` | switch between **latch** mode (default: the peace sign toggles; you stay invisible after dropping your hand) and **hold** mode (invisible only while the peace sign is up) |
| `b` | forget the background and re-learn it (after moving the camera or changing the lighting) |
| `h` | show/hide hand landmarks |
| `d` | debug insets: learned background plate + person mask |
| `c` | clean view: no overlays, for recording a demo |

### Options

```bash
python app.py --camera 1                  # pick another camera (0, 1, 2, ...)
python app.py --mode hold                 # invisible only while the peace sign is up
python app.py --output demo.mp4           # also record what's shown on screen
python app.py --camera clip.mp4 --headless --no-mirror   # process a video file, print state changes
```

## The pipeline

```
camera ─► hand landmarks ─► gesture recognition ─► trigger ─► image manipulation ─► output
 (OpenCV)   (MediaPipe        (geometry on the      (state     (person mask +         (window /
            HandLandmarker)    21 keypoints)         machine)   background plate)      .mp4)
```

| Stage | Code in `app.py` | What it does |
|-------|------------------|--------------|
| **Camera** | `open_source`, main loop | OpenCV reads a frame and mirrors it so it behaves like a mirror. |
| **Hand landmarks** | `vision.HandLandmarker` | A neural network finds the hand and returns 21 (x, y) keypoints: wrist, knuckles, finger joints, fingertips. |
| **Gesture recognition** | `classify_peace_sign` | Plain geometry. A finger is "extended" if its tip is clearly farther from the wrist than its middle joint. Index and middle extended, ring and pinky curled = peace sign. The thumb is ignored, so it can be tucked or out. The distances are relative to palm size, so the check works at any rotation or distance from the camera. |
| **Trigger** | `GestureStateMachine` | Turns noisy per-frame answers into a stable on/off signal. The peace sign must stay up **and steady** (it can't drift more than 30% of a palm width) for 1 s. A 0.4 s grace period stops a single missed detection from cancelling the effect. |
| **Image manipulation** | `ImageSegmenter`, `BackgroundModel`, `composite` | Builds the person mask, the background plate and the final composite (details below). |
| **Output** | `draw_hud`, `cv2.imshow` | Shows hand landmarks, a status pill (NORMAL / HOLD STEADY / DISAPPEARING / INVISIBLE / REAPPEARING), FPS, and optionally writes an .mp4. |

### Who "sees" the hand, and who makes you disappear?

They are two separate jobs, done by two separate models:

- **Seeing the hand** is done only by **MediaPipe HandLandmarker** (`hand_landmarker.task`).
  It's the only part that looks at pixels to find a hand. Everything after it
  (peace-sign logic, hold timer, state machine) is a few lines of geometry and timing on its
  21 output points. No pixels are involved.
- **The visual effect** is done by the **MediaPipe selfie segmenter**
  (`selfie_segmenter.tflite`) plus plain OpenCV/NumPy compositing. The segmenter answers
  "which pixels are a person?" The compositing replaces those pixels with what the room
  looked like when you weren't standing there. The segmenter knows nothing about the
  gesture; the gesture code knows nothing about pixels. The state machine's on/off signal
  is the only link between them.

### How the background is learned (no manual masking)

Each frame, any pixel the segmenter is confident is **not** you gets blended into a running
background plate (`cv2.accumulateWeighted`). As you move around, more of the room gets seen.
Areas you have never moved out of can't be known. They are filled by **inpainting** from
their surroundings, which looks like a smooth smear. That's why stepping aside once at the
start makes the effect look much better.

The person mask is grown by a few pixels and feathered before compositing. This hides the
outline halo around hair and shoulders. During the 0.6 s fade a subtle heat-haze ripple
plays over the vanishing area.

## Tests

```bash
pip install pytest   # optional; `python test_app.py` works without it
python -m pytest -q
```

`test_app.py` covers the parts that don't need a camera:
- the peace-sign classifier: rotations, scales, thumb in or out, and rejecting a fist, open palm, pointing and three fingers
- every state-machine transition in both modes: hold, early release, detector dropout, a moving hand not triggering, latch toggling, walking out of frame
- the background-plate and compositing maths

## Limitations and tips

- **Keep the camera still.** The background plate assumes a fixed camera. If it moves, press `b`.
- **Lighting changes** are followed slowly. Big changes, like a light switching on, may leave a ghost for a second.
- **Shadows and reflections** you cast aren't part of the person mask, so they stay visible.
- The selfie segmenter works best with one person roughly 0.5–4 m from the camera.
- The hand you raise disappears too. The landmark skeleton keeps floating where the hand
  is, which is a nice touch for a demo. Press `h` to hide it, or `c` for a fully clean shot.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Could not open video source 0` / black window | Try `--camera 1` or `--camera 2`. On macOS go to System Settings → Privacy & Security → Camera and enable your terminal/IDE. |
| Crash with `Check failed: service_ Service is unavailable` | You have mediapipe 1.0.1, which has this bug on macOS. Run `pip install -r requirements.txt` (it excludes 1.0.1). |
| Model download fails (proxy/SSL) | The error message prints a `curl` command for each model. Put the files in `models/`. |
| Weird OpenCV import errors | Make sure only one OpenCV package is installed: `pip uninstall opencv-python` (mediapipe ships `opencv-contrib-python`). |
