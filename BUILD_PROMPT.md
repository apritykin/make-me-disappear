# Build Prompt: Make Me Disappear

> Copy everything below the line into your coding agent (Claude Code, Cursor,
> Codex, etc.) from inside an empty folder. The agent should build the whole
> project from this description.

---

## Goal

Build **Make Me Disappear**: a local Python app that shows the webcam feed in a
window. When the user holds a **peace sign** (index and middle fingers up, ring
and pinky curled) steady for 1 second, they fade out of the video and the room
behind them shows through. They stay invisible after lowering their hand.
Holding the peace sign again brings them back.

It's a demo of a full real-time vision pipeline:

```
camera ─► hand landmarks ─► gesture recognition ─► trigger ─► image manipulation ─► output
 (OpenCV)   (MediaPipe        (geometry on the      (state     (person mask +         (window /
            HandLandmarker)    21 keypoints)         machine)   background plate)      .mp4)
```

Everything runs locally at roughly 30 FPS on a laptop CPU. The only network
access is a one-time download of two small MediaPipe models. No cloud APIs.

## Constraints

- **Language:** Python 3.11+
- **Dependencies:** put exactly this in `requirements.txt`, comments included:
  ```
  # mediapipe 1.0.1 aborts on macOS when creating the hand landmarker ("Service is unavailable"),
  # so it is excluded. mediapipe already pulls in opencv-contrib-python; do NOT also install
  # opencv-python — the two packages overwrite each other's files.
  mediapipe>=0.10.35,!=1.0.1
  opencv-contrib-python>=4.9
  numpy>=1.26
  ```
  Use only the standard library beyond these (`argparse`, `urllib.request`,
  `dataclasses`, `collections.deque`, `pathlib`, `time`, `sys`).
- **Files:** exactly these:
  ```
  make-me-disappear/
  ├── app.py            # the whole app, ~580 lines
  ├── test_app.py       # camera-free tests, ~175 lines
  ├── requirements.txt
  ├── README.md
  ├── .gitignore        # .venv/ __pycache__/ .pytest_cache/ models/ *.mp4
  └── models/           # created at runtime, never committed
  ```
- One script. Plain functions plus two small classes (`GestureStateMachine`,
  `BackgroundModel`) and one dataclass (`PalmReading`). No frameworks or extra
  modules. Split `app.py` into sections with banner comments: Configuration,
  Gesture recognition, Trigger, Image manipulation, Drawing / HUD, Main loop.
- Use the **MediaPipe Tasks API** (`mediapipe.tasks.python.vision`), not the
  legacy `mp.solutions` API.
- Use OpenCV for all image work. Colors are BGR.

## `app.py` specification

### Module docstring
Name the app, show the pipeline line above, and say which part does what:
MediaPipe HandLandmarker "sees" the hand; `classify_peace_sign` +
`GestureStateMachine` understand it (plain geometry and timers); the selfie
segmenter + `BackgroundModel` + compositing create the effect. Mention
`python app.py --help`.

### Configuration
- `MODELS_DIR = Path(__file__).resolve().parent / "models"`
- `MODELS` maps file name to URL:
  - `hand_landmarker.task` →
    `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
  - `selfie_segmenter.tflite` →
    `https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite`
- Tuning constants, each with a short comment:
  | Name | Value | Meaning |
  |------|-------|---------|
  | `HOLD_SECONDS` | 1.0 | how long the peace sign must be held steady |
  | `RELEASE_GRACE` | 0.4 | palm may vanish this long (detector flicker) before it counts as released |
  | `STEADY_WINDOW` | 0.35 | seconds of palm-center history for the steadiness check |
  | `STEADY_TOLERANCE` | 0.30 | max palm-center drift in that window, in palm sizes |
  | `PERSON_ABSENT_RESET` | 1.5 | latch mode: reset if nobody is in frame this long |
  | `FADE_SECONDS` | 0.6 | length of the vanish / reappear transition |
- Landmark indices: `WRIST=0, THUMB_IP=3, THUMB_TIP=4, INDEX_MCP=5,
  MIDDLE_MCP=9, PINKY_MCP=17`, and `FINGERS = {"index": (6, 8), "middle":
  (10, 12), "ring": (14, 16), "pinky": (18, 20)}` as `(pip, tip)` pairs.

### `ensure_models()`
Create `models/`. For each model that's missing or empty, print
`Downloading <name> ...`, download to `<name>.part` with
`urllib.request.urlretrieve`, then rename. On any exception, `sys.exit` with
the error and a ready-to-paste `curl -L -o <path> <url>` command.

### Gesture recognition
`@dataclass PalmReading`: `is_peace: bool`, `center: tuple[float, float]`
(pixels), `size: float` (wrist→middle MCP distance), `extended: dict[str, bool]`.

`classify_peace_sign(points)` takes a `(21, 2)` array of pixel coordinates:
- `palm_size = max(dist(WRIST, MIDDLE_MCP), 1e-6)`
- A finger is extended when `dist(WRIST, tip) > 1.15 * dist(WRIST, pip)`.
- Thumb is extended when `dist(THUMB_TIP, PINKY_MCP) > 1.05 * dist(THUMB_IP,
  PINKY_MCP)` **and** `dist(THUMB_TIP, INDEX_MCP) > 0.45 * palm_size`. It's
  reported in `extended` but **ignored** for the peace decision.
- `center` = mean of WRIST, INDEX_MCP, MIDDLE_MCP and PINKY_MCP.
- `is_peace` = index and middle extended, ring and pinky not.
- The docstring explains why: distances from the wrist make it
  rotation-invariant, and comparing ratios makes it scale-invariant.

### Trigger: `GestureStateMachine` (dataclass)
States are string constants `NORMAL`, `ARMING`, `DISAPPEARED`. Put an ASCII
transition diagram in the docstring. `update` takes time explicitly so it's
deterministic and testable.

Fields: `mode="latch"` (or `"hold"`), `state=NORMAL`, `arm_start=0.0`,
`arming_to_reappear=False`, `last_palm_seen=-1e9`, `last_person_seen=0.0`,
`needs_release=False`, `history: deque`.

- `reset()`: state → NORMAL, clear history, `arming_to_reappear=False`.
- `is_steady(now, palm)`: append `(now, center)` to history, drop entries older
  than `STEADY_WINDOW`, return `True` if the bounding-box diagonal of the
  remaining centers is `< STEADY_TOLERANCE * palm.size`.
- `progress(now)`: 0 unless ARMING, otherwise `min(1, (now - arm_start) / HOLD_SECONDS)`.
- `update(now, palm, person_present) -> str`:
  1. `palm_open = palm is not None and palm.is_peace`. If open, set
     `last_palm_seen = now`, otherwise clear history.
  2. If `person_present` or any palm is seen, set `last_person_seen = now`.
  3. `palm_released = now - last_palm_seen > RELEASE_GRACE`. If released,
     clear `needs_release`. `can_arm = palm_open and not needs_release`.
  4. **NORMAL:** if `can_arm` and steady → ARMING (to vanish).
  5. **ARMING:** if released → back to DISAPPEARED when arming to reappear,
     else NORMAL. If the palm is open but not steady → restart the countdown
     (`arm_start = now`). If held `HOLD_SECONDS` → reset() when reappearing,
     else DISAPPEARED. Either way set `needs_release = True`, so the same held
     sign can't toggle twice.
  6. **DISAPPEARED:** in hold mode, reset() once released. In latch mode,
     reset() if `now - last_person_seen > PERSON_ABSENT_RESET`. Otherwise, if
     `can_arm` and steady, start ARMING to reappear.
  7. Also reset() if ARMING to reappear and the person has been absent longer
     than `PERSON_ABSENT_RESET`.
- `wants_invisible` property: DISAPPEARED, or ARMING to reappear (the user
  stays invisible during the reappear countdown).

### Image manipulation
**`BackgroundModel`** learns a clean background plate with no manual masking.
Class constants `LEARN_RATE = 0.12` and `REFILL_EVERY = 10`. State: `plate`
(float32 image), `seen` (uint8 mask, 255 = seen unoccluded), `filled` (uint8
plate with holes inpainted), and `frames_since_fill`.
- `reset()` zeros plate and seen and forces a refill on the next update.
- `coverage`: fraction of pixels ever seen.
- `coverage_behind(person_mask)`: fraction of currently-person pixels
  (`> 0.5`) that have a real plate. Returns 1.0 if there's no person.
- `update(frame, person_mask)`: background = `person_mask < 0.15`. Blend
  frame into plate with `cv2.accumulateWeighted(..., mask=background)`. For
  pixels being seen for the first time, copy the frame value straight in with
  `cv2.copyTo`. OR the background into `seen`. Every `REFILL_EVERY` frames,
  call `_refill()`.
- `_refill()`: if every pixel or no pixel has been seen, `filled = plate`.
  Otherwise inpaint the unseen holes **at quarter resolution**
  (`cv2.inpaint(..., 5, cv2.INPAINT_TELEA)`), upscale, and use the real plate
  wherever `seen` is set.

**`soften_mask(mask, grow_px, feather_px)`**: at quarter resolution, dilate
with an elliptical kernel (grows the mask to hide edge halos), Gaussian blur
(feathers it), then resize back. Note that this is ~15x faster than full
resolution.

**`shimmer(frame, strength, t)`**: heat-haze ripple. Return the frame
unchanged if `strength <= 0.01`. Otherwise `cv2.remap` with
`map_x = xs + amp*sin(ys/9 + t*14)` and `map_y = ys + 0.5*amp*cos(xs/13 + t*11)`,
where `amp = 6 * strength`. Cache the `np.indices` grid per frame size in a
module-level dict.

**`composite(frame, plate, soft_mask, alpha, t)`**: return the frame unchanged
if `alpha <= 0.001`. Otherwise `ripple = 4*alpha*(1-alpha)` (peaks mid-fade),
`inner = addWeighted(shimmer(frame, ripple, t), 1-alpha, plate, alpha)`, and
return `cv2.blendLinear(frame, inner, 1-mask, mask)`.

### Drawing / HUD
- `HAND_CONNECTIONS` from `vision.HandLandmarksConnections.HAND_CONNECTIONS`.
- `STATUS_COLORS` (BGR): NORMAL green, HOLD STEADY orange, DISAPPEARING blue,
  INVISIBLE magenta, REAPPEARING cyan.
- `draw_hand(img, points, palm, progress)`: skeleton lines, teal if peace sign
  else light gray, white dots with a colored outline. While arming, draw a
  hold-progress ring around the palm (`radius = 1.3 * palm.size`, a gray base
  circle plus an orange arc from -90° sweeping `360 * progress`).
- `put_text(...)`: white text over a thicker black outline so it's readable on
  any background.
- `draw_hud(img, status, fps, mode, behind, palm, recording)`: a status pill
  top-left (dark fill, colored border and text). Below it: FPS (plus `REC`
  when recording), mode, gesture (`no hand`, `PEACE SIGN`, or
  `hand (TIM..)`, one letter per extended finger), and
  `background behind you: NN%`. If coverage behind is under 90%, add a yellow
  tip to step out of frame. Put a key-hint footer at the bottom.
- `draw_debug_insets(img, plate, mask)`: two 1/5-size labeled insets on the
  top right: "background plate" and "person mask".

### Main loop
`open_source(source, width, height)`: a digit string opens a camera index (set
width/height), anything else opens a video file. If it fails, `sys.exit` with
a hint to try `--camera 1` and, on macOS, to allow camera access under
*System Settings > Privacy & Security > Camera*.

`main()` CLI flags: `--camera` (default `"0"`, index or file path),
`--mode {latch,hold}` (default latch), `--width 1280`, `--height 720`,
`--no-mirror`, `--output FILE.mp4`, `--headless`, `--max-frames N`.

1. `ensure_models()`. Create the `HandLandmarker` in `RunningMode.VIDEO` with
   `num_hands=1` and detection/presence confidence 0.6, tracking 0.5. Create
   the `ImageSegmenter` in VIDEO mode with confidence masks only.
2. Open the source and read the first frame (exit with a permission hint if
   it returns nothing). Video files run on their own clock
   (`start + frame_no / src_fps`) so the timers behave as in real time.
   Optionally open a `VideoWriter` (`mp4v`).
3. Each frame:
   1. Mirror with `cv2.flip(frame, 1)` unless `--no-mirror`. Update the FPS
      estimate from the last 30 wall-clock times. Cap `dt` at 0.25 s.
   2. Convert BGR→RGB into an `mp.Image`. Use a **strictly increasing**
      timestamp in ms (`int((now-start)*1000) + frame_no`), because MediaPipe
      requires one.
   3. Run `detect_for_video`. If there's a hand, scale the landmarks to pixels
      and call `classify_peace_sign`.
   4. Run `segment_for_video`. The person mask is `confidence_masks[-1]` as
      float32 (squeeze a trailing channel). `person_present` = more than 1% of
      pixels above 0.5.
   5. `fsm.update(...)`. Step `alpha` toward the target (1 if
      `wants_invisible`) by `dt / FADE_SECONDS`.
   6. Learn conservatively with `bg.update(frame, soften_mask(person, 25, 5))`,
      then composite with `soften_mask(person, 12, 10)` and `bg.filled`.
   7. Status: `HOLD STEADY NN%` while arming, then `INVISIBLE` (alpha ≥ 0.999),
      `DISAPPEARING`, `REAPPEARING`, or `NORMAL`.
   8. Unless in clean view, draw hand, HUD, and optional debug insets. Write
      the frame to the video if recording.
   9. **Headless:** print one line per status change or once a second with
      t, fps, state, status, hand, person, and bg coverage. **Windowed:**
      `imshow`, then handle keys `q`/Esc quit, `m` toggle mode + reset FSM,
      `b` reset background, `h` landmarks, `d` debug, `c` clean. Quit when the
      window is closed.
   10. Stop at `--max-frames` if set.
4. Catch `KeyboardInterrupt`. In `finally`, release the capture and writer,
   close both MediaPipe tasks, and `destroyAllWindows()`.

## `test_app.py` specification

Must run with `python -m pytest -q` **and** with plain `python test_app.py`
(a `__main__` block that runs every `test_*` function and prints `ok`).
No camera and no model files needed.

- `make_hand(open_fingers, angle_deg, scale, offset)` builds a synthetic
  21-point hand, fingers up (image y grows down). Extended fingers go straight
  out. Curled fingers fold the tip back toward the palm. The thumb has an open
  and a tucked layout. Then rotate, scale, and offset.
- Gesture tests: peace sign detected with thumb in or out; still detected at
  angles −60, −30, 30, 60, 180 and scales 0.4 and 2.5; fist, open palm,
  pointing, three fingers, and middle+ring are all rejected.
- State-machine tests (30 fps helper `run(fsm, frames, palm, t0, person=True)`):
  - hold mode full cycle: normal → arming → disappeared, stays while held,
    survives a 3-frame dropout, back to normal on a fist
  - releasing early cancels arming
  - a peace sign sweeping across the frame never triggers
  - latch mode: toggle on, no re-arm while still held, stays invisible with
    the hand down, reappear countdown keeps you invisible, toggle off, same
    held sign doesn't re-vanish, walking out of frame resets
  - latch mode: an abandoned reappear countdown returns to DISAPPEARED
- Image test: a flat-colored scene with a white "person" rectangle. Show the
  person on the left, then on the right, 12 updates each → `coverage == 1.0`.
  `composite` at alpha 1 matches the scene within 2 levels. At alpha 0 it
  returns the frame unchanged.

## `README.md` specification

Write it for someone who just wants to try the trick, then explain the
internals.

1. **Title and intro:** the peace-sign trick in two sentences. Local,
   real-time, OpenCV + MediaPipe, no cloud calls.
2. **Quick start:** Python 3.11+, venv commands (with the Windows activate
   variant), `pip install -r requirements.txt`, `python app.py`. Mention the
   first-run model download (~8 MB) and the macOS camera permission.
3. **How to do the trick:** numbered steps: step out of frame briefly so the
   background is learned (point at the HUD percentage), raise a peace sign,
   hold steady 1 s while the ring fills, lower your hand and stay invisible,
   repeat to come back (or walk out for 1.5 s). Only one hand is tracked.
4. **Controls:** key table (q/Esc, m, b, h, d, c), describing latch vs hold.
5. **Options:** example commands for `--camera 1`, `--mode hold`,
   `--output demo.mp4`, and `--camera clip.mp4 --headless --no-mirror`.
6. **The pipeline:** the diagram, then a table mapping each stage to its code
   in `app.py` and what it does.
7. **Who "sees" the hand, and who makes you disappear?** Two separate models:
   HandLandmarker is the only thing that looks at pixels for the gesture.
   Everything after it is geometry and timing. The selfie segmenter plus
   compositing makes the effect. The state machine's on/off signal is the only
   link between them.
8. **How the background is learned:** running average of confident
   non-person pixels; never-seen areas are inpainted (a smooth smear), which
   is why stepping aside first helps; mask growth, feathering, and the
   heat-haze ripple during the fade.
9. **Tests:** how to run them and what they cover.
10. **Limitations and tips:** keep the camera still (`b` after moving it),
    slow lighting adaptation, shadows and reflections stay, best at 0.5–4 m
    with one person, the raised hand vanishes too but the landmark skeleton
    floats (`h` / `c` to hide).
11. **Troubleshooting** table: can't open camera, the mediapipe 1.0.1
    `Service is unavailable` crash, model download failing behind a
    proxy (use the printed `curl` commands), and OpenCV import errors from
    having both `opencv-python` and `opencv-contrib-python` installed.

## Done when

- [ ] `pip install -r requirements.txt` works in a fresh venv
- [ ] `python -m pytest -q` and `python test_app.py` both pass with no camera
- [ ] `python app.py --help` lists all eight flags
- [ ] First run downloads both models into `models/`; later runs skip it
- [ ] `python app.py --camera some_clip.mp4 --headless --no-mirror --max-frames 300`
      runs to completion and prints status lines (if there's no clip, record
      one with `--output` or say this was skipped)
- [ ] `python app.py` opens the webcam. A steady peace sign fills the ring and
      fades you out, you stay invisible with the hand down, a second peace
      sign brings you back, and every key works. *(This is a manual check
      that needs a real camera. If you're an agent without one, say so
      instead of claiming it passed.)*
