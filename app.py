"""Make Me Disappear — hold up a peace sign and vanish from your webcam feed.

Pipeline (one pass per frame):

    camera -> hand landmarks -> gesture recognition -> trigger -> image manipulation -> output

* "Seeing" the hand:   MediaPipe HandLandmarker (21 keypoints per hand).
* Understanding it:    `classify_peace_sign` + `GestureStateMachine` (plain geometry + timers).
* Creating the effect: MediaPipe selfie segmentation (person mask) + `BackgroundModel`
                       (a clean background plate learned automatically) + compositing.

Everything runs locally. Run `python app.py --help` for options.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task",
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}

HOLD_SECONDS = 1.0        # how long the peace sign must be held steady to trigger
RELEASE_GRACE = 0.4       # palm may vanish this long (detector flicker) before we count it as released
STEADY_WINDOW = 0.35      # seconds of palm-centre history used for the steadiness check
STEADY_TOLERANCE = 0.30   # max palm-centre drift within the window, in units of palm size
PERSON_ABSENT_RESET = 1.5 # latch mode: reset if nobody is in frame for this long
FADE_SECONDS = 0.6        # duration of the vanish / reappear transition

# Landmark indices (https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
WRIST, THUMB_IP, THUMB_TIP = 0, 3, 4
INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17
FINGERS = {  # name: (pip, tip)
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}


def ensure_models() -> None:
    """Download the two small MediaPipe models on first run."""
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        path = MODELS_DIR / name
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"Downloading {name} ...", flush=True)
        try:
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(path)
        except Exception as exc:  # network / proxy / certificate problems
            sys.exit(
                f"Could not download {name}: {exc}\n"
                f"Download it manually and place it in {MODELS_DIR}:\n  curl -L -o {path} {url}"
            )


# --------------------------------------------------------------------------------------
# Gesture recognition (pure geometry — no ML beyond the landmarks themselves)
# --------------------------------------------------------------------------------------


@dataclass
class PalmReading:
    is_peace: bool
    center: tuple[float, float]  # pixels
    size: float                  # wrist -> middle-finger MCP distance, pixels
    extended: dict[str, bool]


def classify_peace_sign(points: np.ndarray) -> PalmReading:
    """Decide whether 21 (x, y) pixel landmarks form a peace sign (index + middle up).

    A finger counts as extended when its tip is clearly farther from the wrist than its
    middle joint (PIP). Distances relative to the wrist make this work for any hand
    rotation, and dividing by palm size makes it work at any distance from the camera.
    The thumb is reported but ignored: people hold it tucked or out about equally often.
    """
    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(points[a] - points[b]))

    palm_size = max(dist(WRIST, MIDDLE_MCP), 1e-6)
    extended = {
        name: dist(WRIST, tip) > 1.15 * dist(WRIST, pip) for name, (pip, tip) in FINGERS.items()
    }
    # Thumb: tip moves away from the pinky side of the palm and away from the index knuckle.
    extended["thumb"] = (
        dist(THUMB_TIP, PINKY_MCP) > 1.05 * dist(THUMB_IP, PINKY_MCP)
        and dist(THUMB_TIP, INDEX_MCP) > 0.45 * palm_size
    )
    center = points[[WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP]].mean(axis=0)
    return PalmReading(
        is_peace=extended["index"] and extended["middle"]
        and not extended["ring"] and not extended["pinky"],
        center=(float(center[0]), float(center[1])),
        size=palm_size,
        extended=extended,
    )


# --------------------------------------------------------------------------------------
# Trigger: gesture state machine
# --------------------------------------------------------------------------------------

NORMAL = "NORMAL"
ARMING = "ARMING"
DISAPPEARED = "DISAPPEARED"


@dataclass
class GestureStateMachine:
    """Turns per-frame palm readings into a stable on/off trigger.

    NORMAL --(peace sign appears, steady)--> ARMING --(held HOLD_SECONDS)--> DISAPPEARED
      ^                                       |                                  |
      +------------(palm lost / moved)--------+                                  |
      +------ hold mode:  palm released for RELEASE_GRACE ----------------------+
      +------ latch mode: palm held again, or person left the frame ------------+

    `update` takes the time explicitly so the logic is deterministic and testable.
    """

    mode: str = "latch"  # "latch": palm toggles invisibility. "hold": invisible while the palm is up.
    state: str = NORMAL
    arm_start: float = 0.0
    arming_to_reappear: bool = False  # latch mode: the countdown that brings you back
    last_palm_seen: float = -1e9
    last_person_seen: float = 0.0
    needs_release: bool = False  # palm must drop after a toggle before it can arm again
    history: deque = field(default_factory=deque)

    def reset(self) -> None:
        self.state = NORMAL
        self.history.clear()
        self.arming_to_reappear = False

    def is_steady(self, now: float, palm: PalmReading) -> bool:
        self.history.append((now, palm.center))
        while self.history and now - self.history[0][0] > STEADY_WINDOW:
            self.history.popleft()
        xs = np.array([c for _, c in self.history])
        drift = float(np.linalg.norm(xs.max(axis=0) - xs.min(axis=0)))
        return drift < STEADY_TOLERANCE * palm.size

    def progress(self, now: float) -> float:
        if self.state != ARMING:
            return 0.0
        return min(1.0, (now - self.arm_start) / HOLD_SECONDS)

    def _start_arming(self, now: float, to_reappear: bool) -> None:
        self.state, self.arm_start, self.arming_to_reappear = ARMING, now, to_reappear

    def update(self, now: float, palm: PalmReading | None, person_present: bool) -> str:
        palm_open = palm is not None and palm.is_peace
        if palm_open:
            self.last_palm_seen = now
        else:
            self.history.clear()
        if person_present or palm is not None:
            self.last_person_seen = now
        palm_released = now - self.last_palm_seen > RELEASE_GRACE
        if palm_released:
            self.needs_release = False
        can_arm = palm_open and not self.needs_release

        if self.state == NORMAL:
            if can_arm and self.is_steady(now, palm):
                self._start_arming(now, to_reappear=False)

        elif self.state == ARMING:
            if palm_released:
                self.state = DISAPPEARED if self.arming_to_reappear else NORMAL
            elif palm_open and not self.is_steady(now, palm):
                self.arm_start = now  # moving hand: restart the countdown
            elif now - self.arm_start >= HOLD_SECONDS:
                if self.arming_to_reappear:
                    self.reset()
                else:
                    self.state = DISAPPEARED
                self.needs_release = True

        elif self.state == DISAPPEARED:
            if self.mode == "hold":
                if palm_released:
                    self.reset()
            elif now - self.last_person_seen > PERSON_ABSENT_RESET:
                self.reset()  # latch mode: the user walked out of frame
            elif can_arm and self.is_steady(now, palm):
                self._start_arming(now, to_reappear=True)

        if self.state == ARMING and self.arming_to_reappear and \
                now - self.last_person_seen > PERSON_ABSENT_RESET:
            self.reset()
        return self.state

    @property
    def wants_invisible(self) -> bool:
        return self.state == DISAPPEARED or (self.state == ARMING and self.arming_to_reappear)


# --------------------------------------------------------------------------------------
# Image manipulation: background plate + compositing
# --------------------------------------------------------------------------------------


class BackgroundModel:
    """Learns a clean background plate automatically — no manual painting needed.

    Every frame, pixels the segmenter is confident are *not* the person are blended into
    the plate. Over a few seconds of normal movement the plate fills in; regions that have
    never been uncovered are filled by inpainting from their surroundings.
    """

    LEARN_RATE = 0.12   # how fast already-seen background pixels follow lighting changes
    REFILL_EVERY = 10   # frames between inpainting refreshes (inpainting is the slow bit)

    def __init__(self, shape: tuple[int, int, int]):
        self.plate = np.zeros(shape, np.float32)
        self.seen = np.zeros(shape[:2], np.uint8)  # 255 = this pixel has been seen unoccluded
        self.filled = np.zeros(shape, np.uint8)    # plate with never-seen holes inpainted
        self.frames_since_fill = self.REFILL_EVERY

    def reset(self) -> None:
        self.plate[:] = 0
        self.seen[:] = 0
        self.frames_since_fill = self.REFILL_EVERY

    @property
    def coverage(self) -> float:
        return cv2.countNonZero(self.seen) / self.seen.size

    def coverage_behind(self, person_mask: np.ndarray) -> float:
        """Share of the pixels currently covered by the person that we have a real plate for."""
        person = ((person_mask > 0.5) * 255).astype(np.uint8)
        total = cv2.countNonZero(person)
        return 1.0 if total == 0 else cv2.countNonZero(cv2.bitwise_and(person, self.seen)) / total

    def update(self, frame: np.ndarray, person_mask: np.ndarray) -> None:
        background = ((person_mask < 0.15) * 255).astype(np.uint8)
        first_time = cv2.bitwise_and(background, cv2.bitwise_not(self.seen))
        cv2.accumulateWeighted(frame, self.plate, self.LEARN_RATE, mask=background)
        cv2.copyTo(frame.astype(np.float32), first_time, self.plate)  # new pixels: take as-is
        cv2.bitwise_or(self.seen, background, dst=self.seen)
        self.frames_since_fill += 1
        if self.frames_since_fill >= self.REFILL_EVERY:
            self._refill()

    def _refill(self) -> None:
        self.frames_since_fill = 0
        plate = self.plate.astype(np.uint8)
        unseen = cv2.countNonZero(self.seen) < self.seen.size
        if not unseen or not self.seen.any():
            self.filled = plate
            return
        h, w = self.seen.shape
        small = cv2.resize(plate, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        hole = cv2.resize(cv2.bitwise_not(self.seen), (w // 4, h // 4))
        guess = cv2.resize(cv2.inpaint(small, hole, 5, cv2.INPAINT_TELEA), (w, h))
        self.filled = cv2.copyTo(plate, self.seen, guess)


def soften_mask(mask: np.ndarray, grow_px: int, feather_px: int) -> np.ndarray:
    """Grow the person mask a little (hide edge halos) and feather it (no hard seams).

    Done at quarter resolution — the result is soft anyway and this is ~15x faster.
    """
    h, w = mask.shape
    small = cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    g = max(1, grow_px // 4)
    small = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    f = max(1, feather_px // 4)
    small = cv2.GaussianBlur(small, (2 * f + 1, 2 * f + 1), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


_grid_cache: dict = {}


def shimmer(frame: np.ndarray, strength: float, t: float) -> np.ndarray:
    """Subtle heat-haze ripple used only during the transition."""
    if strength <= 0.01:
        return frame
    h, w = frame.shape[:2]
    if (h, w) not in _grid_cache:
        _grid_cache[(h, w)] = np.indices((h, w), dtype=np.float32)
    ys, xs = _grid_cache[(h, w)]
    amp = 6.0 * strength
    map_x = xs + amp * np.sin(ys / 9.0 + t * 14.0)
    map_y = ys + 0.5 * amp * np.cos(xs / 13.0 + t * 11.0)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def composite(frame, plate, soft_mask, alpha, t):
    """Replace the person with the background plate, weighted by `alpha` (0..1).

    Inside the mask the (rippling) person cross-fades into the plate; outside the mask
    the live frame is untouched.
    """
    if alpha <= 0.001:
        return frame
    ripple = 4 * alpha * (1 - alpha)  # 0 at both ends, peaks mid-transition
    inner = cv2.addWeighted(shimmer(frame, ripple, t), 1 - alpha, plate, alpha, 0)
    m = soft_mask.astype(np.float32)
    return cv2.blendLinear(frame, inner, 1 - m, m)


# --------------------------------------------------------------------------------------
# Drawing / HUD
# --------------------------------------------------------------------------------------

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
STATUS_COLORS = {  # BGR
    "NORMAL": (90, 200, 90),
    "HOLD STEADY": (0, 200, 255),
    "DISAPPEARING": (255, 120, 60),
    "INVISIBLE": (255, 80, 200),
    "REAPPEARING": (255, 200, 60),
}


def draw_hand(img, points, palm: PalmReading, progress: float):
    color = (0, 255, 180) if palm.is_peace else (200, 200, 200)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(points[a].astype(int)), tuple(points[b].astype(int)), color, 2, cv2.LINE_AA)
    for p in points:
        cv2.circle(img, tuple(p.astype(int)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p.astype(int)), 4, color, 1, cv2.LINE_AA)
    if progress > 0:  # hold-progress ring around the palm
        c = tuple(int(v) for v in palm.center)
        r = int(palm.size * 1.3)
        cv2.circle(img, c, r, (60, 60, 60), 3, cv2.LINE_AA)
        cv2.ellipse(img, c, (r, r), -90, 0, 360 * progress, (0, 200, 255), 5, cv2.LINE_AA)


def put_text(img, text, org, scale=0.55, color=(240, 240, 240)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_hud(img, status, fps, mode, behind, palm, recording):
    color = STATUS_COLORS.get("HOLD STEADY" if status.startswith("HOLD") else status, (255, 255, 255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(status, font, 0.9, 2)
    cv2.rectangle(img, (16, 16), (16 + tw + 28, 16 + th + 24), (20, 20, 20), -1)
    cv2.rectangle(img, (16, 16), (16 + tw + 28, 16 + th + 24), color, 2)
    cv2.putText(img, status, (30, 16 + th + 11), font, 0.9, color, 2, cv2.LINE_AA)

    gesture = "no hand"
    if palm is not None:
        gesture = "PEACE SIGN" if palm.is_peace else "hand (" + "".join(
            k[0].upper() if v else "." for k, v in palm.extended.items()) + ")"
    lines = [
        f"FPS {fps:5.1f}" + ("   REC" if recording else ""),
        f"mode: {mode}",
        f"gesture: {gesture}",
        f"background behind you: {behind * 100:3.0f}%",
    ]
    y = 16 + th + 52
    for line in lines:
        put_text(img, line, (18, y))
        y += 24
    if behind < 0.9:
        put_text(img, "tip: step out of frame for a moment so the background can be learned",
                 (18, y + 6), 0.55, (0, 220, 255))
    put_text(img, "hold peace sign to vanish | q quit  m mode  b relearn bg  h landmarks  d debug  c clean",
             (18, img.shape[0] - 18), 0.5)


def draw_debug_insets(img, plate, mask):
    h, w = img.shape[:2]
    iw, ih = w // 5, h // 5
    insets = [cv2.resize(plate, (iw, ih)), cv2.cvtColor(cv2.resize((mask * 255).astype(np.uint8), (iw, ih)), cv2.COLOR_GRAY2BGR)]
    for i, (inset, label) in enumerate(zip(insets, ["background plate", "person mask"])):
        x, y = w - iw - 16, 16 + i * (ih + 26)
        img[y:y + ih, x:x + iw] = inset
        cv2.rectangle(img, (x, y), (x + iw, y + ih), (255, 255, 255), 1)
        cv2.putText(img, label, (x, y + ih + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------


def open_source(source: str, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if source.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        sys.exit(
            f"Could not open video source {source!r}. Try --camera 1 (or 2), and on macOS allow "
            "camera access for your terminal in System Settings > Privacy & Security > Camera."
        )
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description="Hold up a peace sign to disappear from your webcam.")
    parser.add_argument("--camera", default="0", help="camera index (0, 1, ...) or a video file path")
    parser.add_argument("--mode", choices=["latch", "hold"], default="latch",
                        help="latch: peace sign toggles invisibility (default). hold: invisible only while it is up")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true", help="don't flip the image horizontally")
    parser.add_argument("--output", help="also save the shown video to this .mp4 file")
    parser.add_argument("--headless", action="store_true", help="no window; print status (for testing)")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run forever)")
    args = parser.parse_args()

    ensure_models()
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,  # track a single hand only
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    ))
    segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS_DIR / "selfie_segmenter.tflite")),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    ))

    cap = open_source(args.camera, args.width, args.height)
    ok, frame = cap.read()
    if not ok:
        sys.exit("Video source opened but returned no frames (camera permission denied?).")
    # Video files run on their own clock (frame index / fps) so timings match real time.
    is_file = not args.camera.isdigit()
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if args.output:
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps if is_file else 30.0, (w, h))

    fsm = GestureStateMachine(mode=args.mode)
    bg = BackgroundModel(frame.shape)
    alpha = 0.0
    start = last = time.perf_counter()
    frame_times: deque = deque(maxlen=30)
    fps = 0.0
    show_landmarks, show_debug, clean = True, False, False
    last_print, last_status, frame_no = -1e9, None, 0
    window = "Make Me Disappear"
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            if frame_no > 0:
                ok, frame = cap.read()
                if not ok:
                    break
            frame_no += 1
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            wall = time.perf_counter()
            now = start + frame_no / src_fps if is_file else wall
            dt, last = min(now - last, 0.25), now
            frame_times.append(wall)
            if len(frame_times) > 1:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

            # 1) camera frame -> MediaPipe image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((now - start) * 1000) + frame_no  # strictly increasing, as MediaPipe requires

            # 2) hand landmarks — this is the part that actually "sees" the hand
            hand_result = hands.detect_for_video(mp_image, ts_ms)
            palm, points = None, None
            if hand_result.hand_landmarks:
                h, w = frame.shape[:2]
                points = np.array([(lm.x * w, lm.y * h) for lm in hand_result.hand_landmarks[0]], np.float32)
                # 3) gesture recognition
                palm = classify_peace_sign(points)

            # person segmentation — tells the effect which pixels to replace
            seg = segmenter.segment_for_video(mp_image, ts_ms)
            person = seg.confidence_masks[-1].numpy_view().astype(np.float32)
            if person.ndim == 3:
                person = person[..., 0]
            person_present = cv2.countNonZero((person > 0.5).astype(np.uint8)) > 0.01 * person.size

            # 4) trigger
            fsm.update(now, palm, person_present)
            target = 1.0 if fsm.wants_invisible else 0.0
            step = dt / FADE_SECONDS
            alpha = min(target, alpha + step) if target > alpha else max(target, alpha - step)

            # 5) image manipulation
            bg.update(frame, soften_mask(person, grow_px=25, feather_px=5))  # learn conservatively
            soft = soften_mask(person, grow_px=12, feather_px=10)
            out = composite(frame, bg.filled, soft, alpha, now)

            if fsm.state == ARMING:
                status = f"HOLD STEADY {int(fsm.progress(now) * 100)}%"
            elif alpha >= 0.999:
                status = "INVISIBLE"
            elif target == 1.0:
                status = "DISAPPEARING"
            elif alpha > 0.001:
                status = "REAPPEARING"
            else:
                status = "NORMAL"

            # 6) output
            if not clean:
                if show_landmarks and points is not None:
                    draw_hand(out, points, palm, fsm.progress(now))
                draw_hud(out, status, fps, fsm.mode, bg.coverage_behind(person), palm, writer is not None)
                if show_debug:
                    draw_debug_insets(out, bg.filled, soft)
            if writer is not None:
                writer.write(out)

            if args.headless:
                if status.split(" ")[0] != (last_status or "").split(" ")[0] or now - last_print > 1.0:
                    gesture = "none" if palm is None else ("PEACE" if palm.is_peace else "other")
                    print(f"t={now - start:6.2f}s fps={fps:5.1f} state={fsm.state:<11} status={status:<16} "
                          f"hand={gesture:<5} person={person_present} bg={bg.coverage * 100:3.0f}%", flush=True)
                    last_print, last_status = now, status
            else:
                cv2.imshow(window, out)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    fsm.mode = "latch" if fsm.mode == "hold" else "hold"
                    fsm.reset()
                elif key == ord("b"):
                    bg.reset()
                elif key == ord("h"):
                    show_landmarks = not show_landmarks
                elif key == ord("d"):
                    show_debug = not show_debug
                elif key == ord("c"):
                    clean = not clean
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.max_frames and frame_no >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        hands.close()
        segmenter.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
