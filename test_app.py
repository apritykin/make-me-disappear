"""Tests for gesture recognition and the trigger state machine (no camera needed).

Run:  python -m pytest -q      (or simply: python test_app.py)
"""

import numpy as np

import app
from app import ARMING, DISAPPEARED, NORMAL, GestureStateMachine, classify_peace_sign


def make_hand(open_fingers=("thumb", "index", "middle", "ring", "pinky"), angle_deg=0.0,
              scale=1.0, offset=(400.0, 400.0)) -> np.ndarray:
    """Synthetic 21-point hand, palm facing camera, fingers pointing up (image y grows down)."""
    pts = np.zeros((21, 2), np.float32)
    pts[0] = (0, 0)  # wrist
    # thumb: CMC, MCP, IP, TIP
    thumb_open = [(-30, -20), (-55, -40), (-75, -60), (-95, -75)]
    thumb_closed = [(-30, -20), (-40, -45), (-20, -60), (0, -65)]
    pts[1:5] = thumb_open if "thumb" in open_fingers else thumb_closed
    bases = {"index": -30, "middle": -10, "ring": 10, "pinky": 28}
    for i, name in enumerate(["index", "middle", "ring", "pinky"]):
        x = bases[name]
        mcp = 5 + 4 * i
        if name in open_fingers:
            pts[mcp:mcp + 4] = [(x, -90), (x * 1.1, -130), (x * 1.2, -160), (x * 1.3, -185)]
        else:  # curled: tip folds back toward the palm
            pts[mcp:mcp + 4] = [(x, -90), (x, -115), (x, -95), (x, -75)]
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    return (pts @ rot.T) * scale + np.array(offset, np.float32)


PEACE = classify_peace_sign(make_hand(open_fingers=("index", "middle")))
FIST = classify_peace_sign(make_hand(open_fingers=()))


# ---- gesture recognition -------------------------------------------------------------

def test_peace_sign_detected():
    assert PEACE.is_peace
    assert classify_peace_sign(make_hand(open_fingers=("thumb", "index", "middle"))).is_peace  # thumb out


def test_peace_sign_rotation_and_scale_invariant():
    for angle in (-60, -30, 30, 60, 180):
        for scale in (0.4, 2.5):
            assert classify_peace_sign(
                make_hand(open_fingers=("index", "middle"), angle_deg=angle, scale=scale)).is_peace


def test_other_hands_rejected():
    assert not FIST.is_peace
    assert not classify_peace_sign(make_hand()).is_peace                                  # open palm
    assert not classify_peace_sign(make_hand(open_fingers=("index",))).is_peace           # pointing
    assert not classify_peace_sign(make_hand(open_fingers=("index", "middle", "ring"))).is_peace
    assert not classify_peace_sign(make_hand(open_fingers=("middle", "ring"))).is_peace


# ---- state machine -------------------------------------------------------------------

def run(fsm, frames, palm, t0, person=True, fps=30):
    t = t0
    for _ in range(frames):
        fsm.update(t, palm, person)
        t += 1 / fps
    return t


def test_hold_mode_full_cycle():
    fsm = GestureStateMachine(mode="hold")
    t = run(fsm, 10, None, 0.0)
    assert fsm.state == NORMAL and not fsm.wants_invisible

    t = run(fsm, 5, PEACE, t)
    assert fsm.state == ARMING and 0 < fsm.progress(t) < 1

    t = run(fsm, int(app.HOLD_SECONDS * 30) + 2, PEACE, t)
    assert fsm.state == DISAPPEARED and fsm.wants_invisible

    t = run(fsm, 60, PEACE, t)  # keep holding: stay invisible
    assert fsm.state == DISAPPEARED

    t = run(fsm, 3, None, t)   # brief detector dropout (< RELEASE_GRACE): still invisible
    assert fsm.state == DISAPPEARED

    run(fsm, 30, FIST, t)      # hand closed: back to normal
    assert fsm.state == NORMAL and not fsm.wants_invisible


def test_releasing_early_cancels():
    fsm = GestureStateMachine(mode="hold")
    t = run(fsm, 15, PEACE, 0.0)  # 0.5 s, not enough
    assert fsm.state == ARMING
    run(fsm, 30, None, t)
    assert fsm.state == NORMAL


def test_moving_hand_does_not_trigger():
    fsm = GestureStateMachine(mode="hold")
    t = 0.0
    for i in range(90):  # 3 s of a peace sign sweeping across the frame
        palm = classify_peace_sign(make_hand(open_fingers=("index", "middle"), offset=(200 + 15 * i, 400)))
        fsm.update(t, palm, True)
        t += 1 / 30
        assert fsm.state != DISAPPEARED


def test_latch_mode_toggle_and_leave_frame():
    fsm = GestureStateMachine(mode="latch")
    t = run(fsm, 40, PEACE, 0.0)
    assert fsm.state == DISAPPEARED

    t = run(fsm, 30, PEACE, t)        # still holding after toggle: must not re-arm
    assert fsm.state == DISAPPEARED
    t = run(fsm, 30, None, t)        # lower the hand: stay invisible
    assert fsm.state == DISAPPEARED and fsm.wants_invisible

    t = run(fsm, 15, PEACE, t)        # peace sign again: counting down to reappear, still invisible
    assert fsm.state == ARMING and fsm.wants_invisible
    t = run(fsm, 30, PEACE, t)
    assert fsm.state == NORMAL and not fsm.wants_invisible
    t = run(fsm, 60, PEACE, t)        # same held sign must not immediately re-vanish
    assert fsm.state == NORMAL

    t = run(fsm, 30, None, t)
    t = run(fsm, 40, PEACE, t)        # vanish again, then walk out of frame
    assert fsm.state == DISAPPEARED
    run(fsm, int(app.PERSON_ABSENT_RESET * 30) + 5, None, t, person=False)
    assert fsm.state == NORMAL


def test_latch_mode_abandoned_reappear_countdown_stays_invisible():
    fsm = GestureStateMachine(mode="latch")
    t = run(fsm, 40, PEACE, 0.0)
    t = run(fsm, 30, None, t)
    t = run(fsm, 10, PEACE, t)
    assert fsm.state == ARMING
    run(fsm, 30, None, t)
    assert fsm.state == DISAPPEARED


# ---- image manipulation --------------------------------------------------------------

def test_background_model_and_composite_remove_person():
    h, w = 120, 160
    scene = np.full((h, w, 3), (40, 120, 200), np.uint8)
    bg = app.BackgroundModel(scene.shape)
    mask = np.zeros((h, w), np.float32)
    mask[30:90, 60:100] = 1.0
    frame = scene.copy()
    frame[30:90, 60:100] = (255, 255, 255)  # the "person"

    # person on the left for a while, then on the right -> whole plate gets seen
    left = np.roll(mask, -40, axis=1)
    right = np.roll(mask, 40, axis=1)
    for m in (left, right):
        f = scene.copy()
        f[m > 0.5] = (255, 255, 255)
        for _ in range(12):
            bg.update(f, m)
    assert bg.coverage == 1.0

    out = app.composite(frame, bg.filled, mask, 1.0, 0.0)
    assert np.abs(out.astype(int) - scene.astype(int)).max() <= 2
    assert np.array_equal(app.composite(frame, bg.filled, mask, 0.0, 0.0), frame)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(tests)} passed")
