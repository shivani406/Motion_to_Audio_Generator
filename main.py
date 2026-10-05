"""
main.py
=======
Application loop for the Motion-Driven Ambient Audio Sculptor.

    webcam -> mirror -> HandTracker (1-2 hands) -> pose/gestures + ParameterMapper
                                                      |-> AudioEngine (music)
                                                      +-> ParticleEngine -> HUD -> window

Keys
    q / Esc  quit              c  clear canvas         m  mute
    h  toggle HUD              0  beat off             1 2 3  choose beat
    + / -    tempo             p  next sound           s  next scale
    k        chord pad on/off  r  next reverb space    z  rhythm lock on/off
    t        one / two hands   g  gestures on/off

Gestures (hold ~0.7 s): open palm = next beat, peace = next sound,
thumbs up = next scale, fist = chord pad, rock sign = next reverb space.
Play by POINTING with your index finger and moving it.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2

import config
from audio_engine import AudioEngine
from gesture_recognizer import PLAYABLE_POSES, POSE_ACTIONS, GestureTracker, Pose, classify_pose
from motion_detector import HandTracker
from music_theory import build_scale
from parameter_mapper import MappedParams, ParameterMapper
from particle_engine import ParticleEngine

SLOT_COLORS = {0: (255, 220, 120), 1: (120, 170, 255)}      # BGR: cyan-ish / orange-ish


@dataclass
class HandView:
    slot: int
    point: Tuple[int, int]
    wrist: Tuple[int, int]
    params: MappedParams
    pose: Pose
    progress: float


# ---------------------------------------------------------------------- #
# Camera
# ---------------------------------------------------------------------- #
def open_camera(index: int) -> cv2.VideoCapture:
    cap = None
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)      # much faster start-up on Windows
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {index}. Close other apps using the "
            f"webcam or try:  python main.py --camera 1"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                   # keep latency low
    return cap


# ---------------------------------------------------------------------- #
# HUD
# ---------------------------------------------------------------------- #
def put_text(img, text, org, scale=0.48, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def on_off(flag: bool) -> str:
    return "ON" if flag else "off"


def draw_hud(img, st, views: Dict[int, HandView], mapper, fps, frame_ms,
             toast, audio_ok, two_hands, gestures_on):
    h, w = img.shape[:2]

    # Pitch ladder (right edge): one tick per scale step, lit per playing hand.
    lit = {v.params.note_index: SLOT_COLORS[v.slot] for v in views.values() if v.params.active}
    for i in range(mapper.n_notes):
        y = int(mapper.y_for_index(i))
        color = lit.get(i, (110, 110, 110))
        cv2.line(img, (w - 34, y), (w - 8, y), color, 3 if i in lit else 1, cv2.LINE_AA)
    for v in views.values():
        if v.params.active:
            put_text(img, v.params.note_name, (w - 76, int(mapper.y_for_index(v.params.note_index)) + 4),
                     0.42, SLOT_COLORS[v.slot])

    # Info block (top-left).
    fps_color = (0, 255, 0) if frame_ms <= config.TARGET_FRAME_MS else (0, 140, 255)
    put_text(img, f"FPS {fps:4.1f}  frame {frame_ms:4.1f} ms", (10, 20), 0.48, fps_color)
    put_text(img, f"Sound: {st['patch_name']}   Scale: {st['scale_name']}", (10, 40))
    put_text(img, f"Beat: {st['beat_name']}  {st['bpm']:.0f} BPM   Chords: {on_off(st['chord_on'])}", (10, 60))
    put_text(img, f"Space: {st['space_name']}   Rhythm lock: {on_off(st['quantize'])}", (10, 80))
    put_text(img, f"Hands: {2 if two_hands else 1}   Gestures: {on_off(gestures_on)}", (10, 100))

    # Per-hand note + gain meters.
    for slot, v in sorted(views.items()):
        y = 124 + 22 * slot
        c = SLOT_COLORS[slot]
        put_text(img, f"{'L' if slot == 0 else 'R'} {v.params.note_name:>3} {v.params.frequency:6.1f}Hz", (10, y), 0.45, c)
        cv2.rectangle(img, (170, y - 10), (270, y), (80, 80, 80), 1)
        cv2.rectangle(img, (170, y - 10), (170 + int(100 * v.params.gain), y), c, -1)

    # Fingertip cursors, pose labels and gesture hold arcs.
    for v in views.values():
        c = SLOT_COLORS[v.slot]
        cv2.circle(img, v.point, 6, (255, 255, 255), 1, cv2.LINE_AA)
        if gestures_on and v.pose in POSE_ACTIONS:
            label = POSE_ACTIONS[v.pose][1]
            put_text(img, label, (v.wrist[0] - 40, v.wrist[1] + 46), 0.5, c, 1)
            if v.progress > 0:
                cv2.ellipse(img, v.wrist, (26, 26), 0, -90, -90 + int(360 * v.progress), c, 3, cv2.LINE_AA)
    if not views:
        put_text(img, "Show a hand and POINT with your index finger", (w // 2 - 205, h // 2), 0.62, (200, 200, 255))

    status = []
    if st["muted"]:
        status.append("MUTED")
    if not audio_ok:
        status.append("NO AUDIO DEVICE")
    if status:
        put_text(img, " | ".join(status), (10, h - 64), 0.6, (80, 80, 255), 2)

    if toast:
        put_text(img, toast, (max(10, w // 2 - 9 * len(toast)), 40), 0.8, (255, 255, 255), 2)

    put_text(img, "Palm=beat Peace=sound Thumb=scale Fist=chords Rock=space", (10, h - 40), 0.42, (190, 255, 190))
    put_text(img, "1-3 beat  p sound  s scale  k chords  r space  z lock  +/- tempo", (10, h - 24), 0.40, (200, 200, 200))
    put_text(img, "t 1/2 hands  g gestures  m mute  c clear  h hud  q quit", (10, h - 8), 0.40, (200, 200, 200))


# ---------------------------------------------------------------------- #
# Main loop
# ---------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Motion-Driven Ambient Audio Sculptor")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="webcam index (default 0)")
    p.add_argument("--no-audio", action="store_true", help="run the visuals only")
    p.add_argument("--one-hand", action="store_true", help="start in single-hand mode")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cap = None
    tracker = None
    audio = AudioEngine()
    audio_ok = False

    try:
        cap = open_camera(args.camera)
        tracker = HandTracker()
        freqs, names = build_scale(0)
        mapper = ParameterMapper(freqs, names)
        particles = ParticleEngine()
        gestures = GestureTracker()

        if not args.no_audio:
            audio_ok = audio.start()

        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        show_hud = True
        two_hands = not args.one_hand
        gestures_on = True
        toast, toast_until = "", 0.0
        fps, prev_time = 0.0, time.perf_counter()
        print("Running. Point with your index finger and move it. Press 'q' to quit.")

        def apply_action(action: str) -> str:
            if action == "beat":
                audio.next_beat()
                s = audio.status()
                return f"Beat: {s['beat_name']}"
            if action == "patch":
                audio.next_patch()
                return f"Sound: {audio.status()['patch_name']}"
            if action == "scale":
                idx = audio.next_scale()
                f, n = build_scale(idx)
                mapper.set_scale(f, n)
                return f"Scale: {audio.status()['scale_name']}"
            if action == "chords":
                return f"Chords {on_off(audio.toggle_chords())}"
            if action == "space":
                audio.next_space()
                return f"Space: {audio.status()['space_name']}"
            return ""

        while True:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("[camera] Failed to read a frame - exiting.")
                break

            frame = cv2.flip(frame, 1)                      # mirror
            if frame.shape[1] != config.FRAME_WIDTH or frame.shape[0] != config.FRAME_HEIGHT:
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

            # --- Perception ------------------------------------------------ #
            hands = tracker.detect(frame, two_hands=two_hands)
            seen = {h.slot for h in hands}
            now = time.monotonic()
            views: Dict[int, HandView] = {}
            trails = {}

            for slot in (0, 1):
                if slot not in seen:
                    gestures.reset(slot)

            for hand in hands:
                raw_pose = classify_pose(hand.landmarks) if gestures_on else Pose.POINT
                action, pose, progress = gestures.update(hand.slot, raw_pose, now)
                if gestures_on and action:
                    toast, toast_until = apply_action(action), now + 1.6

                playable = (not gestures_on) or pose in PLAYABLE_POSES
                p = mapper.update(hand.slot, hand.tip, playable)
                views[hand.slot] = HandView(hand.slot, hand.tip, hand.wrist, p, pose, progress)
                trails[hand.slot] = (hand.tip, p.note_index, mapper.n_notes, p.gain if p.active else 0.0)

            # --- Sound ----------------------------------------------------- #
            for slot in (0, 1):
                if slot in views:
                    p = views[slot].params
                    audio.update_hand(slot, p.active, p.frequency, p.gain, p.pan, p.brightness, p.trigger)
                else:
                    mapper.reset(slot)
                    audio.update_hand(slot, False, freqs[0], 0.0, 0.0, 0.5, False)

            # --- Visuals --------------------------------------------------- #
            particles.update(trails)
            dimmed = cv2.convertScaleAbs(frame, alpha=config.VIDEO_DIM)
            output = cv2.add(dimmed, particles.render())

            t_now = time.perf_counter()
            frame_ms = (t_now - loop_start) * 1000.0
            inst_fps = 1.0 / max(t_now - prev_time, 1e-6)
            prev_time = t_now
            fps = inst_fps if fps == 0.0 else 0.9 * fps + 0.1 * inst_fps

            if show_hud:
                draw_hud(output, audio.status(), views, mapper, fps, frame_ms,
                         toast if now < toast_until else "", audio_ok, two_hands, gestures_on)
            cv2.imshow(config.WINDOW_NAME, output)

            # --- Keyboard -------------------------------------------------- #
            key = cv2.waitKey(1) & 0xFF
            msg = ""
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                particles.clear()
            elif key == ord("m"):
                msg = "Muted" if audio.toggle_mute() else "Unmuted"
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("0"):
                audio.set_beat(0); msg = "Beat: Off"
            elif key in (ord("1"), ord("2"), ord("3")):
                audio.set_beat(int(chr(key))); msg = f"Beat: {audio.status()['beat_name']}"
            elif key == ord("p"):
                msg = apply_action("patch")
            elif key == ord("s"):
                msg = apply_action("scale")
            elif key == ord("k"):
                msg = apply_action("chords")
            elif key == ord("r"):
                msg = apply_action("space")
            elif key == ord("z"):
                msg = f"Rhythm lock {on_off(audio.toggle_quantize())}"
            elif key in (ord("+"), ord("=")):
                msg = f"Tempo {audio.adjust_bpm(+4):.0f} BPM"
            elif key in (ord("-"), ord("_")):
                msg = f"Tempo {audio.adjust_bpm(-4):.0f} BPM"
            elif key == ord("t"):
                two_hands = not two_hands
                msg = "Two-hand mode" if two_hands else "One-hand mode"
            elif key == ord("g"):
                gestures_on = not gestures_on
                msg = f"Gestures {on_off(gestures_on)}"
            if msg:
                toast, toast_until = msg, time.monotonic() + 1.6

            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[error] {exc}")
        return 1
    finally:
        audio.stop()
        if tracker is not None:
            tracker.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Goodbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
