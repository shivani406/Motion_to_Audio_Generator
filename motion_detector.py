"""
motion_detector.py
==================
Thin wrapper around the MediaPipe Tasks `HandLandmarker` (RunningMode.VIDEO).

Tracks up to two hands. Each hand gets a stable "slot" (0 or 1) so the left
and right hand keep their own voice, colour and smoothing from frame to frame.
Returns the index fingertip (landmark #8) in pixel coordinates plus all 21
landmarks (used by the gesture recogniser).
"""

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config


def ensure_model(path: Path = config.MODEL_PATH) -> Path:
    """Make sure hand_landmarker.task exists; download it if it is missing."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path

    print(f"[model] {path.name} not found - downloading from Google ...")
    try:
        urllib.request.urlretrieve(config.MODEL_URL, str(path))
    except Exception as exc:
        if path.exists():
            path.unlink()
        raise FileNotFoundError(
            f"Could not download the MediaPipe model ({exc}).\n"
            f"Download it manually from:\n  {config.MODEL_URL}\n"
            f"and save it as:\n  {path}"
        ) from exc
    print("[model] Download complete.")
    return path


@dataclass
class HandInfo:
    slot: int                                  # 0 = left side of screen, 1 = right side
    tip: Tuple[int, int]                       # smoothed index fingertip (px)
    wrist: Tuple[int, int]
    landmarks: List[Tuple[float, float]]       # 21 landmarks in pixels


class HandTracker:
    """Detects up to two hands and exposes the index fingertips as (x, y) pixels."""

    STALE_SECONDS = 0.5

    def __init__(self, model_path: Path = config.MODEL_PATH):
        model_path = ensure_model(model_path)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1
        self._smoothed = {0: None, 1: None}
        self._last_wrist = {0: None, 1: None}      # slot -> (x, y, time)

    # ------------------------------------------------------------------ #
    def _assign_slots(self, wrists: List[Tuple[float, float]], now: float, two_hands: bool) -> List[int]:
        if not two_hands:
            return [0] * len(wrists)
        if len(wrists) >= 2:
            order = sorted(range(len(wrists)), key=lambda i: wrists[i][0])
            slots = [0] * len(wrists)
            slots[order[0]], slots[order[1]] = 0, 1
            return slots

        # One hand visible: keep it on the slot it was last seen in.
        x, y = wrists[0]
        best, best_d = None, float("inf")
        for s in (0, 1):
            last = self._last_wrist[s]
            if last is not None and now - last[2] < self.STALE_SECONDS:
                d = (x - last[0]) ** 2 + (y - last[1]) ** 2
                if d < best_d:
                    best, best_d = s, d
        if best is None:
            best = 0 if x < config.FRAME_WIDTH / 2 else 1
        return [best]

    def detect(self, frame_bgr, two_hands: bool = True) -> List[HandInfo]:
        """Run inference on a mirrored BGR frame. Returns hands sorted by slot."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        now = time.monotonic()
        timestamp_ms = int((now - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:          # VIDEO mode needs strictly increasing ms
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        detections = result.hand_landmarks or []
        if not two_hands:
            detections = detections[:1]

        all_lms, wrists = [], []
        for hand in detections:
            pts = [(min(max(p.x, 0.0), 1.0) * config.FRAME_WIDTH,
                    min(max(p.y, 0.0), 1.0) * config.FRAME_HEIGHT) for p in hand]
            all_lms.append(pts)
            wrists.append(pts[0])

        slots = self._assign_slots(wrists, now, two_hands) if wrists else []
        seen = set(slots)
        for s in (0, 1):
            if s not in seen:
                self._smoothed[s] = None                      # forget history of vanished hands

        hands: List[HandInfo] = []
        a = config.LANDMARK_SMOOTHING
        for lms, slot in zip(all_lms, slots):
            x, y = lms[config.INDEX_FINGERTIP_ID]
            prev = self._smoothed[slot]
            sm = (x, y) if prev is None else (a * x + (1 - a) * prev[0], a * y + (1 - a) * prev[1])
            self._smoothed[slot] = sm
            self._last_wrist[slot] = (lms[0][0], lms[0][1], now)
            hands.append(HandInfo(slot, (int(round(sm[0])), int(round(sm[1]))),
                                  (int(lms[0][0]), int(lms[0][1])), lms))
        hands.sort(key=lambda h: h.slot)
        return hands

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
