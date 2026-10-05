"""
gesture_recognizer.py
=====================
Classifies a hand pose from the 21 MediaPipe landmarks and turns *held* poses
into one-shot actions.

Fixed gestures (hold for ~0.7 s):
    Open palm   -> next background beat   (Off -> Lo-Fi -> House -> Heartbeat)
    Peace sign  -> next instrument sound  (Pad -> Pluck -> Bell)
    Thumbs up   -> next scale / key mood
    Fist        -> toggle backing chord pad
    Rock sign   -> next reverb space      (Room -> Hall -> Cathedral)

Playing pose: point with the index finger (other poses are silent so that
gestures never make stray notes).
"""

import math
from collections import Counter, deque
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import config

Point = Tuple[float, float]


class Pose(Enum):
    NONE = "none"
    POINT = "point"
    OTHER = "other"
    FIST = "fist"
    PALM = "palm"
    PEACE = "peace"
    THUMBS_UP = "thumbs up"
    ROCK = "rock"


# pose -> (action id, label shown on the HUD)
POSE_ACTIONS: Dict[Pose, Tuple[str, str]] = {
    Pose.PALM: ("beat", "Next beat"),
    Pose.PEACE: ("patch", "Next sound"),
    Pose.THUMBS_UP: ("scale", "Next scale"),
    Pose.FIST: ("chords", "Chords on/off"),
    Pose.ROCK: ("space", "Next space"),
}

PLAYABLE_POSES = (Pose.POINT, Pose.OTHER)

_TIPS = (8, 12, 16, 20)
_PIPS = (6, 10, 14, 18)


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def classify_pose(lm: Sequence[Point]) -> Pose:
    """lm: 21 (x, y) landmark positions in pixels."""
    wrist = lm[0]
    palm = max(_dist(lm[0], lm[9]), 1e-6)

    idx, mid, ring, pinky = (
        _dist(lm[tip], wrist) > config.FINGER_EXTEND_RATIO * _dist(lm[pip], wrist)
        for tip, pip in zip(_TIPS, _PIPS)
    )
    thumb = _dist(lm[4], lm[5]) > config.THUMB_EXTEND_RATIO * palm
    thumb_up = thumb and lm[4][1] < lm[5][1] - 0.4 * palm       # image y grows downward

    if thumb_up and not (idx or mid or ring or pinky):
        return Pose.THUMBS_UP
    if idx and mid and ring and pinky:
        return Pose.PALM
    if idx and mid and not ring and not pinky:
        return Pose.PEACE
    if idx and pinky and not mid and not ring:
        return Pose.ROCK
    if not (idx or mid or ring or pinky):
        return Pose.FIST
    if idx and not (mid or ring or pinky):
        return Pose.POINT
    return Pose.OTHER


class GestureTracker:
    """Per-hand pose smoothing (majority vote) and hold-to-fire logic."""

    def __init__(self):
        self._votes: Dict[int, deque] = {}
        self._state: Dict[int, dict] = {}

    def reset(self, slot: int) -> None:
        self._votes.pop(slot, None)
        self._state.pop(slot, None)

    def update(self, slot: int, raw_pose: Pose, now: float) -> Tuple[Optional[str], Pose, float]:
        """
        Returns (fired_action or None, stable_pose, hold_progress 0..1).
        """
        votes = self._votes.setdefault(slot, deque(maxlen=config.GESTURE_VOTE_FRAMES))
        votes.append(raw_pose)
        pose = Counter(votes).most_common(1)[0][0]

        st = self._state.get(slot)
        if st is None or st["pose"] != pose:
            st = {"pose": pose, "since": now, "fired": False}
            self._state[slot] = st

        if pose not in POSE_ACTIONS:
            return None, pose, 0.0
        if st["fired"]:
            return None, pose, 1.0

        progress = min(1.0, (now - st["since"]) / config.GESTURE_HOLD_SECONDS)
        if progress >= 1.0:
            st["fired"] = True
            return POSE_ACTIONS[pose][0], pose, 1.0
        return None, pose, progress
