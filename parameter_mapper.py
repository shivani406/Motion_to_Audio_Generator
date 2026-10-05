"""
parameter_mapper.py
===================
Translates each hand's fingertip into musical parameters:

    * vertical Y   -> quantized scale note (snaps to the current scale)
    * horizontal X -> stereo pan + filter brightness
    * speed        -> gain (zero below a threshold => no idle buzzing)
    * note events  -> `trigger` flag when a new note should start
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import config


@dataclass
class MappedParams:
    note_index: int
    note_name: str
    frequency: float
    gain: float
    velocity: float
    active: bool          # hand is playing right now (moving fast enough)
    trigger: bool         # a new note-on should be fired this frame
    pan: float            # -0.7 .. 0.7
    brightness: float     # 0 (hand at left edge) .. 1 (right edge)


class HandMapper:
    """Mapping state for one hand."""

    def __init__(self, freqs: Sequence[float], names: Sequence[str]):
        self._top = config.FRAME_HEIGHT * config.Y_MARGIN_FRACTION
        self._bottom = config.FRAME_HEIGHT * (1.0 - config.Y_MARGIN_FRACTION)
        self._note_index = 0
        self._prev: Optional[Tuple[int, int]] = None
        self._velocity = 0.0
        self._active = False
        self.set_scale(freqs, names)

    def set_scale(self, freqs: Sequence[float], names: Sequence[str]) -> None:
        self._freqs = np.asarray(freqs, dtype=np.float64)
        self._names = list(names)
        self._note_index = int(np.clip(self._note_index, 0, len(self._freqs) - 1))

    def reset(self) -> None:
        self._prev = None
        self._velocity = 0.0
        self._active = False

    def update(self, point: Optional[Tuple[int, int]], playable: bool = True) -> MappedParams:
        if point is None or not playable:
            self.reset()
            return self._build(0.0, False, False, 0.5)

        x, y = point
        n = len(self._freqs)

        # ---- Quantized pitch with hysteresis (top of frame = highest note) -- #
        raw = float(np.interp(y, [self._top, self._bottom], [n - 1, 0]))
        changed = False
        if abs(raw - self._note_index) > 0.5 + config.NOTE_HYSTERESIS:
            new = int(np.clip(round(raw), 0, n - 1))
            changed = new != self._note_index
            self._note_index = new

        # ---- Velocity: Euclidean distance between consecutive frames ------- #
        inst = 0.0 if self._prev is None else math.hypot(x - self._prev[0], y - self._prev[1])
        self._prev = (x, y)
        s = config.VELOCITY_SMOOTHING
        self._velocity = s * inst + (1.0 - s) * self._velocity

        # ---- Activity with hysteresis so the note doesn't chatter ---------- #
        thr = config.VELOCITY_THRESHOLD * (0.6 if self._active else 1.0)
        now_active = self._velocity >= thr
        trigger = now_active and (not self._active or changed)
        self._active = now_active

        gain = 0.0
        if now_active:
            t = (self._velocity - config.VELOCITY_THRESHOLD * 0.6) / (
                config.VELOCITY_MAX - config.VELOCITY_THRESHOLD * 0.6)
            t = float(np.clip(t, 0.0, 1.0)) ** config.GAIN_CURVE_EXPONENT
            gain = config.MIN_GAIN + t * (config.MAX_GAIN - config.MIN_GAIN)

        return self._build(gain, now_active, trigger, x / config.FRAME_WIDTH)

    def _build(self, gain: float, active: bool, trigger: bool, x_norm: float) -> MappedParams:
        i = self._note_index
        x_norm = float(np.clip(x_norm, 0.0, 1.0))
        return MappedParams(
            note_index=i, note_name=self._names[i], frequency=float(self._freqs[i]),
            gain=float(gain), velocity=float(self._velocity), active=active, trigger=trigger,
            pan=float(np.clip((x_norm - 0.5) * 1.4, -0.7, 0.7)), brightness=x_norm,
        )


class ParameterMapper:
    """One HandMapper per hand slot, sharing a scale."""

    def __init__(self, freqs: Sequence[float], names: Sequence[str], num_slots: int = 2):
        self._top = config.FRAME_HEIGHT * config.Y_MARGIN_FRACTION
        self._bottom = config.FRAME_HEIGHT * (1.0 - config.Y_MARGIN_FRACTION)
        self._mappers = [HandMapper(freqs, names) for _ in range(num_slots)]
        self.n_notes = len(freqs)

    def set_scale(self, freqs: Sequence[float], names: Sequence[str]) -> None:
        self.n_notes = len(freqs)
        for m in self._mappers:
            m.set_scale(freqs, names)

    def update(self, slot: int, point: Optional[Tuple[int, int]], playable: bool = True) -> MappedParams:
        return self._mappers[slot].update(point, playable)

    def reset(self, slot: int) -> None:
        self._mappers[slot].reset()

    def y_for_index(self, index: int) -> float:
        """Pixel row at which scale step `index` sits (used by the HUD pitch ladder)."""
        return float(np.interp(index, [self.n_notes - 1, 0], [self._top, self._bottom]))
