"""
particle_engine.py
==================
Glowing particle ribbon renderer (supports two hands, one colour family each).

    * Frame-to-frame linear interpolation (lerp) fills the gap between the
      previous and current fingertip positions -> continuous, gapless ribbon.
    * The float canvas is multiplied by CANVAS_FADE_FACTOR every frame ->
      exponential decay (fading watercolor trail).
"""

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import config


class ParticleEngine:
    def __init__(self):
        self._h = config.FRAME_HEIGHT
        self._w = config.FRAME_WIDTH
        # float32 so repeated multiplication decays smoothly (no stuck low values)
        self._canvas = np.zeros((self._h, self._w, 3), dtype=np.float32)
        self._prev: Dict[int, Optional[Tuple[int, int]]] = {}
        self._rng = np.random.default_rng()

    def clear(self) -> None:
        self._canvas[:] = 0.0
        self._prev.clear()

    def update(self, trails: Dict[int, Tuple[Tuple[int, int], int, int, float]]) -> None:
        """
        trails: slot -> (point, note_index, n_notes, gain) for every visible hand.
        Slots that are absent lose their ribbon head (the trail just fades out).
        """
        # ---- Exponential matrix decay ------------------------------------ #
        self._canvas *= config.CANVAS_FADE_FACTOR
        cv2.threshold(self._canvas, config.CANVAS_CUTOFF, 0, cv2.THRESH_TOZERO, dst=self._canvas)

        for slot in list(self._prev):
            if slot not in trails:
                self._prev[slot] = None
        if not trails:
            return

        layer = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        for slot, (point, note_index, n_notes, gain) in trails.items():
            self._draw_slot(layer, slot, point, note_index, n_notes, gain)

        # Glow at half resolution (4x cheaper), then core + glow into the canvas.
        small = cv2.resize(layer, (self._w // 2, self._h // 2), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), config.GLOW_SIGMA / 2.0)
        glow = cv2.resize(small, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
        added = cv2.addWeighted(glow, config.GLOW_STRENGTH, layer, config.CORE_STRENGTH, 0)
        np.add(self._canvas, added.astype(np.float32), out=self._canvas)
        np.minimum(self._canvas, 255.0, out=self._canvas)

    def _draw_slot(self, layer, slot, point, note_index, n_notes, gain) -> None:
        lo, hi = config.HUE_RANGES[slot % len(config.HUE_RANGES)]
        frac = note_index / max(1, n_notes - 1)
        hue = int(lo + frac * (hi - lo))
        intensity = config.PARTICLE_MIN_INTENSITY + (1.0 - config.PARTICLE_MIN_INTENSITY) * gain
        hsv = np.uint8([[[hue, 200, int(255 * intensity)]]])
        color = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
        radius = int(round(config.PARTICLE_MIN_RADIUS
                           + gain * (config.PARTICLE_MAX_RADIUS - config.PARTICLE_MIN_RADIUS)))

        x1, y1 = point
        prev = self._prev.get(slot)
        if prev is None:
            samples = [(x1, y1)]
        else:
            dist = math.hypot(x1 - prev[0], y1 - prev[1])
            count = int(min(config.MAX_INTERPOLATED_PARTICLES,
                            max(1, math.ceil(dist / config.PARTICLE_SPACING))))
            ts = np.linspace(1.0 / count, 1.0, count)
            samples = [(int(round(prev[0] + (x1 - prev[0]) * t)),
                        int(round(prev[1] + (y1 - prev[1]) * t))) for t in ts]
        self._prev[slot] = point

        for px, py in samples:
            cv2.circle(layer, (px, py), radius, color, -1, cv2.LINE_AA)
        if gain > 0.0:
            for _ in range(config.SPARKLES_PER_FRAME):
                ox, oy = self._rng.normal(0.0, radius * 1.8, size=2)
                cv2.circle(layer, (int(x1 + ox), int(y1 + oy)), 1, (255, 255, 255), -1, cv2.LINE_AA)

    def render(self) -> np.ndarray:
        """Return the trail canvas as a uint8 BGR image."""
        return self._canvas.astype(np.uint8)
