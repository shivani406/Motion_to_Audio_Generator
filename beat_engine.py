"""
beat_engine.py
==============
Synthesised drum kit + three selectable background beat patterns.

Patterns are 16-step (one bar of 16th notes) strings per drum:
    'x' = full-strength hit,  'o' = soft hit,  '.' = rest
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.signal import lfilter

import config

SR = config.SAMPLE_RATE


@dataclass(frozen=True)
class BeatPattern:
    name: str
    bpm: int
    swing: float                # fraction of a 16th note by which odd steps are delayed
    rows: Dict[str, str]        # drum name -> 16-char pattern


BEAT_PATTERNS: List[BeatPattern] = [
    BeatPattern(
        "Lo-Fi Chill", 78, 0.16,
        {
            "kick": "x.....x...x.....",
            "snare": "....x.......x...",
            "hat": "x.o.x.o.x.o.x.o.",
        },
    ),
    BeatPattern(
        "Soft House", 108, 0.0,
        {
            "kick": "x...x...x...x...",
            "clap": "....o.......o...",
            "ohat": "..x...x...x...x.",
            "hat": ".o.o.o.o.o.o.o.o",
        },
    ),
    BeatPattern(
        "Dream Heartbeat", 66, 0.08,
        {
            "kick": "x.x.......x.....",
            "shaker": "x.x.x.x.x.x.x.x.",
            "rim": "............o...",
        },
    ),
]

DRUM_LEVELS = {"kick": 0.95, "snare": 0.55, "clap": 0.50, "hat": 0.28,
               "ohat": 0.30, "shaker": 0.22, "rim": 0.35}


def _peak(x: np.ndarray) -> np.ndarray:
    return x / max(1e-9, float(np.max(np.abs(x))))


def _highpass(noise: np.ndarray, coeff: float = 0.7) -> np.ndarray:
    return noise - lfilter([1.0 - coeff], [1.0, -coeff], noise)


def _build_kit() -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    kit: Dict[str, np.ndarray] = {}

    # Kick: pitch-dropping sine + a touch of 2nd harmonic (audible on laptop speakers).
    t = np.arange(int(0.42 * SR)) / SR
    freq = 48.0 + 95.0 * np.exp(-t / 0.035)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * np.exp(-t / 0.16) + 0.25 * np.sin(2 * phase) * np.exp(-t / 0.08)
    click = np.sin(2 * np.pi * 1800 * t) * np.exp(-t / 0.004) * 0.15
    kit["kick"] = _peak(np.tanh(1.5 * (body + click)))

    # Snare: tone + filtered noise.
    t = np.arange(int(0.30 * SR)) / SR
    noise = _highpass(rng.standard_normal(len(t)), 0.5)
    kit["snare"] = _peak(0.7 * noise * np.exp(-t / 0.10) + 0.5 * np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.06))

    # Clap: three quick noise bursts + tail.
    t = np.arange(int(0.28 * SR)) / SR
    noise = _highpass(rng.standard_normal(len(t)), 0.6)
    env = sum(np.exp(-np.maximum(t - d, 0) / 0.012) * (t >= d) for d in (0.0, 0.011, 0.022))
    env = env / 3.0 + 0.6 * np.exp(-t / 0.09)
    kit["clap"] = _peak(noise * env)

    # Hats.
    t = np.arange(int(0.07 * SR)) / SR
    kit["hat"] = _peak(_highpass(rng.standard_normal(len(t)), 0.85) * np.exp(-t / 0.018))
    t = np.arange(int(0.40 * SR)) / SR
    kit["ohat"] = _peak(_highpass(rng.standard_normal(len(t)), 0.85) * np.exp(-t / 0.13))

    # Shaker: soft attack, short decay.
    t = np.arange(int(0.13 * SR)) / SR
    kit["shaker"] = _peak(_highpass(rng.standard_normal(len(t)), 0.8)
                          * (1 - np.exp(-t / 0.007)) * np.exp(-t / 0.05))

    # Rim: short woody tone.
    t = np.arange(int(0.09 * SR)) / SR
    kit["rim"] = _peak(np.sin(2 * np.pi * 1650 * t) * np.exp(-t / 0.012)
                       + 0.4 * _highpass(rng.standard_normal(len(t)), 0.7) * np.exp(-t / 0.006))
    return kit


class BeatPlayer:
    """Plays drum hits scheduled with sample-accurate offsets inside a block."""

    def __init__(self):
        self._kit = _build_kit()
        self._hits = []          # [buffer, position, offset, gain]

    def trigger(self, drum: str, velocity: float, offset: int = 0) -> None:
        buf = self._kit[drum]
        self._hits.append([buf, 0, int(offset), velocity * DRUM_LEVELS[drum] * config.DRUM_LEVEL])

    def clear(self) -> None:
        self._hits.clear()

    def render(self, frames: int) -> np.ndarray:
        out = np.zeros(frames)
        alive = []
        for buf, pos, off, gain in self._hits:
            if off >= frames:
                alive.append([buf, pos, off - frames, gain])
                continue
            n = min(len(buf) - pos, frames - off)
            out[off:off + n] += buf[pos:pos + n] * gain
            pos += n
            if pos < len(buf):
                alive.append([buf, pos, 0, gain])
        self._hits = alive
        return out
