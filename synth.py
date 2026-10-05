"""
synth.py
========
Instrument voices.

Why this sounds musical rather than "buzzy":
    * Notes have real envelopes (attack / decay / release) instead of a drone.
    * Plucks and bells use partials that decay at different speeds, exactly
      like struck strings / metal; the pad uses detuned, soft partials.
    * Every voice runs through a 2-pole low-pass filter (no harsh highs),
      with the cutoff controlled by the hand's horizontal position.
    * Vibrato fades in after the note starts, like a real player's.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import lfilter

import config

TWO_PI = 2.0 * math.pi
SR = float(config.SAMPLE_RATE)
_RNG = np.random.default_rng(7)


@dataclass(frozen=True)
class Patch:
    name: str
    ratios: Tuple[float, ...]          # partial frequency multipliers
    amps: Tuple[float, ...]            # partial amplitudes
    detune_cents: Tuple[float, ...]    # per-partial detune (chorus / warmth)
    decays: Tuple[float, ...]          # per-partial decay rate (1/s); 0 = sustained
    attack: float                      # seconds
    release: float                     # seconds
    vibrato_depth: float               # fractional frequency deviation
    vibrato_delay: float               # seconds before vibrato reaches full depth
    glide: float                       # portamento coefficient per audio block (0 = none)
    cutoff: float                      # base low-pass cutoff in Hz
    sustained: bool                    # True: holds while gated (pad); False: rings & decays


WARM_PAD = Patch(
    "Warm Pad",
    ratios=(1.0, 1.0, 2.0, 3.0, 0.5),
    amps=(1.0, 1.0, 0.30, 0.10, 0.25),
    detune_cents=(-6.0, 6.0, 0.0, 0.0, 0.0),
    decays=(0.0, 0.0, 0.0, 0.0, 0.0),
    attack=0.10, release=0.70,
    vibrato_depth=0.003, vibrato_delay=0.5,
    glide=0.12, cutoff=2200.0, sustained=True,
)

SOFT_PLUCK = Patch(
    "Soft Pluck",
    ratios=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    amps=(1.0, 0.55, 0.30, 0.18, 0.10, 0.06),
    detune_cents=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    decays=(2.2, 4.5, 7.0, 10.0, 14.0, 18.0),
    attack=0.004, release=0.35,
    vibrato_depth=0.0, vibrato_delay=0.0,
    glide=0.0, cutoff=4500.0, sustained=False,
)

GLASS_BELL = Patch(
    "Glass Bell",
    ratios=(1.0, 2.0, 2.76, 4.07, 5.42),
    amps=(1.0, 0.50, 0.35, 0.20, 0.12),
    detune_cents=(0.0, 0.0, 0.0, 0.0, 0.0),
    decays=(1.1, 1.8, 2.6, 3.6, 5.0),
    attack=0.003, release=0.50,
    vibrato_depth=0.0, vibrato_delay=0.0,
    glide=0.0, cutoff=7000.0, sustained=False,
)

CHORD_PAD = Patch(
    "Chord Pad",
    ratios=(1.0, 1.0, 2.0, 0.5),
    amps=(1.0, 1.0, 0.25, 0.20),
    detune_cents=(-8.0, 8.0, 0.0, 0.0),
    decays=(0.0, 0.0, 0.0, 0.0),
    attack=0.80, release=1.40,
    vibrato_depth=0.0025, vibrato_delay=0.0,
    glide=0.0, cutoff=1300.0, sustained=True,
)

# The patches the player can choose between.
PLAYABLE_PATCHES = [WARM_PAD, SOFT_PLUCK, GLASS_BELL]


class Voice:
    """One sounding note (stateful across audio blocks)."""

    def __init__(self, patch: Patch, freq: float, level: float, pan: float = 0.0,
                 cutoff: Optional[float] = None, start_delay: int = 0, gate: bool = True):
        self.patch = patch
        self.freq = float(freq)
        self.target_freq = float(freq)
        self.level = float(level)
        self.gate = gate
        self.pan = float(pan)
        self.cutoff = float(cutoff if cutoff is not None else patch.cutoff)
        self.dead = False

        self.age = -int(start_delay)       # samples since note-on (negative = delayed start)
        self.env = 0.0
        self._rel = 1.0
        self._pan_now = float(pan)
        self._cut_now = self.cutoff
        self._vib_phase = 0.0

        k = len(patch.ratios)
        cents = np.asarray(patch.detune_cents, dtype=np.float64)
        self._mult = (np.asarray(patch.ratios, dtype=np.float64) * 2.0 ** (cents / 1200.0))[:, None]
        self._amps = np.asarray(patch.amps, dtype=np.float64)[:, None]
        self._decays = np.asarray(patch.decays, dtype=np.float64)[:, None]
        self._has_decay = bool(np.any(self._decays > 0))
        self._min_decay = float(np.min(self._decays)) if self._has_decay else 0.0
        self._norm = 1.0 / float(np.sum(self._amps))
        self._phases = _RNG.uniform(0.0, TWO_PI, k) if patch.sustained else np.zeros(k)
        self._zi1 = np.zeros(1)
        self._zi2 = np.zeros(1)

    # ------------------------------------------------------------------ #
    def render(self, frames: int) -> np.ndarray:
        """Return a (2, frames) stereo block."""
        p = self.patch
        n = np.arange(frames, dtype=np.float64)
        t = (self.age + n) / SR                 # seconds since note-on
        tp = np.maximum(t, 0.0)

        # ---- Pitch: portamento glide (legato) + delayed vibrato ---------- #
        if p.glide > 0.0:
            end = self.freq + (self.target_freq - self.freq) * p.glide
        else:
            end = self.target_freq
        f = np.linspace(self.freq, end, frames, endpoint=False)
        self.freq = end

        if p.vibrato_depth > 0.0:
            step = TWO_PI * config.VIBRATO_RATE_HZ / SR
            vib = np.sin(self._vib_phase + step * (n + 1))
            self._vib_phase = (self._vib_phase + step * frames) % TWO_PI
            onset = np.clip(tp / max(p.vibrato_delay, 1e-3), 0.0, 1.0) if p.vibrato_delay > 0 else 1.0
            f = f * (1.0 + p.vibrato_depth * onset * vib)

        # ---- Additive partials (each with its own phase + decay) --------- #
        base = np.cumsum(TWO_PI * f / SR)
        ph = self._phases[:, None] + self._mult * base[None, :]
        self._phases = ph[:, -1] % TWO_PI
        partials = self._amps * np.sin(ph)
        if self._has_decay:
            partials *= np.exp(-self._decays * tp[None, :])
        wave = partials.sum(axis=0) * self._norm

        # ---- Envelope ----------------------------------------------------- #
        if p.sustained:
            target = self.level if self.gate else 0.0
            tau = max((p.attack if target > self.env else p.release) / 3.0, 1e-3)
            amp = target + (self.env - target) * np.exp(-(n + 1) / (tau * SR))
            self.env = float(amp[-1])
            if not self.gate and self.env < 1e-4:
                self.dead = True
        else:
            a = np.clip(t / max(p.attack, 1e-4), 0.0, 1.0)
            a = a * a * (3.0 - 2.0 * a)         # smoothstep attack (no click)
            amp = self.level * a
            if not self.gate:
                rel = self._rel * np.exp(-(n + 1) / (max(p.release / 3.0, 1e-3) * SR))
                self._rel = float(rel[-1])
                amp = amp * rel
                if self._rel < 1e-3:
                    self.dead = True
            if self._has_decay and (self.age + frames) / SR * self._min_decay > 6.9:
                self.dead = True                # rang out below -60 dB

        # ---- Two-pole low-pass (removes buzz / harshness) ----------------- #
        self._cut_now += (self.cutoff - self._cut_now) * 0.3
        a_lp = math.exp(-TWO_PI * min(self._cut_now, SR * 0.45) / SR)
        b, a_den = [1.0 - a_lp], [1.0, -a_lp]
        wave, self._zi1 = lfilter(b, a_den, wave, zi=self._zi1)
        wave, self._zi2 = lfilter(b, a_den, wave, zi=self._zi2)

        # ---- Equal-power pan ---------------------------------------------- #
        self._pan_now += (self.pan - self._pan_now) * 0.3
        angle = (self._pan_now + 1.0) * math.pi / 4.0
        y = wave * amp
        self.age += frames
        return np.stack([y * math.cos(angle), y * math.sin(angle)])
