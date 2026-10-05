"""
effects.py
==========
Block-based stereo effects that give the sound space and depth.

All effects use circular buffers whose delay lengths are >= the audio block
size, so a whole block can be processed with vectorised NumPy (no per-sample
Python loops).
"""

import math

import numpy as np
from scipy.signal import lfilter

import config

SR = config.SAMPLE_RATE


def _one_pole(coeff: float):
    return [1.0 - coeff], [1.0, -coeff]


class StereoDelay:
    """Ping-pong feedback delay with a darkened feedback path."""

    def __init__(self, max_seconds: float = 2.0):
        self._size = int(SR * max_seconds)
        self._buf = np.zeros((2, self._size))
        self._wp = 0
        self._delay = int(SR * 0.4)
        self.feedback = config.DELAY_FEEDBACK
        self._zi = np.zeros((2, 1))
        self._b, self._a = _one_pole(0.45)

    def set_time_seconds(self, seconds: float) -> None:
        self._delay = int(np.clip(SR * seconds, config.AUDIO_BLOCK_SIZE + 1, self._size - 1))

    def process(self, x: np.ndarray) -> np.ndarray:
        """x: (2, n) stereo in -> (2, n) wet out."""
        n = x.shape[1]
        ar = np.arange(n)
        write_idx = (self._wp + ar) % self._size
        read_idx = (self._wp - self._delay + ar) % self._size
        delayed = self._buf[:, read_idx]                       # (2, n)

        fb, self._zi = lfilter(self._b, self._a, delayed, axis=1, zi=self._zi)
        mono_in = x.sum(axis=0) * 0.5
        self._buf[0, write_idx] = mono_in + self.feedback * fb[1]   # ping-pong
        self._buf[1, write_idx] = self.feedback * fb[0]
        self._wp = (self._wp + n) % self._size
        return delayed


class _Allpass:
    """Schroeder all-pass diffuser (delay must be >= block size)."""

    def __init__(self, delay: int, g: float = 0.6):
        self.d, self.g = delay, g
        self.bx = np.zeros(delay)
        self.by = np.zeros(delay)
        self.wp = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        idx = (self.wp + np.arange(n)) % self.d
        y = -self.g * x + self.bx[idx] + self.g * self.by[idx]
        self.bx[idx] = x
        self.by[idx] = y
        self.wp = (self.wp + n) % self.d
        return y


class FDNReverb:
    """
    4-line feedback delay network (Householder mixing) with input diffusion
    and per-line damping. Produces a smooth, lush stereo tail.
    """

    DELAYS = (1031, 1327, 1523, 1801)

    def __init__(self, feedback: float = 0.88, damping: float = 0.35):
        self._bufs = [np.zeros(d) for d in self.DELAYS]
        self._wp = [0] * 4
        self._mix = 0.5 * np.ones((4, 4)) - np.eye(4)          # orthogonal Householder matrix
        self._zi = np.zeros((4, 1))
        self._b, self._a = _one_pole(damping)
        self.feedback = feedback
        self._diffusers = [_Allpass(557), _Allpass(683)]

    def process(self, mono: np.ndarray) -> np.ndarray:
        """mono: (n,) -> (2, n) wet stereo."""
        n = len(mono)
        x = mono
        for ap in self._diffusers:
            x = ap.process(x)

        ar = np.arange(n)
        outs = np.empty((4, n))
        for i, d in enumerate(self.DELAYS):
            outs[i] = self._bufs[i][(self._wp[i] + ar) % d]

        damped, self._zi = lfilter(self._b, self._a, outs, axis=1, zi=self._zi)
        fb = self.feedback * (self._mix @ damped)

        for i, d in enumerate(self.DELAYS):
            idx = (self._wp[i] + ar) % d
            self._bufs[i][idx] = 0.5 * x + fb[i]
            self._wp[i] = (self._wp[i] + n) % d

        return np.stack([outs[0] + outs[2], outs[1] + outs[3]]) * 0.5
