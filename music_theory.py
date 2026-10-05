"""
music_theory.py
===============
Scales, note naming and the backing chord progression for each scale.

Every scale is built from MIDI notes between C3 (48 = 130.81 Hz) and
A5 (81 = 880.00 Hz), so notes always stay in a comfortable, musical range.
"""

from dataclasses import dataclass
from typing import List, Tuple

ROOT_MIDI = 48    # C3
TOP_MIDI = 81     # A5
_LETTERS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_freq(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def midi_to_name(midi: int) -> str:
    return f"{_LETTERS[midi % 12]}{midi // 12 - 1}"


@dataclass(frozen=True)
class ScalePreset:
    name: str
    intervals: Tuple[int, ...]            # semitones above C
    chords: Tuple[Tuple[int, ...], ...]   # backing progression, one chord per bar (MIDI notes)


SCALES: List[ScalePreset] = [
    ScalePreset(
        "C Major Pentatonic", (0, 2, 4, 7, 9),
        (
            (55, 60, 64),   # C   (G3 C4 E4)
            (57, 60, 64),   # Am  (A3 C4 E4)
            (57, 60, 65),   # F   (A3 C4 F4)
            (55, 59, 62),   # G   (G3 B3 D4)
        ),
    ),
    ScalePreset(
        "C Minor Pentatonic", (0, 3, 5, 7, 10),
        (
            (55, 60, 63),   # Cm
            (56, 60, 63),   # Ab
            (58, 62, 65),   # Bb
            (55, 58, 62),   # Gm
        ),
    ),
    ScalePreset(
        "C Hirajoshi (Japanese)", (0, 2, 3, 7, 8),
        (
            (55, 60, 63),   # Cm
            (56, 60, 63),   # Ab
            (55, 58, 63),   # Eb
            (56, 60, 63),   # Ab
        ),
    ),
]


def build_scale(index: int) -> Tuple[List[float], List[str]]:
    """Return (frequencies, note names) for scale preset `index`, low -> high."""
    preset = SCALES[index % len(SCALES)]
    midis = [m for m in range(ROOT_MIDI, TOP_MIDI + 1) if (m - ROOT_MIDI) % 12 in preset.intervals]
    return [midi_to_freq(m) for m in midis], [midi_to_name(m) for m in midis]
