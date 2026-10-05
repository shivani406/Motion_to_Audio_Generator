"""
audio_engine.py
===============
Threaded real-time music engine built on a `sounddevice.OutputStream`.

PortAudio calls our callback from its own background thread, so the video loop
is never blocked by audio work. The main thread talks to the audio thread only
through a `threading.Lock()`-protected control block.

Signal flow (per 512-sample block):

    hands -> lead voices (pad / pluck / bell) --+
    backing chord pad ---------------------------+--> bus --+--> reverb (FDN) ---+
                                                             +--> ping-pong delay +--> mix -> tanh -> out
    drum sequencer (3 beats) -------------------------------+

A 16-step sequencer clock drives the drums, the chord changes (one chord per
bar) and the optional "rhythm lock" that quantises plucked notes to the beat.
"""

import threading
from typing import Dict, List

import numpy as np

import config
from beat_engine import BEAT_PATTERNS, BeatPlayer
from effects import FDNReverb, StereoDelay
from music_theory import SCALES, midi_to_freq
from synth import CHORD_PAD, PLAYABLE_PATCHES, Voice

SR = float(config.SAMPLE_RATE)


def _blank_hand() -> dict:
    return dict(active=False, freq=config.MUSICAL_SCALE[0], gain=0.0, pan=0.0, brightness=0.5)


class AudioEngine:
    def __init__(self):
        self._lock = threading.Lock()

        # ---- Control block (written by main thread, read by audio thread) -- #
        self._ctl = dict(
            patch=0, beat=0, bpm=float(config.DEFAULT_BPM), chord_on=False,
            scale=0, space=config.DEFAULT_SPACE, muted=False, quantize=True,
        )
        self._hands: Dict[int, dict] = {0: _blank_hand(), 1: _blank_hand()}
        self._trigger_queue: List[dict] = []

        # ---- Audio-thread state ------------------------------------------ #
        self._voices: List[Voice] = []
        self._lead: Dict[int, Voice] = {0: None, 1: None}
        self._chord_voices: List[Voice] = []
        self._pending: Dict[int, dict] = {}
        self._hands_now: Dict[int, dict] = self._hands
        self._beat = BeatPlayer()
        self._delay = StereoDelay()
        self._reverb = FDNReverb()
        self._mute_gain = 1.0

        self._clock = 0                 # absolute sample count at block start
        self._next_base = 0.0           # absolute time of next step (samples)
        self._step = 0
        self._bar = 0

        self._applied = dict(patch=0, quantize=True, scale=0, chord_on=False, space=-1, bpm=-1.0)
        self._stream = None
        self._error_reported = False

    # ================================================================== #
    # Public API (main thread)
    # ================================================================== #
    def update_hand(self, slot: int, active: bool, freq: float, gain: float,
                    pan: float, brightness: float, trigger: bool) -> None:
        with self._lock:
            self._hands[slot] = dict(active=bool(active), freq=float(freq),
                                     gain=float(np.clip(gain, 0.0, 1.0)),
                                     pan=float(pan), brightness=float(brightness))
            if trigger:
                self._trigger_queue.append(dict(slot=slot, **self._hands[slot]))
                del self._trigger_queue[:-16]       # never grow unbounded

    def _cycle(self, key: str, count: int) -> int:
        with self._lock:
            self._ctl[key] = (self._ctl[key] + 1) % count
            return self._ctl[key]

    def next_patch(self) -> int:
        return self._cycle("patch", len(PLAYABLE_PATCHES))

    def next_scale(self) -> int:
        return self._cycle("scale", len(SCALES))

    def next_space(self) -> int:
        return self._cycle("space", len(config.SPACES))

    def next_beat(self) -> int:
        return self.set_beat((self.status()["beat"] + 1) % (len(BEAT_PATTERNS) + 1))

    def set_beat(self, index: int) -> int:
        """0 = off, 1..3 = patterns. Switching to a pattern adopts its default BPM."""
        with self._lock:
            self._ctl["beat"] = int(index)
            if index > 0:
                self._ctl["bpm"] = float(BEAT_PATTERNS[index - 1].bpm)
            return self._ctl["beat"]

    def adjust_bpm(self, delta: float) -> float:
        with self._lock:
            self._ctl["bpm"] = float(np.clip(self._ctl["bpm"] + delta, 50, 150))
            return self._ctl["bpm"]

    def toggle_chords(self) -> bool:
        with self._lock:
            self._ctl["chord_on"] = not self._ctl["chord_on"]
            return self._ctl["chord_on"]

    def toggle_quantize(self) -> bool:
        with self._lock:
            self._ctl["quantize"] = not self._ctl["quantize"]
            return self._ctl["quantize"]

    def toggle_mute(self) -> bool:
        with self._lock:
            self._ctl["muted"] = not self._ctl["muted"]
            return self._ctl["muted"]

    @property
    def muted(self) -> bool:
        with self._lock:
            return self._ctl["muted"]

    def status(self) -> dict:
        with self._lock:
            c = dict(self._ctl)
        c["patch_name"] = PLAYABLE_PATCHES[c["patch"]].name
        c["scale_name"] = SCALES[c["scale"]].name
        c["beat_name"] = "Off" if c["beat"] == 0 else BEAT_PATTERNS[c["beat"] - 1].name
        c["space_name"] = config.SPACES[c["space"]][0]
        return c

    def start(self) -> bool:
        try:
            import sounddevice as sd
            self._stream = sd.OutputStream(
                samplerate=config.SAMPLE_RATE, blocksize=config.AUDIO_BLOCK_SIZE,
                channels=2, dtype="float32", latency="low", callback=self._callback,
            )
            self._stream.start()
            return True
        except Exception as exc:
            print(f"[audio] Could not start audio stream: {exc}")
            self._stream = None
            return False

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                print(f"[audio] Error while closing stream: {exc}")
            finally:
                self._stream = None

    # ================================================================== #
    # Audio thread
    # ================================================================== #
    def _callback(self, outdata, frames, time_info, status):
        try:
            done = 0
            while done < frames:
                n = min(config.AUDIO_BLOCK_SIZE, frames - done)
                outdata[done:done + n] = self.render_block(n).T
                done += n
        except Exception as exc:                      # never let the audio thread die noisily
            outdata.fill(0)
            if not self._error_reported:
                print(f"[audio] render error: {exc!r}")
                self._error_reported = True

    # ------------------------------------------------------------------ #
    def render_block(self, n: int) -> np.ndarray:
        """Render n <= BLOCK_SIZE stereo samples -> (2, n) float32. Device-free (testable)."""
        with self._lock:
            ctl = dict(self._ctl)
            hands = {s: dict(h) for s, h in self._hands.items()}
            triggers, self._trigger_queue = self._trigger_queue, []

        self._sync_controls(ctl)
        self._hands_now = hands
        patch = PLAYABLE_PATCHES[ctl["patch"]]

        # Note-ons coming from the hands.
        if not patch.sustained:
            for t in triggers:
                if ctl["quantize"]:
                    self._pending[t["slot"]] = t       # fire on next 16th-note step
                else:
                    self._spawn_lead(t, 0, patch)
        self._update_pad_leads(patch, hands)

        # Sequencer: drums, chord changes, quantised notes.
        self._advance_sequencer(n, ctl, patch)

        # Voices.
        bus = np.zeros((2, n))
        alive = []
        for v in self._voices:
            bus += v.render(n)
            if not v.dead:
                alive.append(v)
        self._voices = alive[-config.MAX_VOICES:]
        drums = self._beat.render(n)

        # Effects.
        space = config.SPACES[ctl["space"]]
        send = bus.sum(axis=0) * 0.5 + drums * 0.10
        wet_rev = self._reverb.process(send)
        wet_dly = self._delay.process(bus)
        mix = bus + drums[None, :] + wet_rev * space[2] + wet_dly * space[1]

        # Mute (smoothed) + soft limiter.
        target = 0.0 if ctl["muted"] else 1.0
        gain = np.linspace(self._mute_gain, target, n, endpoint=False) if self._mute_gain != target else target
        self._mute_gain += (target - self._mute_gain) * 0.3
        if abs(self._mute_gain - target) < 1e-3:
            self._mute_gain = target
        out = np.tanh(mix * config.MASTER_DRIVE) * gain
        return out.astype(np.float32)

    # ------------------------------------------------------------------ #
    def _sync_controls(self, ctl: dict) -> None:
        ap = self._applied
        if ctl["patch"] != ap["patch"] or ctl["quantize"] != ap["quantize"]:
            self._pending.clear()
            ap["patch"], ap["quantize"] = ctl["patch"], ctl["quantize"]
        if ctl["scale"] != ap["scale"]:
            ap["scale"] = ctl["scale"]
            if ctl["chord_on"]:
                self._chord_start()
        if ctl["chord_on"] != ap["chord_on"]:
            ap["chord_on"] = ctl["chord_on"]
            self._chord_start() if ctl["chord_on"] else self._chord_stop()
        if ctl["space"] != ap["space"]:
            ap["space"] = ctl["space"]
            self._reverb.feedback = config.SPACES[ctl["space"]][3]
        if ctl["bpm"] != ap["bpm"]:
            ap["bpm"] = ctl["bpm"]
            self._delay.set_time_seconds(60.0 / ctl["bpm"] * 0.75)   # dotted 8th echo

    # ------------------------------------------------------------------ #
    @staticmethod
    def _cutoff(patch, brightness: float) -> float:
        return config.CUTOFF_MIN + brightness * max(patch.cutoff * 1.5 - config.CUTOFF_MIN, 1.0)

    def _spawn_lead(self, t: dict, offset: int, patch) -> None:
        self._voices.append(Voice(
            patch, t["freq"], t["gain"] * config.LEAD_LEVEL, pan=t["pan"],
            cutoff=self._cutoff(patch, t["brightness"]), start_delay=offset,
        ))

    def _update_pad_leads(self, patch, hands: Dict[int, dict]) -> None:
        for slot, h in hands.items():
            v = self._lead.get(slot)
            if not patch.sustained:                      # plucks / bells use note triggers
                if v is not None:
                    v.gate = False
                    self._lead[slot] = None
                continue
            if v is not None and (v.patch is not patch or v.dead):
                v.gate = False
                self._lead[slot] = v = None
            if h["active"]:
                if v is None:
                    v = Voice(patch, h["freq"], 0.0, pan=h["pan"])
                    self._voices.append(v)
                    self._lead[slot] = v
                v.target_freq = h["freq"]
                v.level = h["gain"] * config.LEAD_LEVEL
                v.gate = True
                v.pan = h["pan"]
                v.cutoff = self._cutoff(patch, h["brightness"])
            elif v is not None:
                v.gate = False

    # ------------------------------------------------------------------ #
    def _chord_start(self) -> None:
        self._chord_stop()
        prog = SCALES[self._applied["scale"]].chords
        chord = prog[self._bar % len(prog)]
        for i, midi in enumerate(chord):
            v = Voice(CHORD_PAD, midi_to_freq(midi), config.CHORD_LEVEL, pan=(-0.4, 0.0, 0.4)[i % 3])
            self._voices.append(v)
            self._chord_voices.append(v)

    def _chord_stop(self) -> None:
        for v in self._chord_voices:
            v.gate = False
        self._chord_voices = []

    # ------------------------------------------------------------------ #
    def _advance_sequencer(self, n: int, ctl: dict, patch) -> None:
        step_len = SR * 60.0 / ctl["bpm"] / 4.0                 # one 16th note
        pattern = BEAT_PATTERNS[ctl["beat"] - 1] if ctl["beat"] > 0 else None
        swing = pattern.swing if pattern else 0.0
        start, end = self._clock, self._clock + n

        while True:
            swing_off = swing * step_len if self._step % 2 == 1 else 0.0
            t = self._next_base + swing_off
            if t >= end:
                break
            offset = int(min(max(0, t - start), n - 1))
            self._on_step(self._step % 16, offset, pattern, ctl, patch)
            self._step += 1
            self._next_base += step_len
        self._clock = end

    def _on_step(self, step: int, offset: int, pattern, ctl: dict, patch) -> None:
        if pattern is not None:
            for drum, row in pattern.rows.items():
                c = row[step]
                if c == "x":
                    self._beat.trigger(drum, 1.0, offset)
                elif c == "o":
                    self._beat.trigger(drum, 0.5, offset)

        if step == 0:
            self._bar += 1
            if self._applied["chord_on"]:
                self._chord_start()

        # Rhythm lock: quantised note-ons + a pulse on every 8th note while playing.
        if not patch.sustained and ctl["quantize"]:
            for slot, h in self._hands_now.items():
                t = self._pending.pop(slot, None)
                if t is None and step % 2 == 0 and h["active"]:
                    t = dict(slot=slot, **h)
                    t["gain"] = h["gain"] * 0.85
                if t is not None:
                    self._spawn_lead(t, offset, patch)
