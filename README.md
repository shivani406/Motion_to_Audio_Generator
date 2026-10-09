# Motion-Driven Ambient Audio Sculptor

Play music with your hands. A webcam tracks your index fingertips (one or both hands) with MediaPipe, and the app turns your movement into pentatonic notes, drum beats and chords, while drawing a glowing particle ribbon that follows your fingers.

- **Finger height** picks the note (always in tune, snapped to a scale)
- **Finger speed** sets the loudness (a still hand is silent)
- **Left / right position** sets stereo pan and tone brightness
- **Hand gestures** change the beat, instrument, scale, chords and reverb

---

## Requirements

- Python 3.9 to 3.12 (MediaPipe does not support newer versions yet)
- A webcam
- Speakers or headphones

Libraries (pinned in `requirements.txt`): `opencv-python`, `numpy`, `scipy`, `sounddevice`, `mediapipe`.

On Linux you may also need PortAudio: `sudo apt install libportaudio2`.

---

## Setup

```bash
cd ambient_audio_sculptor

# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download the hand model (optional, see note below)
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

**Windows PowerShell:** `curl` is an alias for `Invoke-WebRequest` and rejects `-L`. Use this instead:

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
```

or `curl.exe -L -o hand_landmarker.task <url>`.

If you skip step 3, the app downloads `hand_landmarker.task` automatically on first run. Keep the file next to `main.py`.

---

## Run

```bash
python main.py                 # default webcam, two-hand mode
python main.py --camera 1      # use another webcam
python main.py --one-hand      # start in single-hand mode
python main.py --no-audio      # visuals only
```

**To play:** hold up a hand and **point with your index finger**, then move it. Keep the rest of the hand clearly curled so the app reads it as a pointing pose.

---

## Hand gestures

Hold each pose for about **0.7 seconds**. A ring fills around your wrist and the action fires once. Gesture poses make no sound, so they never cause stray notes.

| Pose | Action | What it means musically |
|---|---|---|
| Point (index only) | Play notes | Melody |
| Open palm | Next background beat (Off, Lo-Fi, House, Heartbeat) | Rhythm and tempo |
| Peace sign | Next sound (Warm Pad, Soft Pluck, Glass Bell) | Instrument / timbre |
| Thumbs up | Next scale (Major, Minor, Hirajoshi) | Mood |
| Fist | Chord pad on / off | Harmony |
| Rock sign (index + little finger) | Next reverb space (Room, Hall, Cathedral) | Sense of space |

---

## Keyboard controls

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `c` | Clear the trail canvas |
| `m` | Mute / unmute |
| `h` | Show / hide the HUD |
| `0` | Beat off |
| `1` `2` `3` | Choose beat: Lo-Fi Chill (78 BPM), Soft House (108 BPM), Dream Heartbeat (66 BPM) |
| `+` / `-` | Tempo up / down (4 BPM steps) |
| `p` | Next sound |
| `s` | Next scale |
| `k` | Chord pad on / off |
| `r` | Next reverb space |
| `z` | Rhythm lock on / off |
| `t` | One-hand / two-hand mode |
| `g` | Gestures on / off |

---

## How it works

Per video frame (about 30 times a second):

1. OpenCV captures a 640x480 frame and mirrors it.
2. MediaPipe Hand Landmarker finds 21 landmarks per hand (up to 2 hands).
3. The index fingertip (landmark 8) is smoothed with a moving average.
4. Pose is classified from landmark distances, with a 5-frame majority vote and a 0.7 s hold.
5. The mapper converts the fingertip into a note (Y), loudness (speed), pan and brightness (X).
6. The audio thread (every 512 samples, about 11.6 ms) mixes voices, chords and drums, adds echo and reverb, and plays the result.
7. The particle engine fades the old trail, interpolates new points, adds glow and shows the result with the HUD.

**Rhythm lock** (`z`, on by default): plucked and bell notes snap to the next 16th-note step, and repeat on every 8th note while you keep moving. Turn it off if you prefer instant, free-timed notes.

---

## Project structure

```text
ambient_audio_sculptor/
├── main.py                # Main loop, gestures to actions, HUD, keyboard
├── config.py              # All constants and tuning values
├── motion_detector.py     # MediaPipe wrapper, two-hand slots, fingertip smoothing
├── gesture_recognizer.py  # Pose classification, voting, hold-to-fire actions
├── parameter_mapper.py    # Fingertip to note / loudness / pan / trigger
├── music_theory.py        # Scales, note names, chord progressions
├── synth.py               # Instrument patches (pad, pluck, bell) and Voice class
├── beat_engine.py         # Drum sounds and the three 16-step beats
├── effects.py             # Ping-pong echo and reverb
├── audio_engine.py        # Audio thread, sequencer, mixing, thread-safe controls
├── particle_engine.py     # Ribbon interpolation, glow and exponential decay
├── requirements.txt
└── README.md
```

---

## Tuning (`config.py`)

| Setting | What it does |
|---|---|
| `VELOCITY_THRESHOLD`, `VELOCITY_MAX` | How fast you must move to start sound, and to reach full volume |
| `NOTE_HYSTERESIS` | Higher means notes flicker less at boundaries |
| `LANDMARK_SMOOTHING` | Lower means smoother but slightly laggier fingertip |
| `GESTURE_HOLD_SECONDS` | How long to hold a gesture |
| `FINGER_EXTEND_RATIO`, `THUMB_EXTEND_RATIO` | Gesture sensitivity; adjust if poses misfire |
| `LEAD_LEVEL`, `CHORD_LEVEL`, `DRUM_LEVEL` | Mix balance |
| `SPACES` | Echo and reverb amounts for Room / Hall / Cathedral |
| `DEFAULT_BPM` | Starting tempo |
| `CANVAS_FADE_FACTOR` | Trail length (closer to 1.0 means longer trails) |
| `MAX_VOICES` | Lower this if audio crackles on a slow CPU |

---


