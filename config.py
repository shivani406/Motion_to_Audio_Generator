"""
config.py
=========
Global constants for the Motion-Driven Ambient Audio Sculptor.

Every tunable value lives here so the other modules contain no magic numbers.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# --------------------------------------------------------------------------- #
# Video / camera
# --------------------------------------------------------------------------- #
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
TARGET_FRAME_MS = 33.0          # latency budget per frame (<= 33 ms)
WINDOW_NAME = "Motion-Driven Ambient Audio Sculptor"

# --------------------------------------------------------------------------- #
# MediaPipe hand tracking
# --------------------------------------------------------------------------- #
INDEX_FINGERTIP_ID = 8          # MediaPipe landmark index for the index fingertip
MAX_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
# Exponential smoothing of the fingertip position: 1.0 = raw, lower = smoother.
LANDMARK_SMOOTHING = 0.65

# --------------------------------------------------------------------------- #
# Musical scale: C major pentatonic (C, D, E, G, A) across three octaves,
# C3 (130.81 Hz) up to A5 (880.00 Hz).
# --------------------------------------------------------------------------- #
MUSICAL_SCALE = [
    130.81, 146.83, 164.81, 196.00, 220.00,    # C3 D3 E3 G3 A3
    261.63, 293.66, 329.63, 392.00, 440.00,    # C4 D4 E4 G4 A4
    523.25, 587.33, 659.25, 783.99, 880.00,    # C5 D5 E5 G5 A5
]
NOTE_NAMES = [
    "C3", "D3", "E3", "G3", "A3",
    "C4", "D4", "E4", "G4", "A4",
    "C5", "D5", "E5", "G5", "A5",
]

# --------------------------------------------------------------------------- #
# Parameter mapping
# --------------------------------------------------------------------------- #
# Fraction of the frame height ignored at the top and bottom so the highest and
# lowest notes are comfortably reachable.
Y_MARGIN_FRACTION = 0.08
# Extra distance (in scale steps) the fingertip must travel past a note boundary
# before the note changes. Prevents flickering between two neighbouring notes.
NOTE_HYSTERESIS = 0.15
# Velocity is measured in pixels per frame and smoothed with an EMA.
VELOCITY_SMOOTHING = 0.5
VELOCITY_THRESHOLD = 2.5        # below this the gain is forced to zero
VELOCITY_MAX = 45.0             # velocity at which gain saturates
GAIN_CURVE_EXPONENT = 0.7       # <1 makes gentle motion louder
MIN_GAIN = 0.12                 # gain right above the threshold
MAX_GAIN = 1.0

# --------------------------------------------------------------------------- #
# Audio engine
# --------------------------------------------------------------------------- #
SAMPLE_RATE = 44100
AUDIO_BLOCK_SIZE = 512          # ~11.6 ms per audio block
AUDIO_CHANNELS = 2
PORTAMENTO_COEFF = 0.15         # fraction of the pitch gap closed per audio block
GAIN_ATTACK_COEFF = 0.30        # fraction of the gain gap closed per block (rising)
GAIN_RELEASE_COEFF = 0.06       # fraction of the gain gap closed per block (falling)
VIBRATO_RATE_HZ = 5.0
VIBRATO_DEPTH = 0.006           # +/- 0.6 % frequency deviation (~10 cents)
# Relative levels of the partials that make up the voice.
SUB_HARMONIC_LEVEL = 0.45       # one octave below the fundamental
SECOND_HARMONIC_LEVEL = 0.22
THIRD_HARMONIC_LEVEL = 0.10
SATURATION_DRIVE = 1.8          # input gain into np.tanh()
OUTPUT_LEVEL = 0.55             # final master volume (0..1)

# --------------------------------------------------------------------------- #
# Particle / visual engine
# --------------------------------------------------------------------------- #
CANVAS_FADE_FACTOR = 0.85       # canvas is multiplied by this every frame
CANVAS_CUTOFF = 0.5             # values below this are snapped to 0 (kills denormals)
PARTICLE_MIN_RADIUS = 4
PARTICLE_MAX_RADIUS = 13
PARTICLE_MIN_INTENSITY = 0.25
MAX_INTERPOLATED_PARTICLES = 200
GLOW_SIGMA = 7.0
GLOW_STRENGTH = 1.6
CORE_STRENGTH = 0.9
PARTICLE_SPACING = 3.0          # pixels between interpolated particles
SPARKLES_PER_FRAME = 3
HUE_RANGE = (90, 165)           # OpenCV hue (0-179): cyan -> blue -> violet

# --------------------------------------------------------------------------- #
# HUD
# --------------------------------------------------------------------------- #
VIDEO_DIM = 0.55                # brightness of the camera feed behind the trails


# --------------------------------------------------------------------------- #
# Gestures (pose classification + hold-to-trigger)
# --------------------------------------------------------------------------- #
GESTURE_HOLD_SECONDS = 0.7      # how long a pose must be held to fire its action
GESTURE_VOTE_FRAMES = 5         # majority vote over the last N frames (anti-flicker)
FINGER_EXTEND_RATIO = 1.15      # tip-to-wrist / pip-to-wrist ratio => finger is extended
THUMB_EXTEND_RATIO = 0.75       # thumb-tip to index-MCP distance / palm size

# --------------------------------------------------------------------------- #
# Music mix / effects
# --------------------------------------------------------------------------- #
DEFAULT_BPM = 80
LEAD_LEVEL = 0.45               # level of the finger-played voices
CHORD_LEVEL = 0.13              # level of each backing chord note
DRUM_LEVEL = 0.60
MASTER_DRIVE = 0.9              # input gain into the final tanh soft limiter
MAX_VOICES = 28
CUTOFF_MIN = 900.0              # lowest voice filter cutoff (hand at left edge), Hz
DELAY_FEEDBACK = 0.42
# (name, delay wet, reverb wet, reverb feedback/length)
SPACES = [
    ("Room", 0.10, 0.16, 0.80),
    ("Hall", 0.20, 0.30, 0.88),
    ("Cathedral", 0.28, 0.45, 0.94),
]
DEFAULT_SPACE = 1
HUE_RANGES = [(90, 165), (2, 32)]   # per hand slot: cyan->violet, gold->orange
