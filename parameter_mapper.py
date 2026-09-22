# Step 4: Spatial feature extraction & smoothing algorithmsimport numpy as np
import config
import numpy as np

PENTATONIC_SCALE = [
    130.81,
    146.83,
    164.81,
    196.00,
    220.00,  # C3, D3, E3, G3, A3
    261.63,
    293.66,
    329.63,
    392.00,
    440.00,  # C4, D4, E4, G4, A4
    523.25,
    587.33,
    659.25,
    783.99,
    880.00,  # C5, D5, E5, G5, A5
]


class ParameterMapper:

  def __init__(self):
    self.prev_x = None
    self.prev_y = None
    self.smoothed_freq = PENTATONIC_SCALE[0]
    self.smoothed_gain = 0.0

  def map_fingertip_to_audio(self, fingertip_coords):
    cx, cy = fingertip_coords

    if cx is None or cy is None:
      raw_gain = 0.0
      target_freq = self.smoothed_freq
    else:
      if self.prev_x is not None and self.prev_y is not None:
        velocity = np.sqrt((cx - self.prev_x) ** 2 + (cy - self.prev_y) ** 2)
      else:
        velocity = 0.0

      self.prev_x, self.prev_y = cx, cy

      if velocity < 3.0:
        raw_gain = 0.0
      else:
        raw_gain = np.interp(velocity, [3.0, 40.0], [0.05, config.MAX_GAIN])

      scale_idx = int(
          np.interp(cx, [0, config.FRAME_WIDTH], [0, len(PENTATONIC_SCALE) - 1])
      )
      target_freq = PENTATONIC_SCALE[scale_idx]

    alpha = 0.15
    self.smoothed_freq = (
        alpha * target_freq + (1.0 - alpha) * self.smoothed_freq
    )
    self.smoothed_gain = (
        alpha * raw_gain + (1.0 - alpha) * self.smoothed_gain
    )

    return self.smoothed_freq, self.smoothed_gain