# Step 4: Spatial feature extraction & smoothing algorithmsimport numpy as np
import config
import numpy as np

class ParameterMapper:

  def __init__(self):
    self.smoothed_freq = config.MIN_FREQ
    self.smoothed_cutoff = config.MIN_CUTOFF
    self.smoothed_gain = 0.0

  def extract_features(self, magnitude):
    motion_mask = magnitude > config.MOTION_THRESHOLD
    active_mags = magnitude[motion_mask]

    if active_mags.size == 0:
      return None, None, 0.0

    y_indices, x_indices = np.where(motion_mask)

    centroid_x = np.mean(x_indices)
    centroid_y = np.mean(y_indices)
    total_energy = np.sum(active_mags)

    return centroid_x, centroid_y, total_energy

  def map_to_audio(self, centroid_x, centroid_y, total_energy):
    if centroid_x is None or centroid_y is None or total_energy == 0.0:
      raw_freq = self.smoothed_freq
      raw_cutoff = self.smoothed_cutoff
      raw_gain = 0.0
    else:
      raw_freq = np.interp(
          centroid_x, [0, config.FRAME_WIDTH], [config.MIN_FREQ, config.MAX_FREQ]
      )

      raw_cutoff = np.interp(
          centroid_y,
          [0, config.FRAME_HEIGHT],
          [config.MAX_CUTOFF, config.MIN_CUTOFF],
      )

      normalized_energy = np.interp(
          total_energy, [0.0, 50000.0], [0.0, config.MAX_GAIN]
      )
      raw_gain = np.clip(normalized_energy, 0.0, config.MAX_GAIN)

    alpha = config.SMOOTHING_ALPHA
    self.smoothed_freq = (
        alpha * raw_freq + (1.0 - alpha) * self.smoothed_freq
    )
    self.smoothed_cutoff = (
        alpha * raw_cutoff + (1.0 - alpha) * self.smoothed_cutoff
    )
    self.smoothed_gain = (
        alpha * raw_gain + (1.0 - alpha) * self.smoothed_gain
    )

    return self.smoothed_freq, self.smoothed_cutoff, self.smoothed_gain


if __name__ == "__main__":
  mapper = ParameterMapper()
  dummy_mag = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH))
  dummy_mag[100:200, 100:200] = 5.0

  cx, cy, energy = mapper.extract_features(dummy_mag)
  freq, cutoff, gain = mapper.map_to_audio(cx, cy, energy)

  print(
      f"Extracted Centroid: ({cx:.1f}, {cy:.1f}), Total Energy: {energy:.1f}"
  )
  print(
      f"Mapped Parameters -> Frequency: {freq:.1f}Hz, Cutoff: {cutoff:.1f}Hz,"
      f" Gain: {gain:.2f}"
  )