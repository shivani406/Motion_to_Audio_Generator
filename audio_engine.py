# Step 1: Threaded Audio Synthesizer class
import threading
import numpy as np
import sounddevice as sd
import config


class AudioSynthesizer:

  def __init__(self, sample_rate=config.SAMPLE_RATE):
    self.sample_rate = sample_rate
    self.frequency = config.MIN_FREQ
    self.filter_cutoff = config.MIN_CUTOFF
    self.gain = 0.0
    self.phase = 0.0
    self.is_running = False
    self.stream = None
    self.lock = threading.Lock()

  def _audio_callback(self, outdata, frames, time_info, status):
    with self.lock:
      target_freq = self.frequency
      target_gain = self.gain

    t = (np.arange(frames) + self.phase) / self.sample_rate
    self.phase = (self.phase + frames) % self.sample_rate

    sine_wave = np.sin(2 * np.pi * target_freq * t)
    audio_signal = sine_wave * target_gain

    outdata[:, 0] = audio_signal.astype(np.float32)

  def start(self):
    self.is_running = True
    self.stream = sd.OutputStream(
        samplerate=self.sample_rate,
        channels=1,
        callback=self._audio_callback,
        blocksize=1024,
    )
    self.stream.start()

  def update_params(self, frequency, filter_cutoff, gain):
    with self.lock:
      self.frequency = float(frequency)
      self.filter_cutoff = float(filter_cutoff)
      self.gain = float(gain)

  def stop(self):
    self.is_running = False
    if self.stream:
      self.stream.stop()
      self.stream.close()


if __name__ == "__main__":
  import time

  print("Testing Audio Engine Thread...")
  synth = AudioSynthesizer()
  synth.start()

  synth.update_params(440.0, 1000.0, 0.3)
  time.sleep(1.0)
  synth.update_params(880.0, 1000.0, 0.3)
  time.sleep(1.0)

  synth.stop()
  print("Audio Engine Test Complete.")