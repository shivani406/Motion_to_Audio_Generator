# Application entry point & execution loopimport cv2
import cv2
import config
from audio_engine import AudioSynthesizer
from motion_detector import MotionDetector
from parameter_mapper import ParameterMapper
from particle_engine import Particle, ParticleEngine


def main():
  detector = MotionDetector()
  mapper = ParameterMapper()
  synth = AudioSynthesizer()
  particles = ParticleEngine()

  synth.start()
  print("Motion-Driven Ambient Audio Sculptor (Phase 2 - MediaPipe) running...")
  print("Controls: Move index finger to sculpt sound & visual trails.")
  print("Press 'c' to clear canvas. Press 'q' to quit.")

  try:
    while True:
      frame, (cx, cy) = detector.get_fingertip_data()
      if frame is None:
        print("Error: Failed to capture video frame.")
        break

      freq, gain = mapper.map_fingertip_to_audio((cx, cy))
      synth.update_params(freq, config.MAX_CUTOFF, gain)

      if cx is not None and cy is not None and gain > 0:
        particles.particles.append(
            Particle(cx, cy, magnitude=8.0, angle=0.0)
        )

      particle_canvas = particles.update_and_render()
      frame = cv2.addWeighted(frame, 0.7, particle_canvas, 1.0, 0)

      if cx is not None and cy is not None:
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 2)

      cv2.putText(
          frame,
          f"Frequency: {freq:.1f} Hz",
          (20, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 255, 255),
          2,
      )
      cv2.putText(
          frame,
          f"Gain: {gain:.2f}",
          (20, 70),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 255, 255),
          2,
      )

      cv2.imshow("Ambient Audio Sculptor - MediaPipe", frame)

      key = cv2.waitKey(1) & 0xFF
      if key == ord("q"):
        break
      elif key == ord("c"):
        particles.clear()

  finally:
    print("\nShutting down engine...")
    synth.stop()
    detector.release()
    cv2.destroyAllWindows()
    print("Cleanup complete. Application exited gracefully.")


if __name__ == "__main__":
  main()