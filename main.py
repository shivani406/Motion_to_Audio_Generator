# Application entry point & execution loopimport cv2
import cv2
import config
from audio_engine import AudioSynthesizer
from motion_detector import MotionDetector
from parameter_mapper import ParameterMapper
from particle_engine import ParticleEngine


def main():
  detector = MotionDetector()
  mapper = ParameterMapper()
  synth = AudioSynthesizer()
  particles = ParticleEngine()

  synth.start()
  print("Motion-Driven Ambient Audio Sculptor (Phase 2) running...")
  print("Controls: Move hands to sculpt sound & visual trails.")
  print("Press 'c' to clear canvas. Press 'q' to quit.")

  try:
    while True:
      frame, gray = detector.get_frame()
      if frame is None:
        print("Error: Failed to capture video frame.")
        break

      magnitude, angle = detector.compute_flow(gray)

      if magnitude is not None and angle is not None:
        cx, cy, energy = mapper.extract_features(magnitude)
        freq, cutoff, gain = mapper.map_to_audio(cx, cy, energy)

        synth.update_params(freq, cutoff, gain)

        particles.spawn_from_flow(magnitude, angle)
        particle_canvas = particles.update_and_render()

        frame = cv2.addWeighted(frame, 0.7, particle_canvas, 1.0, 0)

        if cx is not None and cy is not None and energy > 0:
          cv2.circle(frame, (int(cx), int(cy)), 8, (255, 255, 255), -1)

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
            f"Filter Cutoff: {cutoff:.1f} Hz",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Gain: {gain:.2f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

      cv2.imshow("Ambient Audio Sculptor - Phase 2", frame)

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