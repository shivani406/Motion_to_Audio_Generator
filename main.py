# Application entry point & execution loopimport cv2
import cv2
from audio_engine import AudioSynthesizer
from motion_detector import MotionDetector
from parameter_mapper import ParameterMapper


def main():
  detector = MotionDetector()
  mapper = ParameterMapper()
  synth = AudioSynthesizer()

  synth.start()
  print("Motion-Driven Ambient Audio Sculptor running...")
  print("Controls: Move your hands in front of the camera. Press 'q' to quit.")

  try:
    while True:
      frame, gray = detector.get_frame()
      if frame is None:
        print("Error: Failed to capture video frame.")
        break

      magnitude, angle = detector.compute_flow(gray)

      if magnitude is not None:
        cx, cy, energy = mapper.extract_features(magnitude)
        freq, cutoff, gain = mapper.map_to_audio(cx, cy, energy)

        synth.update_params(freq, cutoff, gain)

        if cx is not None and cy is not None and energy > 0:
          cv2.circle(frame, (int(cx), int(cy)), 12, (0, 255, 0), -1)
          cv2.circle(frame, (int(cx), int(cy)), 20, (0, 255, 255), 2)

        cv2.putText(
            frame,
            f"Frequency: {freq:.1f} Hz",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Filter Cutoff: {cutoff:.1f} Hz",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Gain: {gain:.2f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

      cv2.imshow("Ambient Audio Sculptor - Phase 1", frame)

      if cv2.waitKey(1) & 0xFF == ord("q"):
        break

  finally:
    print("\nShutting down engine...")
    synth.stop()
    detector.release()
    cv2.destroyAllWindows()
    print("Cleanup complete. Application exited gracefully.")


if __name__ == "__main__":
  main()