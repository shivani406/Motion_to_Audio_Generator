# Step 2 & 3: OpenCV capture, preprocessing & Farneback flowimport cv2
import cv2
import numpy as np
import config


class MotionDetector:

  def __init__(
      self,
      camera_index=config.CAMERA_INDEX,
      width=config.FRAME_WIDTH,
      height=config.FRAME_HEIGHT,
  ):
    self.cap = cv2.VideoCapture(camera_index)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    self.prev_gray = None

  def get_frame(self):
    ret, frame = self.cap.read()
    if not ret:
      return None, None
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    return frame, gray_blurred

  def compute_flow(self, current_gray):
    if self.prev_gray is None:
      self.prev_gray = current_gray
      return None, None

    flow = cv2.calcOpticalFlowFarneback(
        prev=self.prev_gray,
        next=current_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    self.prev_gray = current_gray

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude, angle

  def release(self):
    if self.cap.isOpened():
      self.cap.release()


if __name__ == "__main__":
  detector = MotionDetector()
  print("Testing Motion Detector (Press 'q' to exit)...")

  while True:
    frame, gray = detector.get_frame()
    if frame is None:
      break

    magnitude, angle = detector.compute_flow(gray)

    if magnitude is not None:
      motion_mask = (
          (magnitude > config.MOTION_THRESHOLD).astype(np.uint8) * 255
      )
      cv2.imshow("Motion Mask", motion_mask)

    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  detector.release()
  cv2.destroyAllWindows()
  print("Motion Detector Test Complete.")