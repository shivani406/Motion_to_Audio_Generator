# Step 2 & 3: OpenCV capture, preprocessing & Farneback flowimport cv2

import time
import cv2
import config
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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

    base_options = python.BaseOptions(
        model_asset_path="hand_landmarker.task"
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
    )
    self.detector = vision.HandLandmarker.create_from_options(options)

  def get_fingertip_data(self):
    ret, frame = self.cap.read()
    if not ret:
      return None, (None, None)

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(time.time() * 1000)
    result = self.detector.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
      hand = result.hand_landmarks[0]
      index_tip = hand[8]

      cx = int(index_tip.x * config.FRAME_WIDTH)
      cy = int(index_tip.y * config.FRAME_HEIGHT)

      return frame, (cx, cy)

    return frame, (None, None)

  def release(self):
    if self.cap.isOpened():
      self.cap.release()