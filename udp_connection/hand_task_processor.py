import time
from collections import deque
import logging

import cv2
import pyautogui
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
MPImage = mp.Image
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

logger = logging.getLogger(__name__)

class HandTaskProcessor:
    def __init__(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._callback
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.sw, self.sh = pyautogui.size()
        pyautogui.FAILSAFE = False
        self.position_buffer = deque(maxlen=5)
        self.avg_x = self.sw // 2
        self.avg_y = self.sh // 2
        logger.info("HandTaskProcessor initialized")

    def _callback(self, result: HandLandmarkerResult, output_image, timestamp_ms):
        if not result.hand_landmarks or not result.handedness:
            return
        if result.handedness[0][0].category_name != "Right":
            return

        landmarks = result.hand_landmarks[0]
        hand_state = self._get_hand_state(
            landmarks, result.handedness[0][0].category_name
        )
        x, y = int(hand_state['index_tip_pos'][0] * self.sw), int(hand_state['index_tip_pos'][1] * self.sh)
        self.position_buffer.append((x, y))

        if hand_state['click_gesture']:
            pyautogui.click(self.sw - self.avg_x, self.avg_y)
            self.position_buffer.clear()
        elif hand_state['move_gesture']:
            if len(self.position_buffer) >= 2:
                self.avg_x = sum(p[0] for p in self.position_buffer) // len(self.position_buffer)
                self.avg_y = sum(p[1] for p in self.position_buffer) // len(self.position_buffer)
                pyautogui.moveTo(self.sw - self.avg_x, self.avg_y)

    def _is_finger_extended(self, landmarks, tip_landmark, pip_landmark):
        return landmarks[tip_landmark].y < landmarks[pip_landmark].y

    def _is_thumb_extended(self, landmarks, handedness):
        if handedness == 'Right':
            return (
                landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP].x
                > landmarks[mp.solutions.hands.HandLandmark.THUMB_IP].x
            )
        return (
            landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP].x
            < landmarks[mp.solutions.hands.HandLandmark.THUMB_IP].x
        )

    def _get_finger_distance(self, landmarks, finger1_landmark, finger2_landmark):
        dx = landmarks[finger1_landmark].x - landmarks[finger2_landmark].x
        dy = landmarks[finger1_landmark].y - landmarks[finger2_landmark].y
        return np.sqrt(dx ** 2 + dy ** 2)

    def _get_hand_state(self, landmarks, handedness):
        fingers_state = {
            'thumb': self._is_thumb_extended(landmarks, handedness),
            'index': self._is_finger_extended(
                landmarks,
                mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
            ),
            'middle': self._is_finger_extended(
                landmarks,
                mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
            ),
            'ring': self._is_finger_extended(
                landmarks,
                mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
            ),
            'pinky': self._is_finger_extended(
                landmarks,
                mp.solutions.hands.HandLandmark.PINKY_TIP,
                mp.solutions.hands.HandLandmark.PINKY_PIP,
            ),
        }
        move_gesture = (
            fingers_state['index']
            and not fingers_state['middle']
            and not fingers_state['ring']
            and not fingers_state['pinky']
        )
        thumb_index_distance = self._get_finger_distance(
            landmarks,
            mp.solutions.hands.HandLandmark.THUMB_TIP,
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
        )
        pinch_threshold = 0.04
        click_gesture = (
            thumb_index_distance < pinch_threshold
            and fingers_state['middle']
            and fingers_state['ring']
            and fingers_state['pinky']
        )
        return {
            'fingers': fingers_state,
            'move_gesture': move_gesture,
            'click_gesture': click_gesture,
            'thumb_index_distance': thumb_index_distance,
            'index_tip_pos': (
                landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x,
                landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y,
            ),
        }

    def send_frame(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        if width > 320:
            scale = 320 / width
            frame_bgr = cv2.resize(frame_bgr, (int(width * scale), int(height * scale)))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts = int(time.time() * 1000)
        try:
            self.landmarker.detect_async(mp_image, ts)
        except Exception as e:
            logger.error("Error enviando frame para detección de mano: %s", e)

    def close(self):
        self.landmarker.close()
