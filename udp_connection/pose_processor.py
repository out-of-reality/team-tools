import mediapipe as mp
import numpy as np

class PoseProcessor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def extract_landmarks(self, frame_rgb):
        """
        Procesa un frame RGB con MediaPipe y devuelve las landmarks si se detectan.
        """
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            return landmarks, results.pose_landmarks
        return None, None

    def close(self):
        self.pose.close()