import logging

import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class PoseProcessor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmark_names = [lm.name for lm in mp.solutions.pose.PoseLandmark] 

    def extract_landmarks(self, frame_rgb):
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            landmarks_dict = {}
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_name = self.landmark_names[i]
                landmarks_dict[landmark_name] = [landmark.x, landmark.y, landmark.z]
            return landmarks_dict, results.pose_landmarks 
        return None, None

    def close(self):
        self.pose.close()
