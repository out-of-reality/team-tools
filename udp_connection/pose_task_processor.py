import time
import logging

import cv2
import mediapipe as mp
import numpy as np
from butterworth_filter import ButterworthFilter

try:
    from mediapipe.framework.formats import landmark_pb2
except ImportError:
    landmark_pb2 = None 

logger = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
MPImage = mp.Image
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    if landmark_pb2 is None:
        return annotated_image
    for pose_landmarks in pose_landmarks_list:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


class PoseTaskProcessor:
    def __init__(self, result_queue, display_queue=None):
        self.result_queue = result_queue
        self.display_queue = display_queue
        self.current_frame_rgb = None
        self.t0 = time.time()
        self.fps = 30.0

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._callback
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.landmark_names = [lm.name for lm in mp.solutions.pose.PoseLandmark]
        self.filters = ButterworthFilter(order=4, cutoff=0.15)

    def _callback(self, result: PoseLandmarkerResult, output_image: MPImage, timestamp_ms: int):
        if result.pose_world_landmarks:
            lms = result.pose_world_landmarks[0]
            landmarks_dict = {}
            for i, lm in enumerate(lms):
                name = self.landmark_names[i]
                [fx, fy, fz] = self.filters.filter(name, [lm.x, lm.y, lm.z])
                landmarks_dict[name] = [fx, fy, fz]
            self.result_queue.put(landmarks_dict)
            if self.display_queue is not None and self.current_frame_rgb is not None:
                try:
                    annotated_image = draw_landmarks_on_image(self.current_frame_rgb, result)
                    annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                    self.display_queue.put_nowait(annotated_bgr)
                except Exception as e:
                    logger.info(f"Error drawing landmarks: {e}")

    def send_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.current_frame_rgb = frame_rgb.copy()
        mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts = int(time.time() * 1000)
        self.landmarker.detect_async(mp_image, ts)

    def close(self):
        self.landmarker.close()
