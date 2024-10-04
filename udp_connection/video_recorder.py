import cv2
import requests
import os
from datetime import datetime
import config
from urllib.parse import urljoin
from contextlib import suppress


class VideoRecorder:
    def __init__(self, user_id, frame_size=(640, 480), fps=30):
        self.cap = None
        self.out = None
        self.recording = False
        self.user_id = user_id
        self.frame_size = frame_size
        self.fps = fps
        self.video_name = None

    def start_recording(self):
        if self.cap is None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_name = f"{self.user_id}_{now}.avi"
            self.cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(self.video_name, fourcc, self.fps, self.frame_size)
        self.recording = True

    def record(self):
        try:
            while self.recording:
                ret, frame = self.cap.read()
                if ret:
                    self.out.write(frame)
                else:
                    break
        finally:
            self.stop_recording()

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()
        self.send_video_to_api()
        self.cleanup_video_file()

    def send_video_to_api(self):
        if config.url:
            url = urljoin(config.url, 'upload/')
            with suppress(FileNotFoundError, Exception):
                with open(self.video_name, 'rb') as video_file:
                    files = {'video': video_file}
                    data = {'user_id': self.user_id}

                    response = requests.post(url, files=files, data=data)
                    response.raise_for_status()

    def cleanup_video_file(self):
        try:
            if os.path.exists(self.video_name):
                os.remove(self.video_name)
        except Exception:
            pass
