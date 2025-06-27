import logging
import os
import queue
from datetime import datetime

import cv2
import requests

import config

logger = logging.getLogger(__name__)


class VideoRecorder:
    def __init__(self, token, frame_size=(1280, 720), fps=30):
        self.out = None
        self.token = token
        self.frame_size = frame_size
        self.fps = fps
        self.video_name = None
        self.is_recording = False

    def start_recording(self):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_name = f"{now}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.video_name, fourcc, self.fps, self.frame_size)

        if self.out.isOpened():
            logger.info("Recording started with resolution %s, saving to %s", self.frame_size, self.video_name)
            self.is_recording = True
        else:
            logger.error("Could not open VideoWriter for %s", self.video_name)
            self.is_recording = False

    def record(self, frame_queue):
        logger.info("Recording thread started")
        while self.is_recording:
            try:
                frame = frame_queue.get(timeout=1)
                if frame is not None and self.out:
                    resized_frame = cv2.resize(frame, self.frame_size)
                    self.out.write(resized_frame)
            except queue.Empty:
                continue
        logger.info("Recording thread finished")

    def stop_recording(self):
        self.is_recording = False
        if self.out:
            logger.info("Finalizing video file: %s", self.video_name)
            self.out.release()
            self.out = None
            logger.info("Recording stopped. Video saved as %s", self.video_name)
            self.send_video_to_api()
        else:
            logger.warning("Video recorder was not active or already stopped")

    def send_video_to_api(self):
        if not (self.video_name and os.path.exists(self.video_name)):
            logger.error("Video file '%s' not found for upload", self.video_name)
            return

        if config.API_URL:
            url = f"{config.API_URL.rstrip('/')}/upload/"
            headers = {'Authorization': f'Bearer {self.token}'}

            logger.info("Attempting to upload video %s to %s", self.video_name, url)
            try:
                with open(self.video_name, 'rb') as video_file:
                    files = {'video': (os.path.basename(self.video_name), video_file, 'video/mp4')}
                    response = requests.post(url, files=files, headers=headers, timeout=60)
                    response.raise_for_status()
                    logger.info("API response: %s - %s", response.status_code, response.text)
                    logger.info("Video uploaded successfully!")
                    self.cleanup_video_file()
            except requests.exceptions.RequestException as e:
                logger.error("Error sending video to API: %s", e)
            except Exception as e:
                logger.error("Unexpected error during upload: %s", e)
        else:
            logger.warning("API URL not configured in config.py. Video will not be sent")

    def cleanup_video_file(self):
        if self.video_name and os.path.exists(self.video_name):
            try:
                os.remove(self.video_name)
                logger.info("Local file %s deleted", self.video_name)
            except Exception as e:
                logger.error("Error cleaning up video file: %s", e)
