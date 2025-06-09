import logging
import queue
import socket
import threading
from contextlib import suppress

import cv2
import mediapipe as mp

from pose_processor import PoseProcessor
from udp_signal_emitter import PoseSender
from video_recorder import VideoRecorder

logger = logging.getLogger(__name__)


class UDPSignalReceiver:
    def __init__(self, host="localhost", port=9999, queue_size=10, debug_mode=False):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((host, port))
        
        self.recorder = None
        self.debug_mode = debug_mode
        self.is_recording = False
        self.lock = threading.Lock()

        self.recording_thread = None
        self.processing_thread = None
        self.frame_thread = None

        self.recording_queue = queue.Queue(maxsize=queue_size)
        self.processing_queue = queue.Queue(maxsize=queue_size)
        self.display_queue = queue.Queue(maxsize=2)

    def listen_for_signals(self):
        logger.info("UDP listener started, waiting for signals...")
        while True:
            try:
                data, addr = self.udp_socket.recvfrom(1024)
                signal = data.decode('utf-8').strip()
                if not ' ' in signal: 
                    signal = f"{signal} 12345"

                if signal.startswith("start"):
                    self._handle_start_signal(signal, addr)
                elif signal.startswith("stop"):
                    self._handle_stop_signal()
            except Exception as e:
                logger.error(f"Error in UDP listener: {e}")
    
    def _handle_start_signal(self, signal, addr):
        with self.lock:
            if self.is_recording:
                logger.warning("Received 'start' signal but recording is already in progress")
                return

            logger.info("Starting new recording session")
            
            cap_test = cv2.VideoCapture(0)
            if not cap_test.isOpened():
                logger.error("Critical error: Cannot access camera to detect resolution")
                return
            
            ret, frame = cap_test.read()
            if not ret:
                logger.error("Critical error: Could not read frame to detect resolution")
                cap_test.release()
                return
            
            height, width, _ = frame.shape
            frame_size = (width, height)
            cap_test.release()

            self.is_recording = True
            token = signal.split()[1]
            
            self._clear_queues()
            self.recorder = VideoRecorder(token, frame_size=frame_size)
            self.recorder.start_recording()
            
            if not self.recorder.is_recording:
                logger.error("Critical error: Failed to start video recorder")
                self.is_recording = False
                return

            self.frame_thread = threading.Thread(target=self.record_frames)
            self.recording_thread = threading.Thread(target=self.recorder.record, args=(self.recording_queue,))
            self.processing_thread = threading.Thread(target=self.process_frames, args=(addr,))

            self.frame_thread.start()
            self.recording_thread.start()
            self.processing_thread.start()

    def _handle_stop_signal(self):
        with self.lock:
            if not self.is_recording:
                return

            logger.info("Ending recording session")
            self.is_recording = False
            if self.recorder:
                self.recorder.is_recording = False

            logger.info("Waiting for threads to finish...")
            if self.frame_thread: 
                self.frame_thread.join()
            if self.processing_thread: 
                self.processing_thread.join()
            if self.recording_thread: 
                self.recording_thread.join()
            logger.info("Threads finished")

            if self.recorder:
                self.recorder.stop_recording()

            if self.debug_mode:
                self.display_queue.put(None)

            self.frame_thread = None
            self.processing_thread = None
            self.recording_thread = None
            self.recorder = None
            logger.info("Service ready for new 'START' signal")

    def record_frames(self):
        cap = None
        logger.info("Frame capture thread started")
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Fatal error: Could not open camera")
                self.is_recording = False
                return
            logger.info("Camera initialized")
            while self.is_recording:
                ret, frame = cap.read()
                if not ret: 
                    break
                if not self.recording_queue.full(): 
                    self.recording_queue.put(frame)
                if not self.processing_queue.full(): 
                    self.processing_queue.put(frame.copy())
        finally:
            if cap: 
                cap.release()
            logger.info("Frame capture thread finished. Camera released")

    def process_frames(self, addr):
        logger.info("Processing thread started")
        processor = PoseProcessor()
        sender = PoseSender(self.udp_socket)
        try:
            while self.is_recording:
                try:
                    frame = self.processing_queue.get(timeout=1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    landmarks, pose_landmarks = processor.extract_landmarks(frame_rgb)
                    if landmarks: 
                        sender.send_landmarks(landmarks, (addr[0], 9998))
                    if pose_landmarks: 
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                        )
                    if self.debug_mode:
                        if not self.display_queue.full(): 
                            self.display_queue.put(frame)
                except queue.Empty: 
                    continue
        finally:
            processor.close()
            logger.info("Processing thread finished")
    
    def _clear_queues(self):
        for q in [self.recording_queue, self.processing_queue, self.display_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def start(self):
        listener_thread = threading.Thread(target=self.listen_for_signals, daemon=True)
        listener_thread.start()
