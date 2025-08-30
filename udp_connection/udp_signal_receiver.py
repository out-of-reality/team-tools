import logging
import queue
import socket
import threading
import time
import cv2

from udp_signal_emitter import PoseSender
from video_recorder import VideoRecorder
from pose_task_processor import PoseTaskProcessor
from hand_task_processor import HandTaskProcessor

logger = logging.getLogger(__name__)

class UDPSignalReceiver:
    def __init__(self, host="localhost", port=9999, queue_size=10, debug_mode=False):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((host, port))

        self.recorder = None
        self.debug_mode = debug_mode
        self.is_recording = False
        self.shutdown_requested = False
        self.lock = threading.Lock()

        self.recording_thread = None
        self.processing_thread = None
        self.frame_thread = None
        self.hand_thread = None

        self.recording_queue = queue.Queue(maxsize=3)
        self.processing_queue = queue.Queue(maxsize=3)
        self.hand_queue = queue.Queue(maxsize=2)
        self.display_queue = queue.Queue(maxsize=1)

        self.pose_result_queue = queue.Queue()
        self.pose_processor = PoseTaskProcessor(
            self.pose_result_queue,
            self.display_queue if debug_mode else None
        )
        self.hand_processor = HandTaskProcessor()

    def listen_for_signals(self):
        logger.info("UDP listener started, waiting for signals...")
        while True:
            try:
                data, addr = self.udp_socket.recvfrom(1024)
                signal = data.decode('utf-8').strip()
                if ' ' not in signal:
                    signal = f"{signal} 12345"

                if signal.startswith("start"):
                    self._handle_start_signal(signal, addr)
                elif signal.startswith("stop"):
                    self._handle_stop_signal()
                elif signal.startswith("shutdown"):
                    self._handle_shutdown_signal()
            except Exception as e:
                logger.error("Error in UDP listener: %s", e)

    def _handle_start_signal(self, signal, addr):
        with self.lock:
            if self.is_recording:
                logger.warning("Received 'start' signal but recording is already in progress")
                return

            logger.info("Starting new recording session")
            self.camera_id = self.get_camera_id()
            cap_test = cv2.VideoCapture(self.camera_id)
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
            self.hand_thread = threading.Thread(target=self.process_hand_frames)

            self.frame_thread.start()
            self.recording_thread.start()
            self.processing_thread.start()
            self.hand_thread.start()

    def get_camera_id(self, preferred_id=0):
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cap.release()
                    return i
            except Exception:
                pass
        logger.error("No camera found.")
        return preferred_id

    def _handle_stop_signal(self):
        with self.lock:
            if not self.is_recording:
                return
            logger.info("Ending recording session")
            self.is_recording = False
            self.recorder.is_recording = False

            if self.frame_thread:
                self.frame_thread.join()
            if self.processing_thread:
                self.processing_thread.join()
            if self.hand_thread:
                self.hand_thread.join()
            if self.recording_thread:
                self.recording_thread.join()
            logger.info("Threads finished")

            self.recorder.stop_recording()
            if self.debug_mode:
                self.display_queue.put(None)

            self.recorder = None
            logger.info("Service ready for new 'START' signal")
    
    def _handle_shutdown_signal(self):
        logger.info("Requesting application termination...")
        with self.lock:
            if self.is_recording:
                self._handle_stop_signal()
            self.shutdown_requested = True

    def record_frames(self):
        try:
            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                logger.error("Fatal error: Could not open camera")
                self.is_recording = False
                return
        except Exception as e:
            logger.error("Error opening camera: %s", e)
            return

        logger.info("Camera initialized")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        logger.info("Frame capture thread started")
        frame_counter = 0

        try:
            while self.is_recording:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                frame_counter += 1
                if self.recording_queue.full():
                    try:
                        self.recording_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.recording_queue.put(frame)
                if self.processing_queue.full():
                    try:
                        self.processing_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.processing_queue.put(frame.copy())
                if frame_counter % 4 == 0:
                    if self.hand_queue.full():
                        try:
                            self.hand_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.hand_queue.put(frame.copy())
        finally:
            cap.release()

    def process_frames(self, addr):
        logger.info("Processing thread started")
        sender = PoseSender(self.udp_socket)
        while self.is_recording:
            try:
                frame = self.processing_queue.get(timeout=0.5)
                self.pose_processor.send_frame(frame)

                try:
                    landmarks_dict = self.pose_result_queue.get(timeout=0.5)
                    sender.send_landmarks(landmarks_dict, (addr[0], 9998))
                except queue.Empty:
                    pass
            except queue.Empty:
                continue

    def process_hand_frames(self):
        while self.is_recording:
            try:
                frame = self.hand_queue.get(timeout=0.5)
                self.hand_processor.send_frame(frame)
            except queue.Empty:
                continue

    def _clear_queues(self):
        for q in [self.recording_queue, self.processing_queue, self.hand_queue,
                  self.display_queue, self.pose_result_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def start(self):
        listener_thread = threading.Thread(target=self.listen_for_signals, daemon=True)
        listener_thread.start()
