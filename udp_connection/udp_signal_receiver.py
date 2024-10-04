import socket
import threading
from video_recorder import VideoRecorder


class UDPSignalReceiver:
    def __init__(self, host="localhost", port=9999):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((host, port))
        self.recorder = None
        self.recording_thread = None

    def listen_for_signals(self):
        try:
            while True:
                data, addr = self.udp_socket.recvfrom(1024)
                signal = data.decode('utf-8')

                if signal.startswith("start"):
                    user_id = signal.split()[1]
                    if self.recorder and self.recorder.recording:
                        continue
                    else:
                        self.recorder = VideoRecorder(user_id)
                        self.recorder.start_recording()
                        self.recording_thread = threading.Thread(target=self.recorder.record)
                        self.recording_thread.start()

                elif signal == "stop" and self.recorder:
                    self.recorder.stop_recording()
                    if self.recording_thread:
                        self.recording_thread.join()
        finally:
            self.udp_socket.close()

    def start(self):
        listener_thread = threading.Thread(target=self.listen_for_signals)
        listener_thread.start()
