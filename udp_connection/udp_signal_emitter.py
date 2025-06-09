import logging

logger = logging.getLogger(__name__)


class PoseSender:
    def __init__(self, socket_obj):
        self.socket = socket_obj

    def send_landmarks(self, landmarks, addr_tuple):
        try:
            serialized_landmarks = ";".join([f"{l[0]:.4f},{l[1]:.4f},{l[2]:.4f}" for l in landmarks])
            self.socket.sendto(serialized_landmarks.encode('utf-8'), addr_tuple)
        except Exception as e:
            logger.error(f"Error sending landmarks: {e}")
