import logging
import json

logger = logging.getLogger(__name__)


class PoseSender:
    def __init__(self, socket_obj):
        self.socket = socket_obj

    def send_landmarks(self, landmarks_dict, addr_tuple):
        try:
            json_data = json.dumps(landmarks_dict)
            serialized = json_data.encode('utf-8')
            self.socket.sendto(serialized, addr_tuple)
        except Exception as e:
            print(f"Error enviando landmarks: {e}")
