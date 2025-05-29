import socket

class PoseSender:
    def __init__(self, socket_obj):
        self.socket = socket_obj

    def send_landmarks(self, landmarks, addr_tuple):
        """
        Serializa y envía las landmarks como string.
        """
        try:
            serialized = str(landmarks).encode('utf-8')
            self.socket.sendto(serialized, addr_tuple)
        except Exception as e:
            print(f"Error enviando landmarks: {e}")