import socket
import threading
import queue
import cv2
from video_recorder import VideoRecorder
import mediapipe as mp 
from pose_processor import PoseProcessor
from udp_signal_emitter import PoseSender
from mediapipe.python.solutions.pose import PoseLandmark

class UDPSignalReceiver:
    def __init__(self, host="localhost", port=9999, queue_size=10, debug_mode=True):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((host, port))
        self.recorder = None
        self.recording_thread = None
        self.processing_thread = None # Hilo para manejar el procesamiento con MediaPipe
        self.frame_thread = None
        self.frame_queue = queue.Queue(maxsize=queue_size) # Cola que almacena los fotogramas capturados
        self.cap = None  # Para la cámara
        self.running = False # Bandera para controlar la ejecución de los hilos.
        self.debug_mode = debug_mode
        
    def listen_for_signals(self):
        """
        Al recibir la señal "start", inicializa la cámara y los hilos de grabación y procesamiento.
        
        Al recibir la señal "stop", detiene todos los hilos y libera los recursos.
        """
        try:
            while True:
                data, addr = self.udp_socket.recvfrom(1024)
                signal = data.decode('utf-8').strip()
                signal = signal + " 12345"

                if signal.startswith("start"):
                    print("Señal recibida")
                    user_id = signal.split()[1]
                    if not self.running:
                        self.running = True
                        self.start_camera()
                        self.recorder = VideoRecorder(user_id)
                        self.recorder.start_recording()

                        # Crear hilos para procesar y grabar
                        self.frame_thread = threading.Thread(target=self.record_frames)
                        self.recording_thread = threading.Thread(target=self.recorder.record, args=(self.frame_queue,))
                        self.processing_thread = threading.Thread(target=self.process_frames, args=(addr,))
                        
                        # Inicializar hilos para procesar y grabar
                        self.frame_thread.start()
                        self.recording_thread.start()
                        self.processing_thread.start()
                        
                elif signal.startswith("pause") and self.running:
                    if self.recorder:
                        self.recorder.paused = True  # Actualiza el estado a pausado

                elif signal.startswith("resume") and self.running:
                    if self.recorder:
                        self.recorder.paused = False  # Actualiza el estado a reanudado   

                elif signal.startswith("stop") and self.running:
                    self.stop_all()
                    
        finally:
            print("Cerrando conexion")
            self.udp_socket.close()
            
    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            print("Inicializando la cámara...")
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                print("Cámara inicializada correctamente.")
            else:
                print("Error al inicializar la cámara.")
        else:
            print("No entre a start_camara")

    def record_frames(self):
        """
        Bucle que captura fotogramas de la cámara y los coloca en -self.frame_queue-
        siempre que esté en ejecución (self.running)
        """
        while self.running:
            
            if not self.cap.isOpened():
                print("Cámara no está lista.")
                break
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                else:
                    self.frame_queue.get()  # Discard the oldest frame
                    self.frame_queue.put(frame)
            else:
                break
            
    def process_frames(self, addr):
        """
        Crea una instancia de MediaPipe Pose (mp.solutions.pose).
        Un bucle que consume fotogramas de la cola (self.frame_queue) y los procesa con MediaPipe para obtener los puntos del cuerpo.
        Serializa y envía los puntos a la dirección del cliente utilizando el socket UDP (self.udp_socket.sendto).
        """
        processor = PoseProcessor()
        sender = PoseSender(self.udp_socket)
        try:
            while self.running:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    try:
                        landmarks, pose_landmarks = processor.extract_landmarks(frame)

                        if landmarks:
                            landmarks_named = { PoseLandmark(idx).name: [x, y, z] for idx, (x, y, z) in enumerate(landmarks)}
                            client_address = (addr[0], 9998)
                            sender.send_landmarks(landmarks_named, client_address)

                        if self.debug_mode and pose_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(
                                frame,
                                pose_landmarks,
                                mp.solutions.pose.POSE_CONNECTIONS
                            )
                            cv2.imshow('Debug - Pose Detection', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                    except Exception as e:
                        print(f"Error procesando frame: {e}")
        finally:
            processor.close()
            print("Hilo processing_thread terminado. (process_frames)")
            
    def stop_all(self):
        """
        Detiene los hilos y libera los recursos de la cámara (self.cap.release()).
        Llama a stop_recording de la clase VideoRecorder.
        """
        print('-- Entre a la función STOP')
        print("Deteniendo todos los procesos...")
        self.running = False
        print(f"1: {self.running}")
        
        if self.frame_thread:
            self.frame_thread.join()
            self.frame_thread = None
        print(f"2: {self.frame_thread}")    
        
        if self.recording_thread:
            print("Esperando a que termine frame_thread...")
            self.recording_thread.join()
            self.recording_thread = None
            print("Hilo frame_thread terminado.")
        print(f"3: {self.recording_thread}")
            
        if self.processing_thread:
            print("Esperando a que termine recording_thread...")
            self.processing_thread.join()
            self.processing_thread = None
            print("Hilo recording_thread terminado.")
        print(f"4: {self.processing_thread}")
        
        if self.cap:
            print("Esperando a que termine processing_thread...")
            self.cap.release()
            self.cap = None
            print("Hilo processing_thread terminado.")
        print(f"5: {self.cap}")

        if self.recorder:
            self.recorder.stop_recording()
            self.recorder = None
        print(f"6: {self.recorder}")

        # Vaciar la cola de frames
        self.frame_queue = queue.Queue()
        
        # cv2.destroyAllWindows()
        cv2.destroyAllWindows()
        
        print("Todos los procesos detenidos y recursos liberados.")
    
    def start(self):
        listener_thread = threading.Thread(target=self.listen_for_signals)
        listener_thread.start()
