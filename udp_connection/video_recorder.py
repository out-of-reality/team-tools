import cv2
import requests
import os
from datetime import datetime
import config
from urllib.parse import urljoin
from contextlib import suppress
import time


class VideoRecorder:
    def __init__(self, user_id, frame_size=(640, 480), fps=30):
        #self.cap = None
        self.out = None
        self.recording = False
        self.user_id = user_id
        self.frame_size = frame_size
        self.fps = fps
        self.video_name = None
        self.paused = False #Nuevo atributo

    def start_recording(self):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_name = f"{self.user_id}_{now}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        self.out = cv2.VideoWriter(self.video_name, fourcc, self.fps, self.frame_size)
        self.recording = True
        print(f"Recording started, saving to {self.video_name}")

    def record(self, frame_queue):
        try:
            print("Iniciando grabación en record...")
            empty_frame_count = 0  # Contador para frames vacíos consecutivos
            while self.recording:
                if self.paused:
                    print("Grabación pausada.")
                    time.sleep(0.1)  # Esperar mientras está en pausa
                    continue
                
                if not frame_queue.empty():
                    frame = frame_queue.get()
                    if frame is not None:
                        self.out.write(frame)
                        #print("Frame grabado correctamente.")
                        empty_frame_count = 0  # Reiniciar contador si se graba un frame
                    else:
                        pass
                        #print("Frame inválido detectado.")
                else:
                    #print("Cola de frames vacía. Esperando frames...")
                    empty_frame_count += 1
                    time.sleep(0.1)  # Esperar un poco antes de revisar nuevamente

                    # Si la cola permanece vacía por un tiempo prolongado, salir del bucle
                    if empty_frame_count > 50:  # Por ejemplo, 5 segundos (50 * 0.1s)
                        print("Cola de frames vacía por demasiado tiempo. Finalizando grabación.")
                        break
        finally:
            self.stop_recording()

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if self.out is not None:
            print(f"Liberando VideoWriter para el archivo: {self.video_name}")
            self.out.release()
        print(f"Grabación detenida, video guardado como {self.video_name}")
        #self.send_video_to_api()
        #self.cleanup_video_file()

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
