import PyInstaller.__main__
import os
import sys

APP_NAME = "GestureControlApp"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pose_model = os.path.join(BASE_DIR, "pose_landmarker_lite.task")
hand_model = os.path.join(BASE_DIR, "hand_landmarker.task")

sep = ";" if os.name == "nt" else ":"

PyInstaller.__main__.run([
    "main.py",
    f"--name={APP_NAME}",
    "--onedir",
    "--clean",
    "--noconfirm",
    "--hidden-import=mediapipe"
    "--hidden-import=cv2",
    "--hidden-import=numpy",
    "--hidden-import=pyautogui",
    f"--add-data={pose_model}{sep}dependencies",
    f"--add-data={hand_model}{sep}dependencies",
])

print(f"\nBuild finished! Check dist/{APP_NAME}/ for your app.\n")
