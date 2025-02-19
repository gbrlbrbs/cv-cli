import cv2
from pathlib import Path
import time


def preprocess_frame(f: cv2.Mat):
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(f, 9, 10, 15)

    # edges
    edged = cv2.Canny(filtered, 30, 100)
    return gray, edged

def process_video(path: str, frame_skip=4):
    """Process the video file

    Args:
        path (str): Path of the video file
        frame_skip (int, optional): Process every {frame_skip} frames. Defaults to 4.

    Raises:
        ValueError: Whenever the file doesn't exist
        RuntimeError: Whenever the video capture is not opened
    """
    file = Path(path).resolve()
    if not file.exists():
        raise ValueError(f"Path doesn't exists: {path}")
    
    cap = cv2.VideoCapture(str(file))
    if not cap.isOpened():
        raise RuntimeError("Could not open video capture")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps

    print(f"Duration: {duration:.2f} seconds; {frame_count} frames")

    detected_plates = set()
    frame_n = 0
    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_n % frame_skip == 0:
            gray, edged = preprocess_frame(frame)


