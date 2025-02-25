import cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.engine.results import Results
# import time

VEHICLE_CLASS_IDS = (2, 3, 5, 7)


def detect_license_plates(frame: cv2.Mat, detections: Results, model_file: Path):
    
    license_plate_detection = YOLO(model_file)
    for det in detections.boxes.data.tolist():
        x1, y1, x2, y2, track_id, score, class_id = det
        if int(class_id) in VEHICLE_CLASS_IDS and score > 0.5:
            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            license_plates: Results = license_plate_detection(roi)[0]
            for plate in license_plates.boxes.data.tolist():
                px1, px2, py1, py2, pscore, _ = plate
                print(f"license plate at bounding box ({py1}, {px1}) ({py2}, {px2}), track_id={track_id}")


def process_video(path: str, model_path: str, frame_skip=4):
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
        raise ValueError(f"Path doesn't exists: {file}")
    
    cap = cv2.VideoCapture(str(file))
    if not cap.isOpened():
        raise RuntimeError("Could not open video capture")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps

    print(f"Duration: {duration:.2f} seconds; {frame_count} frames")

    frame_n = 0
    model_file = Path(model_path).resolve()
    car_detection = YOLO('yolov8s.pt')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_n % frame_skip == 0:
            detections = car_detection.track(frame, persist=True)[0]
            detect_license_plates(frame, detections, model_file)

