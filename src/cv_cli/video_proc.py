import cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.engine.results import Results
from dataclasses import dataclass
import easyocr
import torch

VEHICLE_CLASS_IDS = (2, 3, 5, 7)


@dataclass
class LicensePlate:
    frame: cv2.Mat
    track_id: int
    score: float

    def save(self, filename: str | None = None):
        if self.has_frame():
            if filename:
                cv2.imwrite(f"{filename}.jpg", self.frame)
            else:
                cv2.imwrite(f"{self.track_id}.jpg", self.frame)
        else:
            print("No frame to save!")

    def has_frame(self):
        return self.frame.size != 0


def detect_license_plates(
    frame: cv2.Mat, detections: Results, model_file: Path
) -> list[LicensePlate]:
    license_plate_objs = []
    license_plate_detection = YOLO(model_file)
    for det in detections.boxes.data.tolist():
        x1, y1, x2, y2, track_id, score, class_id = det
        if int(class_id) in VEHICLE_CLASS_IDS and score > 0.5:
            roi = frame[int(y1) : int(y2), int(x1) : int(x2)]
            license_plates: Results = license_plate_detection(roi)[0]
            for plate in license_plates.boxes.data.tolist():
                px1, py1, px2, py2, pscore, _ = plate
                plate_frame = roi[int(py1) : int(py2), int(px1) : int(px2)]
                lp = LicensePlate(
                    frame=plate_frame, track_id=int(track_id), score=pscore
                )
                license_plate_objs.append(lp)
                print(
                    f"license plate at bounding box ({py1}, {px1}) ({py2}, {px2}), track_id={track_id}"
                )
    return license_plate_objs


def read_license_plate(frame: cv2.Mat, reader: easyocr.Reader):
    detections = reader.readtext(frame)

    for det in detections:
        _, text, _ = det
        return text.upper()

    return None


def preprocess_license_plate(lp: LicensePlate):
    gray = cv2.cvtColor(lp.frame, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 4
    )
    return gray, thresh


def process_video(path: str, model_path: str, frame_skip: int):
    """Process the video file

    Args:
        path (str): Path of the video file
        frame_skip (int, optional): Process every {frame_skip} frames.

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
    car_detection = YOLO("yolov8s.pt")
    gpu_available = torch.cuda.is_available()
    reader = easyocr.Reader(["en"], gpu=gpu_available)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_n % frame_skip == 0:
            detections = car_detection.track(frame, persist=True)[0]
            license_plates = detect_license_plates(frame, detections, model_file)
            for i, plate in enumerate(license_plates):
                if plate.has_frame():
                    _, thresh = preprocess_license_plate(plate)
                    text = read_license_plate(thresh, reader)
                    print(text)

        frame_n += 1

    cap.release()
