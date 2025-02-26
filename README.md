# cv-cli

## How to run

It is recommended to use [uv](https://docs.astral.sh/uv/) for package management. The `pyproject.toml` file has a script defitinion that points to `src/cv-cli/__init__.py`, so you can run with `uv run cli path_to_video_file`.

## Models used

This project uses 3 models:

1. A YOLOv8 for car detection;
2. A retrained YOLOv8 (retraining is in the `notebooks/retrain_yolo.ipynb`) for license plate detection;
3. An EasyOCR reader for license plate reading.

OpenCV is used for image processing for use in OCR.

## Likely issues

EasyOCR may raise an error about SSL certificates, so check if you are using correct certificates and set the `SSL_CERT_FILE` environment variable.