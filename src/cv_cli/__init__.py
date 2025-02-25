import argparse
from .video_proc import process_video

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract license plate information and prints to terminal")
    ap.add_argument("video_path", help="path to the video file")
    ap.add_argument("-m", "--model", help="Path to retrained model, if blank it defaults to ./runs/detect/train/best.pt", default="./runs/detect/train/best.pt")
    args = ap.parse_args()
    process_video(args.video_path, args.model)