import argparse
from .video_proc import process_video

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract license plate information and prints to terminal")
    ap.add_argument("video_path", help="path to the video file")
    args = ap.parse_args()
    process_video(args.video_path)