import argparse
from .video_proc import process_video


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract license plate information and prints to terminal"
    )
    ap.add_argument("videopath", help="path to the video file")
    ap.add_argument(
        "-m",
        "--model",
        help="Path to retrained model, if blank it defaults to ./runs/detect/train/weights/best.pt",
        default="./runs/detect/train/weights/best.pt",
    )
    ap.add_argument(
        "-f",
        "--frame-skip",
        help="Process every frame-skip frames. Defaults to 10.",
        default=10,
        type=int,
    )
    args = ap.parse_args()
    process_video(args.videopath, args.model, args.frame_skip)
