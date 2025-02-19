import argparse

def main():
    ap = argparse.ArgumentParser(description="Extract license plate information and prints to terminal")
    ap.add_argument("video_path", help="path to the video file")
    args = ap.parse_args()
    

if __name__ == "__main__":
    main()
