import cv2
import os

VIDEO_DIR = "./Dataset"
FRAME_DIR = "./Frames"

FPS_SAMPLE = 1  # extract 1 frame per second

os.makedirs(FRAME_DIR, exist_ok=True)

for label in ["Fake", "Real"]:
    video_folder = os.path.join(VIDEO_DIR, label)
    out_folder = os.path.join(FRAME_DIR, label)
    os.makedirs(out_folder, exist_ok=True)

    for video_file in os.listdir(video_folder):
        if not video_file.endswith(".mp4"):
            continue

        video_id = video_file.replace(".mp4", "")
        video_path = os.path.join(video_folder, video_file)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = max(1, int(fps * FPS_SAMPLE))

        video_out_dir = os.path.join(out_folder, video_id)
        os.makedirs(video_out_dir, exist_ok=True)

        frame_idx = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                cv2.imwrite(
                    os.path.join(video_out_dir, f"frame_{saved:04d}.jpg"),
                    frame
                )
                saved += 1

            frame_idx += 1

        cap.release()
        print(f"Extracted {saved} frames from {video_file}")
