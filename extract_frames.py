import cv2
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Explicit mapping of folder names to labels
CLASSIFIER1_MAP = {
    # Child present scenarios
    "child_alone": "child_present",
    "child_with_parent1": "child_present",
    "child_with_parent2": "child_present",
    "child_with_both_parents": "child_present",
    "child_screen_interaction": "child_present",
    "child_parent1_screen": "child_present",
    "child_parent2_screen": "child_present",
    "child_both_parents_screen": "child_present",

    # Child not present scenarios
    "empty_room": "child_not_present",
    "parent1_alone": "child_not_present",
    "parent2_alone": "child_not_present"
}

CLASSIFIER2_MAP = {
    # Screen interaction scenarios
    "child_screen_interaction": "screen_interaction",
    "child_parent1_screen": "screen_interaction",
    "child_parent2_screen": "screen_interaction",
    "child_both_parents_screen": "screen_interaction",

    # No interaction scenarios
    "child_alone": "no_interaction",
    "child_with_parent1": "no_interaction",
    "child_with_parent2": "no_interaction",
    "child_with_both_parents": "no_interaction",
    "empty_room": "no_interaction",
    "parent1_alone": "no_interaction",
    "parent2_alone": "no_interaction"
}


def process_video(video_path, output_root_dir, frame_rate=1):
    scenario = Path(video_path).parent.name.lower()

    try:
        # Get labels from explicit mapping
        class1_label = CLASSIFIER1_MAP[scenario]
        class2_label = CLASSIFIER2_MAP[scenario]
    except KeyError:
        print(f"⚠️ Unknown folder: {scenario}. Using fallback labels.")
        class1_label = "child_present" if "child" in scenario else "child_not_present"
        class2_label = "screen_interaction" if "screen" in scenario else "no_interaction"

    # --- ADD BACK THE MISSING FRAME PROCESSING LOGIC ---
    # Create output directories
    output_dir1 = os.path.join(output_root_dir, "classifier1", class1_label)
    output_dir2 = os.path.join(output_root_dir, "classifier2", class2_label)
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps // frame_rate)

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Process frame
            frame = cv2.resize(frame, (640, 360))  # Downscale
            frame_filename = f"frame_{saved_count:04d}.jpg"

            # Save to both classifier directories
            cv2.imwrite(os.path.join(output_dir1, frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            cv2.imwrite(os.path.join(output_dir2, frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Processed {saved_count} frames → {class1_label} (Classifier1) and {class2_label} (Classifier2)")
    # --- END OF MISSING LOGIC ---


def main():
    input_dir = "D:/Projects/screen-time-monitoring-ai/data/raw_videos"
    output_dir = "data/processed/frames"

    video_paths = list(Path(input_dir).rglob("*.mp4"))
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(lambda v: process_video(str(v), output_dir), video_paths)


if __name__ == "__main__":
    main()