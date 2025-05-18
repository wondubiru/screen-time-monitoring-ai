import cv2
import os
import albumentations as A
from pathlib import Path

# Define augmentations (customize as needed)
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.4),
    A.Rotate(limit=15, p=0.5),
    A.GaussianBlur(p=0.3),
    A.CoarseDropout(max_holes=5, max_height=20, max_width=20, p=0.3),
])

def augment_frames(input_dir, output_dir, num_augments=3):
    """Augment frames while preserving directory structure."""
    # Iterate through all class folders
    for class_dir in Path(input_dir).glob("*/*"):  # e.g., classifier1/child_present
        if not class_dir.is_dir():
            continue

        # Create mirrored output directory
        relative_path = class_dir.relative_to(input_dir)
        output_class_dir = Path(output_dir) / relative_path
        output_class_dir.mkdir(parents=True, exist_ok=True)

        # Process all images in the class folder
        for img_path in class_dir.glob("*.jpg"):  # Adjust extension if needed
            # Load original frame
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            # Save original to augmented folder (optional)
            original_output = output_class_dir / f"orig_{img_path.name}"
            cv2.imwrite(str(original_output), frame)

            # Generate augmented versions
            for i in range(num_augments):
                augmented = augmenter(image=frame)["image"]
                aug_output = output_class_dir / f"aug_{i}_{img_path.name}"
                cv2.imwrite(str(aug_output), augmented)

# Example usage
augment_frames(
    input_dir="data/processed/frames",
    output_dir="data/augmented_frames",
    num_augments=3  # 3 augmented versions per frame
)