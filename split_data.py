# File: src/preprocessing/split_data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits(data_dir, output_dir, train_ratio=0.7):
    """Split data into train/test CSV files."""
    # Collect all frame paths and labels
    frames = []
    for class_label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_label)
        if not os.path.isdir(class_dir):
            continue
        for frame_file in os.listdir(class_dir):
            frame_path = os.path.join(class_dir, frame_file)
            frames.append({
                "path": frame_path,
                "label": class_label
            })

    # Convert to DataFrame and split
    df = pd.DataFrame(frames)
    train_df, test_df = train_test_split(df, train_size=train_ratio, shuffle=True, random_state=42)

    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

if __name__ == "__main__":
    # For Classifier 1: Child Presence
    create_splits(
        data_dir="data/processed/frames/classifier1",
        output_dir="data/processed/splits/classifier1",
        train_ratio=0.7
    )

    # For Classifier 2: Screen Interaction
    create_splits(
        data_dir="data/processed/frames/classifier2",
        output_dir="data/processed/splits/classifier2",
        train_ratio=0.7
    )