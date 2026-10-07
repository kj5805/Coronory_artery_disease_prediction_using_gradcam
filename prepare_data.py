"""
Data Preparation Script for CAD Risk Classification
Uses EchoNet-Dynamic dataset with EF-based labeling

Normal: EF >= 50%
Abnormal (CAD-risk): EF < 40%
"""

import pandas as pd
import numpy as np
import os
import shutil
import argparse
from pathlib import Path

def prepare_cad_dataset(data_dir, output_dir):
    """
    Organize EchoNet-Dynamic data into binary classification format

    Args:
        data_dir: Path to EchoNet-Dynamic root directory
        output_dir: Where to save organized data
    """

    # Read the FileList.csv which contains EF values
    filelist_path = os.path.join(data_dir, 'a4c-video-dir', 'FileList.csv')
    df = pd.read_csv(filelist_path)

    print(f"Total videos in dataset: {len(df)}")
    print(f"\nEjection Fraction statistics:")
    print(df['EF'].describe())

    # Define classes based on EF
    # Normal: EF >= 50%
    # Abnormal (CAD-risk): EF < 40%
    # We exclude the borderline cases (40 <= EF < 50) for clearer classification

    normal_df = df[df['EF'] >= 50].copy()
    abnormal_df = df[df['EF'] < 40].copy()

    print(f"\n--- Class Distribution ---")
    print(f"Normal (EF >= 50%): {len(normal_df)} videos")
    print(f"Abnormal/CAD-risk (EF < 40%): {len(abnormal_df)} videos")
    print(f"Excluded (borderline, 40 <= EF < 50): {len(df[(df['EF'] >= 40) & (df['EF'] < 50)])} videos")

    # Add binary labels
    normal_df['CAD_Label'] = 0  # Normal
    abnormal_df['CAD_Label'] = 1  # CAD-risk

    # Combine
    cad_dataset = pd.concat([normal_df, abnormal_df], ignore_index=True)

    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the labeled dataset
    output_csv = output_path / 'CAD_FileList.csv'
    cad_dataset.to_csv(output_csv, index=False)

    print(f"\n✅ Dataset prepared successfully!")
    print(f"Saved to: {output_csv}")
    print(f"\nFinal dataset size: {len(cad_dataset)} videos")
    print(f"  - Normal: {len(cad_dataset[cad_dataset['CAD_Label'] == 0])}")
    print(f"  - Abnormal: {len(cad_dataset[cad_dataset['CAD_Label'] == 1])}")

    # Create train/val/test splits
    # We'll use the existing Split column from EchoNet
    train_df = cad_dataset[cad_dataset['Split'] == 'TRAIN']
    val_df = cad_dataset[cad_dataset['Split'] == 'VAL']
    test_df = cad_dataset[cad_dataset['Split'] == 'TEST']

    print(f"\n--- Split Distribution ---")
    print(f"Train: {len(train_df)} videos")
    print(f"  Normal: {len(train_df[train_df['CAD_Label'] == 0])}")
    print(f"  Abnormal: {len(train_df[train_df['CAD_Label'] == 1])}")

    print(f"\nValidation: {len(val_df)} videos")
    print(f"  Normal: {len(val_df[val_df['CAD_Label'] == 0])}")
    print(f"  Abnormal: {len(val_df[val_df['CAD_Label'] == 1])}")

    print(f"\nTest: {len(test_df)} videos")
    print(f"  Normal: {len(test_df[test_df['CAD_Label'] == 0])}")
    print(f"  Abnormal: {len(test_df[test_df['CAD_Label'] == 1])}")

    return cad_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create binary CAD-risk labels from EchoNet-Dynamic metadata."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/echonet"),
        help="EchoNet-Dynamic data root containing a4c-video-dir/FileList.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cad"),
        help="Directory in which CAD_FileList.csv is written.",
    )
    args = parser.parse_args()

    print("="*60)
    print("EchoNet-Dynamic to CAD Dataset Preparation")
    print("="*60)

    dataset = prepare_cad_dataset(args.data_dir, args.output_dir)

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Review the CAD_FileList.csv file")
    print("2. Proceed to train the CAD classification model")
    print("3. Implement Grad-CAM for explainability")
