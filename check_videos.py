"""
Diagnostic script to check video files and dataset
"""

import pandas as pd
import os
import cv2

# Paths
DATA_DIR = "C:/Users/anany/dynamic"
CAD_CSV = "C:/Users/anany/dynamic/cad_data/CAD_FileList.csv"
VIDEO_DIR = "C:/Users/anany/dynamic/a4c-video-dir/Videos"  # Correct path

print("="*60)
print("VIDEO FILES DIAGNOSTIC CHECK")
print("="*60)

# Check if CSV exists
if not os.path.exists(CAD_CSV):
    print(f"\n❌ CAD_FileList.csv NOT FOUND at: {CAD_CSV}")
    print("\nYou need to run: python prepare_cad_data.py first!")
    exit()

print(f"\n✅ CAD_FileList.csv found")

# Load the CSV
df = pd.read_csv(CAD_CSV)
print(f"\nTotal videos in dataset: {len(df)}")
print(f"  - Normal (label=0): {len(df[df['CAD_Label']==0])}")
print(f"  - CAD-risk (label=1): {len(df[df['CAD_Label']==1])}")

# Check video directory
if not os.path.exists(VIDEO_DIR):
    print(f"\n❌ Videos directory NOT FOUND at: {VIDEO_DIR}")
    exit()

print(f"\n✅ Videos directory found at: {VIDEO_DIR}")

# Count actual video files
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]
print(f"\nActual .avi files in Videos folder: {len(video_files)}")

# Check first 10 videos from CSV
print("\n" + "="*60)
print("CHECKING FIRST 10 VIDEOS FROM DATASET")
print("="*60)

good_count = 0
bad_count = 0

for i in range(min(10, len(df))):
    filename = df.iloc[i]['FileName']
    
    # Try different path combinations
    video_path_1 = os.path.join(VIDEO_DIR, filename)
    video_path_2 = os.path.join(VIDEO_DIR, filename + '.avi')
    
    exists_1 = os.path.exists(video_path_1)
    exists_2 = os.path.exists(video_path_2)
    
    print(f"\n{i+1}. {filename}")
    print(f"   Path 1 (as-is): {video_path_1}")
    print(f"   Exists: {exists_1}")
    print(f"   Path 2 (with .avi): {video_path_2}")
    print(f"   Exists: {exists_2}")
    
    # Try to open the video
    if exists_2:
        cap = cv2.VideoCapture(video_path_2)
        can_open = cap.isOpened()
        if can_open:
            ret, frame = cap.read()
            can_read = ret and frame is not None
            print(f"   ✅ Can open: {can_open}, Can read frame: {can_read}")
            good_count += 1
        else:
            print(f"   ❌ Cannot open video")
            bad_count += 1
        cap.release()
    elif exists_1:
        cap = cv2.VideoCapture(video_path_1)
        can_open = cap.isOpened()
        if can_open:
            ret, frame = cap.read()
            can_read = ret and frame is not None
            print(f"   ✅ Can open: {can_open}, Can read frame: {can_read}")
            good_count += 1
        else:
            print(f"   ❌ Cannot open video")
            bad_count += 1
        cap.release()
    else:
        print(f"   ❌ FILE NOT FOUND")
        bad_count += 1

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Good videos: {good_count}/10")
print(f"Bad videos: {bad_count}/10")

if bad_count > 0:
    print("\n⚠️ Some videos are missing or corrupted")
    print("This is normal - the training script will skip them automatically")
else:
    print("\n✅ All checked videos are good!")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print("The training script will automatically skip corrupted videos.")
print("You should be able to train with the good videos.")
print("\nRun: python train_cad_model.py")