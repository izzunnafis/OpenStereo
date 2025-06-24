import os
import re
from collections import defaultdict
import argparse

def extract_info(filename):
    # Handle both .jpg and .png, and optional 'disparity' prefix
    match = re.match(r"(left|right|disparity)_image_(\d+)_(\d+)_(\d+)_(\d+)\.(jpg|png|npy)", filename)
    if match:
        side = match.group(1)
        date = match.group(2)
        time = match.group(3)
        millis = match.group(4)
        number = int(match.group(5))
        # Optionally, you can combine date and time as timestamp if needed
        timestamp = int(f"{date}{time}{millis}")
        return side, timestamp, number
    return None, None, None

def pair_images(folder_path):
    left_images = []
    right_images = []
    disparity_images = []

    # Collect all files and classify
    for file in os.listdir(folder_path):
        if file.endswith((".jpg", ".png")):
            side, timestamp, number = extract_info(file)
            if side is not None and timestamp is not None and number is not None:
                if side == "left":
                    left_images.append((file, timestamp, number))
                elif side == "right":
                    right_images.append((file, timestamp, number))
        if file.endswith(".npy") and "disparity" in file:
            side, timestamp, number = extract_info(file)
            if side == "disparity" and timestamp is not None and number is not None:
                disparity_images.append((file, timestamp, number))
                

    # Group images by timestamp
    grouped = defaultdict(dict)
    for filename, timestamp, number in left_images:
        grouped[timestamp]['left'] = filename
    for filename, timestamp, number in right_images:
        grouped[timestamp]['right'] = filename
    for filename, timestamp, number in disparity_images:
        grouped[timestamp]['disparity'] = filename

    # Make pairs of 3 (left, right, disparity) if all exist
    triplets = []
    for timestamp, files in grouped.items():
        if 'left' in files and 'right' in files and 'disparity' in files:
            triplets.append((files['left'], files['right'], files['disparity']))

    return triplets

def save_pairs_to_txt(pairs, output_path):
    with open(output_path, "w") as f:
        for left, right, disparity in pairs:
            f.write(f"{left} {right} {disparity}\n")


if __name__ == "__main__":
    folder_path = "/home/rispro-sils/ADAS_Kedaireka/Dataset/manual_lab/output_images"
    output_file = "/home/rispro-sils/ADAS_Kedaireka/Perception/OpenStereo/data/manual/zed_file_new.txt"
    pairs = pair_images(folder_path)
    save_pairs_to_txt(pairs, output_file)
    
    print(f"Pairs saved to {output_file}")
    print(f"Found {len(pairs)} pairs of images.")