import os
import re
from collections import defaultdict
import argparse

def extract_info(filename):
    match = re.match(r"image_(left|right)_(\d+)_(\d+)\.jpg", filename)
    if match:
        side = match.group(1)
        group = int(match.group(2))
        timestamp = int(match.group(3))
        return side, group, timestamp
    return None, None, None

def pair_images(folder_path):
    left_images = []
    right_images = []

    # Collect all files and classify
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            side, group, timestamp = extract_info(file)
            if side == "left":
                left_images.append((file, group, timestamp))
            elif side == "right":
                right_images.append((file, group, timestamp))

    # Group right images by group for faster lookup
    right_grouped = defaultdict(list)
    for filename, group, timestamp in right_images:
        right_grouped[group].append((filename, timestamp))

    # Pairing
    pairs = []
    for left_file, group, left_time in left_images:
        if group in right_grouped:
            # Find right image with smallest time diff
            closest = min(right_grouped[group], key=lambda x: abs(x[1] - left_time))
            pairs.append((left_file, closest[0]))

    return pairs

def save_pairs_to_txt(pairs, output_path):
    with open(output_path, "w") as f:
        for left, right in pairs:
            f.write(f"{left} {right}\n")


if __name__ == "__main__":
    folder_path = "/home/rispro-sils/ADAS_Kedaireka/Dataset/Manually_gathered_17_07_24/zed_cam/test"
    output_file = "/home/rispro-sils/ADAS_Kedaireka/Perception/OpenStereo/data/manual/zed_file.txt"
    pairs = pair_images(folder_path)
    save_pairs_to_txt(pairs, output_file)
    
    print(f"Pairs saved to {output_file}")
    print(f"Found {len(pairs)} pairs of images.")