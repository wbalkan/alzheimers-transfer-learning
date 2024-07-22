"""
Author: Dawson Haddox
Date: 5/12/24
"""

import os
import shutil
from PIL import Image
import numpy as np

def calculate_entropy(image_path):
    img = Image.open(image_path)
    grayscale_image = img.convert("L")
    histogram = grayscale_image.histogram()
    histogram_length = sum(histogram)
    normalized_histogram = [float(h) / histogram_length for h in histogram]
    entropy = -sum(p * np.log2(p) for p in normalized_histogram if p > 0)
    return entropy

def select_images(base_path, output_path, percentage=0.4):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            entropy_list = []
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                entropy = calculate_entropy(image_path)
                entropy_list.append((entropy, image_path))
            entropy_list.sort(reverse=True, key=lambda x: x[0])
            selected_count = int(len(entropy_list) * percentage)
            selected_images = entropy_list[:selected_count]
            # Create a subfolder in the output directory
            output_folder_path = os.path.join(output_path, f"{folder} Subset Entropy")
            if not os.path.exists(output_folder_path):
                os.mkdir(output_folder_path)
            # Copy selected images to the new directory using shutil.copy2
            for _, image_path in selected_images:
                shutil.copy2(image_path, os.path.join(output_folder_path, os.path.basename(image_path)))
            print(f"Copied {selected_count} images from {folder} to {output_folder_path} based on entropy.")

base_path = "/Users/dawsonhaddox/Documents/COSC 78/Final Project/OASIS Data"
output_path = "/Users/dawsonhaddox/Documents/COSC 78/Final Project/OASIS Subset Entropy"
select_images(base_path, output_path)
