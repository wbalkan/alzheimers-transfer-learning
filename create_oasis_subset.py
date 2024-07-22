"""
Author: Dawson Haddox
Date: 5/29/24
Purpose: This script is designed to create a subset of the OASIS dataset for Alzheimer's disease classification. 
It randomly selects 20% of the MRI scans from each category ("Non Demented", "Very Mild Dementia", "Mild Dementia", "Moderate Dementia") 
in the "OASIS Data" folder and copies them into corresponding subfolders within a new "OASIS Data Subset" directory. 
This helps in managing dataset size for preliminary analysis or testing machine learning models.
The script could also be used for other datasets and subset_ratios by changing the parameters.
"""

import os
import shutil
import random

def subset_data(src_directory, dest_directory, subset_ratio=0.4):
    # Ensure the destination directory exists
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    # List all subdirectories in the source directory
    categories = [d for d in os.listdir(src_directory) if os.path.isdir(os.path.join(src_directory, d))]

    for category in categories:
        src_folder = os.path.join(src_directory, category)
        dest_folder = os.path.join(dest_directory, category + ' Subset')

        # Ensure the destination subfolder exists
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # List all files in the source subfolder
        files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

        # Randomly sample files
        sample_size = int(len(files) * subset_ratio)
        sampled_files = random.sample(files, sample_size)

        # Copy the sampled files to the destination folder
        for file in sampled_files:
            src_file_path = os.path.join(src_folder, file)
            dest_file_path = os.path.join(dest_folder, file)
            shutil.copy2(src_file_path, dest_file_path)

        print(f"Copied {sample_size} files from {category} to {category + ' Subset'}.")

# Set the path to your source and destination directories
src_directory = "/Users/dawsonhaddox/Documents/COSC 78/Final Project/OASIS Data"
dest_directory = "/Users/dawsonhaddox/Documents/COSC 78/Final Project/OASIS Data Subset"

# Subset the data
subset_data(src_directory, dest_directory)
