"""
Author: Dawson Haddox
Date: 5/9/24
Purpose: This script is designed to split the subset MRI scan data from the "OASIS Data Subset" into training, 
validation, and test sets, ensuring that all scans from the same participant are kept together in the same set. 
The separation of scans by participant ID prevents data leakage and maintains the integrity of the dataset.
This split facilitates effective training, tuning, and testing of models for AD classification based on MRI scans.
The split is performed on each class separately, ensuring a proportional number of participants in each class.
"""

import os
import shutil
from sklearn.model_selection import train_test_split

def split_data_by_category(src_directory, train_directory, val_directory, test_directory, train_size=0.7, val_size=0.15, test_size=0.15):
    # Create directories if they do not exist
    for directory in [train_directory, val_directory, test_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Process each category
    categories = [d for d in os.listdir(src_directory) if os.path.isdir(os.path.join(src_directory, d))]
    for category in categories:
        # Dictionary to hold participant ID and their files
        participant_files = {}
        category_path = os.path.join(src_directory, category)
        files = os.listdir(category_path)

        # Group files by participant
        for file in files:
            participant_id = file.split('_')[1]  # Assumes ID is second element in filename
            if participant_id not in participant_files:
                participant_files[participant_id] = []
            participant_files[participant_id].append(os.path.join(category_path, file))

        # Split participants into train, val, and test
        participants = list(participant_files.keys())
        train_participants, test_val_participants = train_test_split(participants, train_size=train_size, test_size=val_size + test_size, random_state=42)
        val_participants, test_participants = train_test_split(test_val_participants, train_size=val_size / (val_size + test_size), random_state=42)

        # Function to copy files of participants to the designated folder
        def copy_files(participants, directory, category):
            destination_dir = os.path.join(directory, category)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            for participant in participants:
                for file_path in participant_files[participant]:
                    shutil.copy(file_path, destination_dir)

        # Copy participant files to respective directories
        copy_files(train_participants, train_directory, category)
        copy_files(val_participants, val_directory, category)
        copy_files(test_participants, test_directory, category)

        print(f"Data split into training, validation, and test sets for {category}.")

# Specify the paths
src_directory = 'OASIS Data Subset'
train_directory = 'Train Data'
val_directory = 'Validation Data'
test_directory = 'Test Data'

# Split the data
split_data_by_category(src_directory, train_directory, val_directory, test_directory)
