# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:03:42 2024

@author: Experimenter
"""

import cv2
import numpy as np
import deeplabcut
import os
import pandas as pd
import matplotlib.pyplot as plt

#%% extract random frames from the video

def extract_random_frames(video_path, output_folder, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sorted(np.random.choice(length, num_frames, replace=False))

    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{output_folder}/frame_{frame_index}.png", frame)
        else:
            print(f"Failed to extract frame: {frame_index}")

    cap.release()

video_path = r"Q:\pupil\headplate_holder\test_videos\Notos_2023-02-21_Video1_640_480.avi"
output_folder = r"Q:\pupil\headplate_holder\test_videos"
extract_random_frames(video_path, output_folder)

#%% 
def analyze_frames_with_dlc(config_path, frames_directory, frame_file_type='.png'):
    """
    Analyze the saved frames using DeepLabCut's analyze_time_lapse_frames function.

    Parameters:
    config_path (str): Absolute path to the DeepLabCut config file.
    frames_directory (str): Directory where the frames are stored.
    frame_file_type (str): The file extension of the frames to be analyzed.
    """
    
    deeplabcut.analyze_time_lapse_frames(config_path, frames_directory, frametype=frame_file_type, 
                                         shuffle=1, trainingsetindex=0, gputouse=None, 
                                         save_as_csv=True)

config_path = r"C:\Users\Experimenter\Desktop\HeadplateHolder-SchroederLab-2024-01-29\config.yaml"
frames_directory = r"Q:\pupil\headplate_holder\test_videos\frames"
analyze_frames_with_dlc(config_path, frames_directory)


#%% 

#TODO: maybe include this function in analyze_frames_with_dlc and pass it as a parameter?

def create_labeled_images(df, frames_directory, output_directory):
    for index, row in df.iterrows():
        frame_path = os.path.join(frames_directory, index)
        frame = cv2.imread(frame_path)
        if frame is not None:
            # Draw keypoints
            for bodypart in df.columns.levels[0]:
                x, y = row[(bodypart, 'x')], row[(bodypart, 'y')]
                likelihood = row[(bodypart, 'likelihood')]
                if likelihood > 0.6:  # You can adjust this threshold
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            # Save the labeled frame
            output_path = os.path.join(output_directory, index)
            cv2.imwrite(output_path, frame)

# Load the DataFrame from the CSV file
df_csv_path = r"Q:\pupil\headplate_holder\test_videos\frames\framesDLC_resnet50_HeadplateHolderJan29shuffle1_100000.csv"
df_csv = pd.read_csv(df_csv_path, header=[1, 2], index_col=0)

# Directories
frames_directory = r"Q:\pupil\headplate_holder\test_videos\frames"
output_directory = r"Q:\pupil\headplate_holder\test_videos\frames\labeled"

create_labeled_images(df_csv, frames_directory, output_directory)

#%% 

def calculate_distances(df):
    distances = []
    for _, row in df.iterrows():
        x1, y1 = row[('top', 'x')], row[('top', 'y')]
        x2, y2 = row[('bottom', 'x')], row[('bottom', 'y')]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distances.append(distance)
    return distances


def calculate_median_and_std(distances):
    """
    Calculate the median and standard deviation of distances.
    
    Parameters:
    distances (list): List of distances between markers in each frame.
    
    Returns:
    tuple: median distance, standard deviation of distances
    """
    median_distance = np.median(distances)
    std_deviation = np.std(distances)
    return median_distance, std_deviation

# Load the DataFrame from the CSV file
df_csv_path = r"Q:\pupil\headplate_holder\test_videos\frames\framesDLC_resnet50_HeadplateHolderJan29shuffle1_100000.csv"
df_csv = pd.read_csv(df_csv_path, header=[1, 2], index_col=0)

# Calculate distances and statistics
distances = calculate_distances(df_csv)
median_distance, std_deviation = calculate_median_and_std(distances)

# Print results
print("Median Distance:", median_distance)
print("Standard Deviation:", std_deviation)


#%% 

# Load the diameter values
diameter_file_path = r"Q:\pupil\headplate_holder\Elias\2022-08-03\pupil\xyPos_diameter\1\eye.diameter.npy"
output_file_path = r"Q:\pupil\headplate_holder\Elias\2022-08-03\pupil\xyPos_diameter\1"


def convert_and_save_diameters(diameter_pixels_file_path, output_folder_path, hh_median_distance, hh_actual_size_mm=4):
    """
    Convert pupil diameters from pixels to millimeters and save the result to a file.

    Parameters:
    diameter_pixels_file_path (str): Path to the .npy file with diameter measurements in pixels.
    output_file_path (str): Path where the converted diameters will be saved.
    median_distance_HH (float): The median distance measured between markers in pixels.
    actual_size_mm (float): The actual size of the object in millimeters (default is 4 mm).
    """
    # Load the diameter values
    diameters_pixels = np.load(diameter_pixels_file_path)

    # Calculate the scale in mm per pixel
    scale_mm_per_pixel = hh_actual_size_mm / hh_median_distance

    # Convert each diameter from pixels to millimeters
    diameters_mm = [d * scale_mm_per_pixel for d in diameters_pixels]
    
    # Construct the output file path
    filename = os.path.basename(diameter_pixels_file_path)
    output_filename = os.path.splitext(filename)[0] + '_mm.npy'
    output_file_path = os.path.join(output_folder_path, output_filename)

    # Save the converted diameters
    np.save(output_file_path, np.array(diameters_mm))
    
    return diameters_mm

# Example usage
diameter_pixels_file_path = r"Q:\pupil\headplate_holder\Elias\2022-08-03\pupil\xyPos_diameter\1\eye.diameter.npy"
output_folder_path =  r"Q:\pupil\headplate_holder\Elias\2022-08-03\pupil\xyPos_diameter\1"
hh_median_distance = 53.51497563662302  # Example value from your previous calculation

convert_and_save_diameters(diameter_pixels_file_path, output_folder_path, hh_median_distance)

#%% 

output_file_path_mm = r"Q:\pupil\headplate_holder\Elias\2022-08-03\pupil\xyPos_diameter\1\eye.diameter_mm.npy"
diameters_mm = np.load(output_file_path_mm)

# Assuming you have a similar path for the diameters in pixels
output_file_path_pixels = r"Q:\pupil\headplate_holder\Elias\2022-08-03\pupil\xyPos_diameter\1\eye.diameter.npy"
diameters_pixels = np.load(output_file_path_pixels)

# Plotting the diameters in millimeters
plt.figure(figsize=(10, 12))  # Increased figure size for two subplots

plt.subplot(2, 1, 1)  # (rows, columns, panel number)
plt.plot(diameters_mm, label='Pupil Diameter (mm)')
plt.xlabel('Frame Number')
plt.ylabel('Diameter (mm)')
plt.title('Pupil Diameter Over Time in Millimeters')
plt.legend()

# Plotting the diameters in pixels
plt.subplot(2, 1, 2)  # (rows, columns, panel number)
plt.plot(diameters_pixels, label='Pupil Diameter (pixels)', color='orange')
plt.xlabel('Frame Number')
plt.ylabel('Diameter (pixels)')
plt.title('Pupil Diameter Over Time in Pixels')
plt.legend()

plt.tight_layout()  # To ensure there's no overlap of subplots
plt.show()









