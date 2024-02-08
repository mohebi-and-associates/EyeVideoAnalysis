# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:09:08 2024

@author: Experimenter
"""

from moviepy.editor import VideoFileClip
import cv2
import os
import glob



#%% resize step-wise

def resize_video_stepwise(input_path, output_folder, steps):
    video_clip = VideoFileClip(input_path)
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    
    for width, height in steps:
        output_path = os.path.join(output_folder, f"{name}_{width}_{height}{ext}")
        # Resize the video clip to the target resolution
        resized_clip = video_clip.resize(newsize=(width, height))
        # Write the resized video clip to a new file
        resized_clip.write_videofile(output_path, codec='mpeg4', bitrate='8000k')  # for MP4 files

# Define a list of resolutions to step down through
steps_FG006_Elias = [
    (640, 480),  # VGA
    (480, 360),
    (320, 240),  # QVGA
]

steps_FG002 = [
    (1280, 960),
    (800, 600),   # SVGA
    (640, 480),   # VGA
    (320, 240),   # QVGA
]

# Call the function for each video
# resize_video_stepwise(r"Z:\RawData\FG006\2023-11-09\1\Video0.avi", r"C:\Users\Experimenter\Deeplabcut_files\pupil_analysis\videos", steps_FG006_Elias)
resize_video_stepwise(r"Z:\RawData\Notos\2023-02-21\1\Video1.avi", r"C:\Users\Experimenter\Deeplabcut_files\pupil_analysis\videos", steps_FG006_Elias)
# resize_video_stepwise(r"Z:\RawData\FG002\2022-10-27\1\Video0.avi", r"C:\Users\Experimenter\Deeplabcut_files\pupil_analysis\videos", steps_FG002)

#%% standard resize with bitrate

# def resize_video(input_path, output_path, target_resolution, resampling_method='lanczos'):
#     video_clip = VideoFileClip(input_path)
#     # Resize the video clip to the target resolution using the specified resampling method
#     resized_clip = video_clip.resize(newsize=target_resolution)
#     # Write the resized video clip to a new file with a high bitrate to maintain quality
#     resized_clip.write_videofile(output_path, codec='mpeg4', bitrate='8000k')  # Adjust bitrate as needed

# # Example usage
# input_video_path = r"Z:\RawData\FG006\2023-11-09\1\Video0.avi"
# output_video_path = r"C:\Users\Experimenter\Deeplabcut_files\pupil_analysis\videos\FG006_640_480.avi"
# target_resolution = (640, 480)  # Your target resolution


# for i in input_videos: 
#     resize_video(input_video_path, output_video_path, target_resolution)

# r"Z:\RawData\Styx\2024-01-11\1\VideoBottom0.avi"
# r"Z:\RawData\Notos\2023-05-23\4\VideoBottom0.avi"
# r"Z:\RawData\Tara\2023-12-19\2\VideoBottom0.avi"
# r"Z:\RawData\Glaucus\2022-11-14\3\VideoBottom1.avi"
# r"Z:\RawData\FG004\2023-06-09\5\VideoBottom0.avi"


def resize_video(input_path, output_path, target_resolution, resampling_method='lanczos'):
    video_clip = VideoFileClip(input_path)
    resized_clip = video_clip.resize(newsize=target_resolution)
    resized_clip.write_videofile(output_path, codec='mpeg4', bitrate='8000k')

# Base directory for output videos
base_output_dir = r"C:\Users\Experimenter\Deeplabcut_files\pupil_analysis\videos"

# List of input videos
input_videos = [
    r"Z:\RawData\Styx\2024-01-11\1\VideoBottom0.avi",
    r"Z:\RawData\Notos\2023-05-23\4\VideoBottom0.avi",
    r"Z:\RawData\Tara\2023-12-19\2\VideoBottom0.avi",
    r"Z:\RawData\Glaucus\2022-11-14\3\VideoBottom1.avi",
    r"Z:\RawData\FG004\2023-06-09\5\VideoBottom0.avi"
]

# Target resolution
target_resolution = (720, 540)

# Process each video
for input_video_path in input_videos:
    # Construct output path based on input path
    relative_path = os.path.relpath(input_video_path, start=r"Z:\RawData")
    output_video_path = os.path.join(base_output_dir, relative_path)
    output_folder = os.path.dirname(output_video_path)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Resize video
    resize_video(input_video_path, output_video_path, target_resolution)


#%% batch video - 

#TODO: naming/folder need to be corrected

def resize_video(input_path, output_folder, target_resolution):
    video_clip = VideoFileClip(input_path)
    # Resize the video clip to the target resolution
    resized_clip = video_clip.resize(target_resolution)
    
    # Construct the output file path
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_folder, f"{name}_{target_resolution[0]}_{target_resolution[1]}{ext}")
    
    # Write the resized video clip to a new file
    resized_clip.write_videofile(output_path, codec='mpeg4')  # You can adjust codec if needed

# List of input video file paths
video_paths = [
    r"Z:\RawData\FG006\2023-11-09\1\Video0.avi",
    r"Z:\RawData\FG002\2022-10-27\1\Video0.avi",
    r"Z:\RawData\Elias\2022-08-03\1\Video0.avi"
    # Add more video paths here
]

# Output folder
output_folder = r"C:\Users\Experimenter\Deeplabcut_files\pupil_analysis\videos"

# List of target resolutions
resolutions = [
    (640, 480),  # Standard Definition
    (320, 240),  # Lower SD
    # Add more resolutions if needed
]

# Process each video at each resolution
for video_path in video_paths:
    for res in resolutions:
        resize_video(video_path, output_folder, res)


#%%

def find_high_resolution_videos(directory, min_width=1440, min_height=1080):
    high_res_videos = []

    # Search for video files in the directory
    for video_file in glob.glob(os.path.join(directory, '**/*.avi'), recursive=True):
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print(f"Could not open video file: {video_file}")
            continue

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if width >= min_width and height >= min_height:
            high_res_videos.append(video_file)

        cap.release()

    return high_res_videos

# Example usage
directory = 'Z:\RawData\Quille'
high_res_videos = find_high_resolution_videos(directory)
print("High resolution videos found:")
for video in high_res_videos:
    print(video)


