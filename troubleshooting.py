# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:09:08 2024

@author: Experimenter
"""

from moviepy.editor import VideoFileClip
import cv2
import os
import glob
import ffmpeg


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


#%% check height and width ratio

# df = pd.read_csv(r"Z:\ProcessedData\Memphis\2023-10-18\pupil\dlc\5\Video0DLC_resnet101_MousePupilAug2shuffle1_440000.csv", header=[1, 2]) 
# del df["bodyparts"]
#   #df.head()
# width = df[("rightpupil", "x")] - df[("leftpupil", "x")]
# height = df[("ventralpupil", "y")] - df[("dorsalpupil", "y")]

# # Calculate the height-to-width ratio
# ratio = height / width

# # To handle division by zero or very small widths, you can replace infinite values or NaNs
# ratio.replace([np.inf, -np.inf], np.nan, inplace=True)

# # Calculate the average of the ratios, ignoring NaN values
# average_ratio = ratio.mean()
# std = ratio.std()

# print("Average height-to-width ratio & std:", average_ratio, std)

#%% 

import cv2  
import os   

def check_video_length(file_path):
    """Check the length of the video at the given file path."""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Failed to open video: {file_path}")
        return False
    
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Assuming a video is considered to have missing metadata if its length is 0
    return length == 0

def find_videos_with_missing_metadata(root_dir):
    """Find all .avi videos with missing metadata within the specified directory tree."""
    videos_with_missing_metadata = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".avi"):
                file_path = os.path.join(root, file)
                if check_video_length(file_path):
                    videos_with_missing_metadata.append(file_path)
                    print(f"Missing metadata: {file_path}")
    
    return videos_with_missing_metadata

# Example usage
root_directory = r"Z:\RawData"  # Change this to the path of your top-level directory
missing_metadata_videos = find_videos_with_missing_metadata(root_directory)

print(f"Found {len(missing_metadata_videos)} videos with missing metadata.")

#%% copying videos and changing the name of the file to a pathname

import shutil

# List of original file paths
file_paths = [
    "Z:\\RawData\\Elias\\2022-09-23\\Video1.avi",
    "Z:\\RawData\\Elias\\2022-10-06\\Video0.avi",
    "Z:\\RawData\\Elias\\2022-10-06\\Video1.avi",
    "Z:\\RawData\\Elias\\2022-10-06\\Video2.avi",
    "Z:\\RawData\\FG002\\2022-10-25\\7\\Video0.avi",
    "Z:\\RawData\\FG002\\2022-10-25\\7\\VideoBottom0.avi",
    "Z:\\RawData\\FG002\\2022-10-26\\1\\VideoBottom0.avi",
    "Z:\\RawData\\FG002\\2022-10-28\\4\\Video0.avi",
    "Z:\\RawData\\FG002\\2022-10-31\\1\\Video0.avi",
    "Z:\\RawData\\FG002\\2022-10-31\\1\\VideoBottom0.avi",
    "Z:\\RawData\\FG002\\2022-10-31\\2\\VideoBottom0.avi",
    "Z:\\RawData\\FG002\\2022-10-31\\4\\VideoBottom0.avi",
    "Z:\\RawData\\FG002\\2022-10-31\\6\\Video0.avi",
    "Z:\\RawData\\FG002\\2022-10-31\\6\\VideoBottom0.avi",
    "Z:\\RawData\\FG003\\2023-03-10\\5\\Video1.avi",
    "Z:\\RawData\\FG003\\2023-03-10\\5\\VideoBottom0.avi",
    "Z:\\RawData\\FG003\\2023-03-11\\3\\VideoBottom0.avi",
    "Z:\\RawData\\FG003\\2023-03-14\\2\\Video1.avi",
    "Z:\\RawData\\FG003\\2023-03-14\\2\\VideoBottom0.avi",
    "Z:\\RawData\\FG003\\2023-03-14\\3\\Video1.avi",
    "Z:\\RawData\\FG003\\2023-03-14\\3\\VideoBottom0.avi",
    "Z:\\RawData\\FG004\\2023-06-09\\2\\VideoBottom0.avi",
    "Z:\\RawData\\FG004\\2023-06-10\\8\\Video0.avi",
    "Z:\\RawData\\FG004\\2023-06-10\\8\\VideoBottom0.avi",
    "Z:\\RawData\\FG004\\2023-06-12\\6\\VideoBottom0.avi",
    "Z:\\RawData\\FG005\\2023-08-16\\3\\VideoBottom0.avi",
    "Z:\\RawData\\FG005\\2023-08-17\\4\\VideoBottom0.avi",
    "Z:\\RawData\\FG005\\2023-08-21\\7\\VideoBottom0.avi",
    "Z:\\RawData\\FG006\\2023-11-09\\4\\Video1.avi",
    "Z:\\RawData\\FG006\\2023-11-09\\4\\VideoBottom0.avi",
    "Z:\\RawData\\FG006\\2023-11-13\\5\\Video1.avi",
    "Z:\\RawData\\FG006\\2023-11-13\\5\\VideoBottom0.avi",
    "Z:\\RawData\\FG006\\2023-11-14\\10\\Video0.avi",
    "Z:\\RawData\\FG006\\2023-11-14\\10\\VideoBottom0.avi",
    "Z:\\RawData\\Giuseppina\\2022-11-03\\4\\Video8.avi",
    "Z:\\RawData\\Glaucus\\2022-07-20\\1\\Video0.avi",
    "Z:\\RawData\\Glaucus\\2022-10-10\\3\\Video4.avi",
    "Z:\\RawData\\Glaucus\\2022-10-31\\1\\Video1.avi",
    "Z:\\RawData\\Hedes\\2022-08-02\\Video2.avi",
    "Z:\\RawData\\Hedes\\2022-08-04\\3\\Video02.avi",
    "Z:\\RawData\\Hedes\\2022-09-02\\3\\Video2.avi",
    "Z:\\RawData\\SS109\\2022-06-17\\Video0.avi",
    "Z:\\RawData\\SS112\\Reward\\2022-05-10 Test\\1\\Video0.avi",
    "Z:\\RawData\\SS112\\Reward\\2022-05-10 Test\\2\\Video0.avi",
    "Z:\\RawData\\SS113\\2022-07-11\\3\\Video0.avi",
    "Z:\\RawData\\SS113\\2022-10-06\\Video0.avi",
    "Z:\\RawData\\SS113\\2022-12-07\\1\\Video1.avi",
    "Z:\\RawData\\SS113\\2022-12-07\\5\\Video7.avi"
]

dest_dir = "Q:\\video_related\\original_videos_missing_metadata"

# Function to convert file paths into new names
def convert_path_to_name(path):
    # Remove the base path and replace backslashes with underscores
    new_name = path.replace("Z:\\RawData\\", "").replace("\\", "_")
    return new_name

for path in file_paths:
    new_name = convert_path_to_name(path)
    dest_path = os.path.join(dest_dir, new_name)
    
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Copy the file
    shutil.copy2(path, dest_path)
    print(f"Copied {path} to {dest_path}")

#%% replace the videos in RawData with the new ones

import os
import shutil

def move_and_convert_videos(source_dir = r"Q:\video_related\original_videos_missing_metadata", base_destination=r"Z:/RawData"):
    for filename in os.listdir(source_dir):
        if filename.endswith(".mp4"):  # Checks if it's a video file
            # Parse the filename to construct the destination path and new filename
            destination_path, new_name = construct_destination_path_and_name(filename, base_destination)
            
            if destination_path:  # Proceed if the destination path was successfully constructed
                source_path = os.path.join(source_dir, filename)
                destination_file_path = destination_path
                
                # Check if the destination directory exists
                if os.path.exists(destination_path):
                    # Move and replace if the file already exists
                    if os.path.exists(destination_file_path):
                        os.remove(destination_file_path)
                    shutil.move(source_path, destination_file_path)
                    print(f"Moved and converted: {filename} to {destination_file_path}")
                else:
                    print(f"Destination directory does not exist, file not copied: {filename}")
            else:
                print(f"Could not construct a destination path for: {filename}")

def construct_destination_path_and_name(filename, base_destination):
    """
    Construct the destination path based on the filename.
    Assume filename format: 'Elias_2022-09-23_1_Video1.mp4'
    """
    try:
        parts = filename.split('_')
        name_part = parts[-1].rsplit('.', 1)[0]  # Get the name part and remove extension
        date_part = parts[1]
        number_part = parts[2]
        new_name = f"{name_part}.avi"  # Change the file extension to .avi
        
        # Construct the new path
        new_path = os.path.join(base_destination, parts[0], date_part, number_part, new_name)
        return new_path, new_name
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None, None

move_and_convert_videos()

