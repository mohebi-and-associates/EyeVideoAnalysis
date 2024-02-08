#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import os
import deeplabcut
import pandas as pd
import glob
import re
import shutil


#%%  
def read_dataentry_produce_video_dirs(dataEntry, rawDataBaseFolder):

    """
    Parses a single data entry from the configuration CSV and invokes the produce_directories function.
    
    Parameters
    ----------
    dataEntry : pd.Series
        A single row from the CSV configuration file containing settings for video analysis.
    rawDataBaseFolder : str ["YourDirectoryPath"]
        The base directory where the raw data videos are stored.
    
    Returns
    -------
    tuple
        Returns a tuple containing name, date, experiment numbers, paths to eye videos, 
        paths to plot trajectories videos, and paths to create videos.
    """ 
    exp_numbers = str(dataEntry["exp_number"]).split(",") #split exp_numbers if there are several exp_numbers in csv (e.g., "1,2,3")
    return produce_directories(
        dataEntry["name"], 
        dataEntry["date"], 
        exp_numbers, 
        dataEntry["analyze_video"],
        dataEntry["plot_trajectories"],
        dataEntry["create_video"], 
        rawDataBaseFolder
    )

def produce_directories(name, date, exp_numbers, analyze_video, plot_trajectories, create_video, rawDataBaseFolder):
    
    """
    Determines the directory paths based on the configuration provided in the CSV row (data entry).
    
    Parameters
    ----------
    name : str
        Name of the animal.
    date : str
        Date of the experiment.
    exp_numbers : list of str when  
        List of experiment numbers for which analysis is required. 
        Each string can be either "all", a single number, or multiple comma-separated numbers.
    analyze_video : bool
        Flag indicating whether to analyze video.
    plot_trajectories : bool
        Flag indicating whether to plot trajectories.
    create_video : bool
        Flag indicating whether to create labeled videos.
    rawDataBaseFolder : str
        Base directory where raw data is stored.
    
    Returns
    -------
    tuple
        Returns a tuple containing name, date, experiment numbers, paths to videos, 
        paths to videos for plotting trajectories, and paths to videos for creating labeled videos.
    """
    eyeVideoPaths = []
    
    for exp_number in exp_numbers:
        if exp_number.strip().lower() == "all":
            folder_path = os.path.join(rawDataBaseFolder, name, date)
            eye_avi_files = glob.glob(os.path.join(folder_path, "*", "*.avi"))
            print(f"Looking in folder: {folder_path}")
            print(f"Found files: {eye_avi_files}")
        else:
            folder_path = os.path.join(rawDataBaseFolder, name, date, exp_number.strip())
            eye_avi_files = glob.glob(os.path.join(folder_path, "*.avi"))
            print(f"Looking in folder: {folder_path}")
            print(f"Found files: {eye_avi_files}")

        for file in eye_avi_files:
            match = re.match(r"Video(\d)\.avi", os.path.basename(file)) #if the videofile name is Video# where # is a number
            # match = re.match(r"Video(\d+).*\.avi", os.path.basename(file)) # if the name of the file is not simply Video#

            if match:
                digit = int(match.group(1)) #if the videofile name is Video# where # is a number
                if digit >= 0 and digit <= 9: 
                    eyeVideoPaths.append(file)
                
    print("eyeVideoPaths:", eyeVideoPaths)
    if not eyeVideoPaths:
        raise ValueError("No eye video files found in the directories.")

    plot_videos = []
    if analyze_video and plot_trajectories:  # Both conditions must be met
        plot_videos = eyeVideoPaths.copy()
        
    create_videos = []
    if analyze_video and create_video:
        create_videos = eyeVideoPaths.copy()    

    return name, date, exp_numbers, eyeVideoPaths, plot_videos, create_videos

#TODO: modify docstrings


def create_dlc_ops(eye_video_path, destBaseFolder, videoDestBaseFolder, rawDataBaseFolder, model_path, create_video=False):
    shuffle = 1
    config = os.path.join(model_path, 'config.yaml') 
    """

    This function constructs a 'dlc' sub-directory within a destination folder based on the relative path of 
    the eye video from the raw data base folder.  
    It also provides the shuffle parameter and the path to the DLC configuration file.

    Parameters
    ----------
    eye_video_path : str
        The path to the video file that will be analyzed using the DLC model.
    destBaseFolder : str
        The base folder where the processed data and results will be stored.
    videoDestBaseFolder: str
        The base folder where the labelled and cropped DLC videos will be stored 
    rawDataBaseFolder : str
        The base folder where the raw data videos are stored.
    model_path : str
        The file path to the DeepLabCut model.

    Returns
    -------
    dlc_folder : str
        The path to the newly created sub-folder within the destination folder, dedicated to DLC output files.
    dlc_video_folder: str 
        The path to the folder within the destination folder for DLC videos.
    shuffle : int
        The shuffle parameter used in DLC operations.
    config : str
        The path to the DeepLabCut model configuration file.
    """

    rel_path = os.path.relpath(eye_video_path, rawDataBaseFolder)
    folder_number = os.path.basename(os.path.dirname(rel_path))
    dest_folder_elements = os.path.dirname(rel_path).split(os.sep)[:-1]
    dest_folder = os.path.join(destBaseFolder, *dest_folder_elements)
    dlc_folder = os.path.join(dest_folder, "pupil", "dlc", folder_number)
    
    # Initialize dlc_video_folder
    dlc_video_folder = None  # or dlc_folder if you want it to default to dlc_folder

    if create_video and videoDestBaseFolder:
        dest_video_folder = os.path.join(videoDestBaseFolder, *dest_folder_elements)
        dlc_video_folder = os.path.join(dest_video_folder, "pupil", "dlc", folder_number)

        if not os.path.exists(dlc_video_folder):
            os.makedirs(dlc_video_folder)

    # Create the dlc_folder if it doesn't exist
    if not os.path.exists(dlc_folder):
        os.makedirs(dlc_folder)

    return dlc_folder, dlc_video_folder, shuffle, config

def copy_non_video_files(src_folder, dst_folder):
    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dst_path = os.path.join(dst_folder, item)

        # Skip the roi_coordinates.json file
        if item == 'roi_coordinates.json':
            continue

        if os.path.isdir(src_path):
            if not os.path.exists(dst_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)  # For directories
        elif not src_path.endswith('.mp4'):  # Copy if not an MP4 file
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
