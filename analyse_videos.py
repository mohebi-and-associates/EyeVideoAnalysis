#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import os
import deeplabcut
import pandas as pd
import glob
import re


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
        else:
            folder_path = os.path.join(rawDataBaseFolder, name, date, exp_number.strip())
            eye_avi_files = glob.glob(os.path.join(folder_path, "*.avi"))
    
        for file in eye_avi_files:
            match = re.match(r"Video(\d)\.avi", os.path.basename(file))
            if match:
                digit = int(match.group(1))
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


def create_dlc_ops(eye_video_path, destBaseFolder, rawDataBaseFolder, model_path):

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
    rawDataBaseFolder : str
        The base folder where the raw data videos are stored.
    model_path : str
        The file path to the DeepLabCut model.

    Returns
    -------
    dlc_folder : str
        The path to the newly created sub-folder within the destination folder, dedicated to DLC output files.
    shuffle : int
        The shuffle parameter used in DLC operations.
    config : str
        The path to the DeepLabCut model configuration file.
    """
    
    shuffle = 1
    config = os.path.join(model_path,'config.yaml') 

    # Create the new destination folder structure
    rel_path = os.path.relpath(eye_video_path, rawDataBaseFolder)

    # Extract the folder number (it's the parent directory of the video file)
    folder_number = os.path.basename(os.path.dirname(rel_path))

    # Construct the destination folder without the folder number
    dest_folder_elements = os.path.dirname(rel_path).split(os.sep)[:-1]
    dest_folder = os.path.join(destBaseFolder, *dest_folder_elements)

    dlc_folder = os.path.join(dest_folder, "pupil", "dlc", folder_number)

    # Create the folder if it doesn't exist
    if not os.path.exists(dlc_folder):
        os.makedirs(dlc_folder)

    return dlc_folder, shuffle, config 
