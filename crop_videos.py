#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:49:33 2023

@author: rg483
"""

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import os

#%%

def read_dataentry_produce_directories(dataEntry, destBaseFolder):
    

    """
    Parses a single data entry from the configuration CSV and invokes the produce_directories function.
    
    Parameters
    ----------
    dataEntry : pd.Series
        A single row from the CSV configuration file containing settings for video analysis.
    destBaseFolder : str
        Base directory where DLC output is saved to be used as input in pupil measurements in this script.
        
    Returns
    -------
        A list of file paths on which pupil measurements will be performed.
    """
    
    exp_numbers = str(dataEntry["exp_number"]).split(",")
    return produce_directories(
        dataEntry["name"], 
        dataEntry["date"], 
        exp_numbers, 
        dataEntry["analyze_video"],
        dataEntry["plot_trajectories"],
        dataEntry["create_video"],
        dataEntry["measure_pupil"],
        destBaseFolder
    )

def produce_directories(name, date, exp_numbers, analyze_video, plot_trajectories, create_video, measure_pupil, destBaseFolder):
    """
    Determines the directory paths based on the configuration provided in the CSV row (data entry).
    
    Parameters
    ----------
    name : str
        Name of the animal.
    date : str
        Date of the experiment.
    exp_numbers : list
        List of experiment numbers for which analysis is required.
    analyze_video : bool
        Flag indicating whether to analyze video.
    plot_trajectories : bool
        Flag indicating whether to plot trajectories.
    create_video : bool
        Flag indicating whether to create labeled videos.
    measure_pupil : bool
        Flag indicating whether to perform pupil measurements.
    destBaseFolder : str
        Base directory where DLC output is saved to be used as input in pupil measurements.
    
    Returns
    -------
    list
        A list of file paths for pupil measurements.
        
    Raises
    ------
    ValueError
        If no DeepLabCut output files are found.
    """

    pupilAnalysisFiles = []
    
    for exp_number in exp_numbers:
        if exp_number.strip().lower() == "all":
            dlc_folder_path = os.path.join(destBaseFolder, name, date, "pupil", "dlc")
            dlc_csv_files = glob.glob(os.path.join(dlc_folder_path, "*", "*.csv"))
            print("Checking in folder:", dlc_folder_path)
            print("Files found:", dlc_csv_files)
        else:
            dlc_folder_path = os.path.join(destBaseFolder, name, date, "pupil", "dlc", exp_number.strip())
            dlc_csv_files = glob.glob(os.path.join(dlc_folder_path, "*.csv"))
            print("Checking in folder:", dlc_folder_path)
            print("Files found:", dlc_csv_files)
        for file in dlc_csv_files:
            pupilAnalysisFiles.append(file)
    if not pupilAnalysisFiles:
        raise ValueError("No Deeplabcut output files") 
        
    return pupilAnalysisFiles

def select_roi(video_pathname, filepath='roi_coordinates.json'):
    
    """

    This function opens a video file and lets the user define an ROI by clicking to select two diagonal 
    corners of a rectangle (top left corner and bottom right corner). 
    The selected ROI coordinates are then returned. The user can reselect the ROI 
    by right-clicking or exit the selection process by pressing the 'Esc' key.

    Parameters
    ----------
    video_pathname : str
        Path to the video file from which the ROI is to be selected.
    filepath : str
        name of the JSON file which will store the ROI coordinates.

    Returns
    -------
    tuple or None
        A tuple of four integers representing the coordinates of the selected ROI in the format 
        (x1, y1, width, height), where (x1, y1) is the top-left corner of the rectangle. Returns 
        None if no ROI is selected.

    """
    
       
    cap = cv2.VideoCapture(video_pathname)
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    
    p1, p2 = None, None
    state = 0
    
    def on_mouse(event, x, y, flags, userdata):
        nonlocal state, p1, p2
    
        if event == cv2.EVENT_LBUTTONUP:
            if state == 0:
                p1 = (x, y)
                state += 1
            elif state == 1:
                p2 = (x, y)
                state += 1
        if event == cv2.EVENT_RBUTTONUP:
            p1, p2 = None, None
            state = 0
    
    cv2.setMouseCallback('Video', on_mouse)
    
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            if frame is not None:
                if state > 1:
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 10)

                cv2.imshow('Video', frame)
        
                key = cv2.waitKey(50)
                if key == 27:
                    break
            else:
                print("Failed to grab frame")
        else:
            print("Could not read video")
            break

    cv2.destroyAllWindows()
    cap.release()
    
    if p1 is not None and p2 is not None:
        roi_coordinates = p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1]
        return roi_coordinates 
    else:
        print("ROI coordinates were not selected.")
        return None
    

def save_roi_to_file(roi_coordinates, eye_video_path, videoDestBaseFolder, rawDataBaseFolder, filepath='roi_coordinates.json'):

    """

    This function takes ROI coordinates and saves them in a JSON file within a directory structure 
    based on the provided video path. The directory is created if it does not exist. 

    Parameters
    ----------
    roi_coordinates : tuple
        A tuple containing the coordinates of the ROI (x, y, width, height).
    eye_video_path : str
        The path to the video file associated with the ROI.
    destBaseFolder : str
        The base directory where the processed data and results will be stored.
    rawDataBaseFolder : str
        The base directory where the raw data videos are located.
    filepath : str, optional
        The name of the JSON file to save the ROI coordinates. Default is 'roi_coordinates.json'.

    Returns
    -------
    tuple
        The ROI coordinates that were saved to the file.

    """
    
    rel_path = os.path.relpath(eye_video_path, rawDataBaseFolder)

    # Extract the folder number (it's the parent directory of the video file)
    folder_number = os.path.basename(os.path.dirname(rel_path))

    # Construct the destination folder without the folder number
    dest_folder_elements = os.path.dirname(rel_path).split(os.sep)[:-1]
    dest_folder = os.path.join(videoDestBaseFolder, *dest_folder_elements)

    dlc_video_folder = os.path.join(dest_folder, "pupil", "dlc", folder_number)
    if not os.path.exists(dlc_video_folder):
        os.makedirs(dlc_video_folder)
    filepath = os.path.join(dlc_video_folder, filepath)
    with open(filepath, 'w') as f:
        json.dump(roi_coordinates, f) 
    return roi_coordinates




def read_roi_coordinates_from_dlc_folder(eye_video_path, videoDestBaseFolder, rawDataBaseFolder):
    
    
    """
    Reads the region-of-interest (ROI) coordinates from a JSON file saved in the DLC folder.
    
    Parameters:
    - eye_video_path (str): The path to the eye video being processed.
    - destBaseFolder (str): The destination base folder.
    - rawDataBaseFolder (str): The raw data base folder.
    
    Returns:
    - dict: ROI coordinates read from the JSON file.
    """

    
    # Calculate the path to the dlc_folder where the roi_coordinates.json is saved
    rel_path = os.path.relpath(eye_video_path, rawDataBaseFolder)

    # Extract the folder number (it's the parent directory of the video file)
    folder_number = os.path.basename(os.path.dirname(rel_path))

    # Construct the destination folder without the folder number
    dest_folder_elements = os.path.dirname(rel_path).split(os.sep)[:-1]
    dest_folder = os.path.join(videoDestBaseFolder, *dest_folder_elements) 

    dlc_video_folder = os.path.join(dest_folder, "pupil", "dlc", folder_number)
    
    # Construct the path to the roi_coordinates.json file
    roi_filepath = os.path.join(dlc_video_folder, 'roi_coordinates.json')
    
    try:
        # Read roi_coordinates from the JSON file
        with open(roi_filepath, 'r') as f:
            roi_coordinates = json.load(f)
    except FileNotFoundError:
        print("Error: roi_coordinates.json file not found.")
        roi_coordinates = None
    
    except json.JSONDecodeError:
        print("Error: Could not decode JSON file.")
        roi_coordinates = None
    
    return roi_coordinates 



def process_video(video_pathname, roi_coordinates, dlc_video_folder):
        
    """

    This function reads a video, crops it to the specified Region of Interest (ROI), and overlays
    DeepLabCut (DLC) marker positions on each frame. The processed video is saved with markers 
    visually represented in the specified DLC folder. The function also handles marker positions 
    with a likelihood lower than a threshold by marking them with a cross.

    Parameters
    ----------
    video_pathname : str
        Path to the original video file to be processed.
    roi_coordinates : tuple
        Coordinates for the Region of Interest in the format (x1, y1, width, height).
    dlc_folder : str
        Path to the folder where the DLC analysis files are stored and the processed video will be saved.

    Notes
    -----
    - The processed video is saved in AVI format in the same DLC folder with '_cropped' appended to the original filename.
    """
    
       
    dlc_csv_path = glob.glob(os.path.join(dlc_video_folder, "*.csv"))[0]
    video_basename_without_extension = os.path.splitext(os.path.basename(video_pathname))[0]    
    output_video_name = f"{video_basename_without_extension}_cropped.avi"
    output_path = os.path.join(dlc_video_folder, output_video_name)

    cap = cv2.VideoCapture(video_pathname)
    if not cap.isOpened():
        print("Error: Couldn't open video capture.")
        return
    
    w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    x1, y1, h, w = roi_coordinates
    

    desired_width = h * 4
    desired_height = w * 4
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (desired_width, desired_height))

    if not out.isOpened():
        print("Error: Couldn't open video writer.")
        return

    df = pd.read_csv(dlc_csv_path, header=[1, 2])
    del df["bodyparts"]
    
    marker_coordinates = []
    for row in df.values:
        frame_coordinates = []
        for i in range(0, len(row), 3):
            x, y, likelihood = map(float, row[i:i+3])
            frame_coordinates.append((x, y, likelihood))
        marker_coordinates.append(frame_coordinates)
    
    cmap = plt.get_cmap('rainbow')
    
    while cap.isOpened():
        _, frame = cap.read()
    
        if frame is None:
            break
    
        crop_frame = frame[y1:y1+w, x1:x1+h]
        resized_frame = cv2.resize(crop_frame, (desired_width, desired_height), interpolation=cv2.INTER_LANCZOS4)
    
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        current_frame_coordinates = marker_coordinates[frame_index]
    
        for i, (x, y, likelihood) in enumerate(current_frame_coordinates):
            new_x = int((x - x1) * (desired_width / h))
            new_y = int((y - y1) * (desired_height / w))
    
            if likelihood > 0.6:
                color = cmap(i / len(current_frame_coordinates))
                color = np.array(color) * 255
                cv2.circle(resized_frame, (int(new_x), int(new_y)), 5, color.astype(np.uint8).tolist(), -1)
            else:
                color_black = (0, 0, 0)
                thickness = 3
                length = 7
                cv2.line(resized_frame, (int(new_x) - length, int(new_y)), (int(new_x) + length, int(new_y)), color_black, thickness)
                cv2.line(resized_frame, (int(new_x), int(new_y) - length), (int(new_x), int(new_y) + length), color_black, thickness)
    
        out.write(resized_frame)
    
    cap.release()
    out.release()

