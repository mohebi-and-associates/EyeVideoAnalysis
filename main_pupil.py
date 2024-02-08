# -*- coding: utf-8 -*-


#%%


import os

# working_directory_path = "C:\dev\workspaces"
# os.chdir(working_directory_path)
# print(os.getcwd())

import pandas as pd
import deeplabcut


# set console working directory so it doesn't complain about suite2p that is imported in user_defs

# from Data import user_defs
import user_defs
import analyse_videos
import measure_pupil
import crop_videos
import glob


#%% load directories and database

dirs = user_defs.define_directories()

eye_model_path = dirs["DLCeyeModelDir"]
hh_model_path = dirs["DLChhModelDir"] #Headplate Holder DLC model
csvDir = dirs["DLCdataDefFile"]
rawDataBaseFolder = dirs["rawDataBaseFolder"]
destBaseFolder = dirs["destBaseFolder"]
videoDestBaseFolder = dirs["videoDestBaseFolder"]

database = pd.read_csv(
    csvDir,
    dtype={
        "name": str,
        "date": str,
        "exp_number": str,
        "analyze_video": bool,
        "measure_pupil": bool,
        "crop_videos": bool
    },
)

#%% analyse videos with dlc

for i in range(len(database)):
    
    """
    For each entry in the database, this loop checks if the video needs to be analyzed and cropped.
    If so, it iterates over each video path, allows the user to select an ROI, and saves the ROI 
    coordinates to a file. The ROI is selected via a GUI interface for each video.

    Parameters 
    -----------------------------
    database : DataFrame
        A DataFrame containing information about each video to be processed, including flags for analysis and cropping.

    Notes
    -----
    - This loop is only concerned with the collection of ROIs and does not perform any video analysis or cropping.
    - The ROI coordinates are saved in a JSON file within a dlc folder.
    """

    analyze_video = database.loc[i]["analyze_video"]
    measure_pupil_bool = database.loc[i]["measure_pupil"]
    crop_videos_bool = database.loc[i]["crop_videos"]
    if analyze_video:
        name, date, exp_numbers, eyeVideoPaths, plot_videos, create_videos = analyse_videos.read_dataentry_produce_video_dirs(database.loc[i], rawDataBaseFolder)
        if crop_videos_bool: # Select ROI and save to file for each video
            for eye_video_path in eyeVideoPaths:
                roi_coordinates = crop_videos.select_roi(eye_video_path)
                if roi_coordinates is not None:
                    crop_videos.save_roi_to_file(roi_coordinates, eye_video_path, videoDestBaseFolder, rawDataBaseFolder)

for i in range(len(database)):
        
    """

    This loop iterates over each entry in the database and performs a series of operations including DeepLabCut analysis,
    pupil diameter measurement, and video cropping, depending on the flags set in the database for each video. 

    Parameters
    -----------------------------
    database : DataFrame
        A DataFrame containing information about each video to be processed, including flags for analysis, pupil measurement, and cropping.

    Notes
    -----
    - The loop uses external functions from various modules such as `analyse_videos`, `deeplabcut`, `measure_pupil`, and `crop_videos` for specific processing tasks.
    """

    analyze_video = database.loc[i]["analyze_video"]
    measure_pupil_bool = database.loc[i]["measure_pupil"]
    crop_videos_bool = database.loc[i]["crop_videos"]
    
    name, date, exp_numbers, eyeVideoPaths, plot_videos, create_videos = analyse_videos.read_dataentry_produce_video_dirs(database.loc[i], rawDataBaseFolder)
    
    if analyze_video:    
        for eye_video_path in eyeVideoPaths:
            create_video_flag = eye_video_path in create_videos

            dlc_folder, dlc_video_folder, shuffle, config = analyse_videos.create_dlc_ops(eye_video_path, destBaseFolder, videoDestBaseFolder, rawDataBaseFolder, eye_model_path, create_video=create_video_flag)
            dest_folder = dlc_video_folder if create_videos else dlc_folder 

            deeplabcut.analyze_videos(config, [eye_video_path], shuffle=shuffle, save_as_csv=True, destfolder=dest_folder)
            if eye_video_path in create_videos:
                deeplabcut.create_labeled_video(config, [eye_video_path], shuffle=shuffle, destfolder=dest_folder)
            if eye_video_path in plot_videos:
                deeplabcut.plot_trajectories(config, [eye_video_path], shuffle=shuffle, destfolder=dest_folder)
            if dest_folder == dlc_video_folder:
                analyse_videos.copy_non_video_files(dlc_video_folder,dlc_folder)

    pupilAnalysisFiles = measure_pupil.read_dataentry_produce_directories(database.loc[i], destBaseFolder)
        
    if measure_pupil_bool:
        
        for pupilAnalysisFile in pupilAnalysisFiles:
            df, width, height, center_x, valid = measure_pupil.process_raw_data(pupilAnalysisFile, MIN_CERTAINTY=0.6, plot=False)
            F = measure_pupil.estimate_height_from_width_pos(width, height, center_x, valid, plot=False)
            center_y_adj, height_adj = measure_pupil.adjust_center_height(df, F, width, height, center_x, plot=False)
            blinks, bl_starts, bl_stops = measure_pupil.detect_blinks(df, width, center_x, center_y_adj, height_adj, print_out=True, plot=False)
            center_x, center_y_adj, height_adj = measure_pupil.apply_medfilt(center_x, center_y_adj, height_adj, SMOOTH_SPAN = 5)
            data, center, diameter  = measure_pupil.adjust_for_blinks(center_x, center_y_adj, height_adj, width, blinks,plot=False)
            measure_pupil.plot_and_save_data(pupilAnalysisFile, data, blinks, bl_starts, bl_stops, destBaseFolder)
        for eye_video_path in eyeVideoPaths:
            measure_pupil.analyse_HH_video(eye_video_path, destBaseFolder, hh_model_path, num_frames=10)
    if crop_videos_bool:
        # name, date, exp_numbers, eyeVideoPaths, plot_videos, create_videos = analyse_videos.read_dataentry_produce_video_dirs(database.loc[i], rawDataBaseFolder) # can be deleted if absence does not cause any problems 
        for eye_video_path, pupilAnalysisFile in zip(eyeVideoPaths, pupilAnalysisFiles):
            dlc_folder, dlc_video_folder, shuffle, config = analyse_videos.create_dlc_ops(eye_video_path, destBaseFolder, videoDestBaseFolder, rawDataBaseFolder, eye_model_path, create_video=create_video_flag)
            roi_coordinates = crop_videos.read_roi_coordinates_from_dlc_folder(eye_video_path, videoDestBaseFolder, rawDataBaseFolder)
            crop_videos.process_video(eye_video_path, roi_coordinates, dlc_video_folder)       
          

#%% check pupil npy files

# loaded_xyPos = np.load(r"C:\Users\Experimenter\Deeplabcut_files\all_pupil_videos_processed\Name1\2022-01-01\4\measure_pupil\eye.xyPos.npy")
# loaded_diameter = np.load(r"C:\Users\Experimenter\Deeplabcut_files\all_pupil_videos_processed\Name1\2022-01-01\4\measure_pupil\eye.diameter.npy")

#%% check if tensorflow recognises GPU

# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))   

# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# for index, device in enumerate(physical_devices):
#     print(f"Index: {index}, GPU: {device}")


