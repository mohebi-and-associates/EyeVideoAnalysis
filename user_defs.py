# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:23:14 2023

@author: Experimenter
"""

def define_directories():
    """
    Creates a dictionary containing the directory paths needed for pupil analysis.

    Returns
    -------
    directoryDb : dict
        A dictionary containing the following keys and their respective directory paths:
        - 'model_path': The directory where the trained model for pupil analysis is located.
        - 'csvDir': Directory path to the CSV file containing video analysis configurations.
        - 'rawDataBaseFolder': The base directory where the raw experimental data is stored.
        - 'destBaseFolder': The base directory where all processed pupil videos will be saved.
    }
    """
    directoryDb = {

        
        
        # 'model_path': r"C:\Users\Experimenter\Deeplabcut_files\MousePupil-SchroederLab-2023-08-02",
        # 'csvDir': r"C:\Users\Experimenter\Deeplabcut_files\pupil_analysis\pupil_analysis_config.csv",
        # 'rawDataBaseFolder': r"Z:\RawData",
        # 'destBaseFolder': r"Z:\ProcessedData",
        # 'videoDestBaseFolder': r"C:\Users\Experimenter\Deeplabcut_files\pupil_analysis\videos"
        
        "DLCeyeModelDir": r"C:\\Users\\Experimenter\\Deeplabcut_files\\MousePupil-SchroederLab-2023-08-02",
        "DLChhModelDir": r"C:\Users\Experimenter\Desktop\HeadplateHolder-SchroederLab-2024-01-29",
        'DLCdataDefFile': r"C:\dev\workspaces\EyeVideoAnalysis\pupil_analysis_config.csv", #r"C:\Users\Experimenter\Deeplabcut_files\pupil_analysis\pupil_analysis_config.csv",
        'rawDataBaseFolder': r"Z:\RawData",
        # 'rawDataBaseFolder': r"C:\Users\Experimenter\Deeplabcut_files\pupil_analysis\videos",
        # 'destBaseFolder': r"C:\Users\Experimenter\Deeplabcut_files\pupil_analysis\videos_analysed",
        'destBaseFolder': r"Z:\ProcessedData",
        'videoDestBaseFolder': r"C:\Users\Experimenter\Deeplabcut_files\pupil_analysis\videos_analysed" #can also be set to None
        
    }
    return directoryDb
