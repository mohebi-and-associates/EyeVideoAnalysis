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

        
        "DLCeyeModelDir": r"C:\GitHub\EyeVideoAnalysis\MousePupil-SchroederLab-2023-08-02", # "Q:\MousePupil-SchroederLab-2023-08-02",
        'DLCdataDefFile': r"C:\GitHub\EyeVideoAnalysis\pupil_analysis_config.csv", #r"C:\Users\Experimenter\Deeplabcut_files\pupil_analysis\pupil_analysis_config.csv",
    }
    return directoryDb
