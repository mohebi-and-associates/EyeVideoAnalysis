#%% import modules

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import SmoothBivariateSpline
from scipy.signal import medfilt

#%% Load data 

MIN_CERTAINTY = 0.6

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

#%% Scatterplot with raw and valid raw data

def process_raw_data(file_path, MIN_CERTAINTY, plot=False):

   """
    Processes raw data from a DLC CSV output file to calculate pupil width, height, and center coordinates.
   
    Parameters
    ----------
    file_path : str
        Path to the CSV file containing raw data with x, y coordinates for each marker.
    MIN_CERTAINTY : float
        Minimum certainty threshold for valid data points at the beginning.
    plot : bool, optional
        Flag to indicate if plots should be generated.
   
    Returns
    -------
    tuple
        A tuple containing the following:
        - df : pd.DataFrame, [n x 27]
          x and y coordinates and likelihoods for markers in each video frame excluding unnecessary headers.
        - width : pd.Series, [n,]
          Pupil width calculated as the difference in x-coordinates of right and left pupils.
        - height : pd.Series, [n,]
          Pupil height calculated as the difference in y-coordinates of ventral and dorsal pupils.
        - center_x : pd.Series, [n,]
          X-coordinate of the pupil center calculated as half of the width.
        - valid : pd.Series, [n,]
          Boolean series indicating valid data points based on the MIN_CERTAINTY threshold.
    """
   
   df = pd.read_csv(file_path, header=[1, 2]) 
   del df["bodyparts"]
    #df.head()
   width = df[("rightpupil", "x")] - df[("leftpupil", "x")]
   height = df[("ventralpupil", "y")] - df[("dorsalpupil", "y")]
   center_x = df[("leftpupil", "x")] + 0.5 * width
   valid = (df.loc[:, (slice(None), "likelihood")] > MIN_CERTAINTY).all(axis=1)
    
   if plot:
        # Scatterplot from raw width, height, center_x with height color-coded 
        plt.figure()
        plt.scatter(center_x, width, c=height, cmap='jet')
        plt.colorbar(label='Height')
        plt.xlabel('CenterX')
        plt.ylabel('Width')
        plt.title('Raw data')
        plt.show()
    
        # Scatterplot from valid raw width, height, center_x
        plt.figure()
        plt.scatter(center_x[valid], width[valid], c=height[valid], cmap='jet')
        plt.colorbar(label='Height')
        plt.xlabel('CenterX')
        plt.ylabel('Width')
        plt.title('Valid raw data')
        plt.show()
        
   return df, width, height, center_x, valid

# df = process_raw_data(r"Z:\ProcessedData\FG003\2023-03-10\4\dlc\Video1DLC_resnet101_MousePupilAug2shuffle1_440000.csv", MIN_CERTAINTY, plot=False)

#%% Define estimate_height_from_width_pos; plot interpolation at grid points; plot smoothed interpolation at grid points

def estimate_height_from_width_pos(width, height, center_x, valid, plot=False):
    """
    Performs a grid-based linear interpolation of height based on width and center_x coordinates, 
    and applies a Smooth Bivariate Spline to the interpolated data.
   
    Parameters
    ----------
    width : np.array, [n,]
        width values.
    height : np.array, [n,]
        height values.
    center_x : np.array, [n,]
        center x-coordinates.
    valid : np.array, dtype=bool, [n,]
        valid entries across width, height, and center_x.
    plot : bool, optional
        If True, plots the interpolated and smoothed data.
   
    Returns
    -------
    F : SmoothBivariateSpline object
        A fitted SmoothBivariateSpline object that can be used for making height predictions.
    
    Notes
    -----
    This function uses SciPy's griddata for interpolation and SmoothBivariateSpline for smoothing.
    """
    points = np.column_stack((center_x[valid], width[valid]))
    values = height[valid]

    x_grid, y_grid = np.meshgrid(np.linspace(min(center_x[valid]), max(center_x[valid]), 100), np.linspace(min(width[valid]), max(width[valid]), 100))
    z_grid = griddata(points, values, (x_grid, y_grid), method='linear')

    ind = np.logical_not(np.isnan(np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()))).any(axis=1))
    x_notnan = x_grid.flatten()[ind] 
    y_notnan = y_grid.flatten()[ind]
    z_notnan = z_grid.flatten()[ind]
    
    F = SmoothBivariateSpline(x_notnan, y_notnan, z_notnan, kx=3, ky=3)
    
    if plot:
        # Scatterplot with grid points
        plt.figure()
        plt.scatter(x_grid, y_grid, c=z_grid, cmap='jet')
        plt.colorbar(label='Height')
        plt.xlabel('CenterX')
        plt.ylabel('Width')
        plt.title('Interpolation')
        plt.show()

        # Scatterplot after smoothing
        plt.figure()
        z_smoothed = F.ev(x_notnan, y_notnan)
        plt.scatter(x_notnan, y_notnan, c=z_smoothed, cmap='jet')
        plt.colorbar(label='Height')
        plt.xlabel('CenterX')
        plt.ylabel('Width')
        plt.title('Interpolation after smoothing')
        plt.show()

    return F

#%% Define adjust_center_height, plot adjusted height and center_y

# height is adjusted at invalid points using interpolation
# bottom = ventral 
# top = dorsal 

def adjust_center_height(df, F, width, height, center_x, plot=False):
    
    """
    Adjusts the height of the pupil using a SmoothBivariateSpline under specific conditions where the height values
    may be unreliable. These conditions include:
    
    - Only the bottom of the pupil is valid (i.e., the top has low likelihood).
    - Only the top of the pupil is valid (i.e., the bottom has low likelihood).
    - The eyelid is near the top of the pupil (i.e., less than 5 pixels away).
    - The eyelid is near the bottom of the pupil (i.e., less than 5 pixels away).

    Parameters
    ----------
    df : pd.DataFrame, shape (n, 27)
        DataFrame containing x and y coordinates and likelihoods for markers in each video frame.
    F : SmoothBivariateSpline object
        A SmoothBivariateSpline object used for estimating height adjustments.
    width : np.array, shape (n,)
        Array containing width values.
    height : np.array, shape (n,)
        Array containing height values that will be adjusted.
    center_x : np.array, shape (n,)
        Array containing x-coordinates of the pupil's center.
    plot : bool, optional
        Flag to indicate if plots should be generated.

    Returns
    -------
    center_y : np.array, shape (n,)
        Adjusted y-coordinates of the center of the pupil.
    height : np.array, shape (n,)
        Adjusted height values based on the conditions specified.
        
    """

    MAX_DIST_PUPIL_LID = 5  # in pixels

    only_bottom_valid = (df[("dorsalpupil", "likelihood")] < MIN_CERTAINTY) & (df[("ventralpupil", "likelihood")] > MIN_CERTAINTY)
    only_top_valid = (df[("dorsalpupil", "likelihood")] > MIN_CERTAINTY) & (df[("ventralpupil", "likelihood")] < MIN_CERTAINTY)
    lid_near_pupil_top = (df[("dorsalpupil", "y")] - df[("dorsaleyelid", "y")]) < MAX_DIST_PUPIL_LID
    lid_near_pupil_bottom = df[("ventraleyelid", "y")] - df[("ventralpupil", "y")] < MAX_DIST_PUPIL_LID
    width_valid = (df[("leftpupil", "likelihood")] > MIN_CERTAINTY) & (df[("rightpupil", "likelihood")] > MIN_CERTAINTY)
    ind = width_valid & (only_bottom_valid | only_top_valid | lid_near_pupil_top | lid_near_pupil_bottom)
    height[ind] = F.ev(center_x[ind], width[ind]) # original height is adjusted now at height[ind]
    
    center_y = df[("ventralpupil", "y")] - 0.5 * height 
    center_y[width_valid & only_top_valid] = df.loc[width_valid & only_top_valid, ("dorsalpupil", "y")] + 0.5 * height[width_valid & only_top_valid]
    
    if plot: 
        plt.figure()
        plt.scatter(center_y, width, c=height, cmap='jet')
        plt.colorbar(label='height_adj')
        plt.xlabel('center_y_adj')
        plt.ylabel('Raw Width')
        plt.title('Height and center_y adjusted')
        plt.show()
        
        plt.figure()
        plt.scatter(center_x, width, c=height, cmap='jet')
        plt.colorbar(label='height_adj')
        plt.xlabel('center_x')
        plt.ylabel('Raw Width')
        plt.title('Height adjusted')
        plt.show()
    
    return center_y, height

#%% Detect blinks 
# bottom = ventral 
# top = dorsal

def detect_blinks(df, width, center_x, center_y, height, print_out=True, plot=False): 
    
    """
    Detects blinks based on various conditions, including:
    - Both the left and right of the pupil are low likelihood.
    - Small distance between upper and lower eyelids.
    - Both the top and bottom of the pupil are low likelihood.
    - Small distance between center and either the upper and lower eyelids. 
    
    Parameters
    ----------
    df : pd.DataFrame, [n x 27]
        Dataframe containing x, y coordinates and likelihoods for markers in each video frame.
    width : np.array, [n,]
        width values.
    center_x : np.array, [n,]
        x-coordinates of the pupil center.
    center_y : np.array, [n,]
        y-coordinates of the pupil center.
    height : np.array, [n,]
        height values.
    print_out : bool, optional
        If True, prints the blink episodes.
    plot : bool, optional
        If True, plots the data excluding blinks.
    
    Returns
    -------
    blinks : np.array, [n,], dtype=bool
        Boolean array indicating the frames where blinks occur.
    starts : np.array
        Indices where blink episodes start.
    stops : np.array
        Indices where blink episodes stop.
    
    Notes
    -----
    The function uses constants `LID_MIN_STD`, `MIN_DIST_LID_CENTER`, and `SURROUNDING_BLINKS` to adjust the blink detection criteria.
    """
    
    LID_MIN_STD = 7
    MIN_DIST_LID_CENTER = 0.5 # in number of pupil heights   
    SURROUNDING_BLINKS = 5; # in frames
    
    blinks = np.logical_or(df[("leftpupil", "likelihood")] < MIN_CERTAINTY, df[("rightpupil", "likelihood")] < MIN_CERTAINTY)
    
    # distance between upper and lower eye lids is very small -> add to blinks 
        
    lid_distance = df[("ventraleyelid", "y")] - df[("dorsaleyelid", "y")]
    lid_valid = (df[("ventraleyelid", "likelihood")] > MIN_CERTAINTY) & (df[("dorsaleyelid", "likelihood")] > MIN_CERTAINTY)
    lid_mean = np.mean(lid_distance[lid_valid])
    lid_std = np.std(lid_distance[lid_valid])
    blinks[lid_valid] = np.logical_or(blinks[lid_valid], lid_distance[lid_valid] < (lid_mean - LID_MIN_STD * lid_std))
    
    # if top and bottom of pupil uncertain -> add to blinks
    blinks = np.logical_or(blinks, np.logical_and(df[("dorsalpupil", "likelihood")] < MIN_CERTAINTY, df[("ventralpupil", "likelihood")] < MIN_CERTAINTY))
    
    # get minimum distance of center to lid (top or bottom); if distance too small -> add to blinks
    dist_top_lid_to_center = center_y - df[("dorsaleyelid", "y")]
    dist_top_lid_to_center[df[("dorsaleyelid", "likelihood")] < MIN_CERTAINTY] = np.nan
    dist_bottom_lid_to_center = df[("ventraleyelid", "y")] - center_y
    dist_bottom_lid_to_center[df[("ventraleyelid", "likelihood")] < MIN_CERTAINTY] = np.nan
    dist_lid_center = np.min(np.column_stack((dist_top_lid_to_center, dist_bottom_lid_to_center)), axis=1)
    blinks = np.logical_or(blinks, dist_lid_center < (MIN_DIST_LID_CENTER * height))
    
    # include n frames before and after detected blinks
    tmp = blinks.copy()
    for t in range(1, SURROUNDING_BLINKS + 1):
        blinks = np.logical_or(blinks, np.concatenate((np.zeros(t, dtype=bool), tmp[:-t])))
        blinks = np.logical_or(blinks, np.concatenate((tmp[t:], np.zeros(t, dtype=bool))))
    blinks_array = blinks.to_numpy()
    
    if print_out:
        
        d = np.diff(blinks_array)
        starts = np.where(d == 1)[0] + 1
        stops = np.where(d == -1)[0]

        print("Blink episodes:")
        for start, stop in zip(starts, stops):
            print(f"  {start}\t\t{stop}")
            
    if plot: 
        
        width_non_bl = width[blinks == False]
        height_non_bl = height[blinks == False]
        center_y_non_bl = center_y[blinks == False] 
        center_x_non_bl = center_x[blinks == False] 
        
        plt.figure()
        plt.scatter(center_x_non_bl, width_non_bl, c = height_non_bl, cmap='jet')
        plt.colorbar(label='Height')
        plt.ylabel('Width')
        plt.xlabel('CenterX')
        plt.title('After excluding blinks')
        plt.show() 
        
        
        plt.figure()
        plt.scatter(center_y_non_bl, width_non_bl, c = height_non_bl, cmap='jet')
        plt.colorbar(label='Height')
        plt.ylabel('Width')
        plt.xlabel('CenterY')
        plt.title('After excluding blinks')
        plt.show() 

    return blinks, starts, stops
#%% Apply medfilt and find diameter and center
    
def apply_medfilt(center_x, center_y_adj, height_adj, SMOOTH_SPAN):
    
    """

    This function uses a median filter to smooth the given arrays of x-coordinates, adjusted y-coordinates, 
    and adjusted heights. The median filter helps in reducing noise or fluctuations in the data. The degree 
    of smoothing is controlled by the SMOOTH_SPAN parameter, which defines the kernel size for the median filter.

    Parameters
    ----------
    center_x : ndarray, [n,]
        Array of x-coordinates.
    center_y_adj : ndarray, [n,]
        Array of adjusted y-coordinates.
    height_adj : ndarray, [n,]
        Array of adjusted heights.
    SMOOTH_SPAN : int
        The kernel size for the median filter. It specifies the number of adjacent values used in smoothing.

    Returns
    -------
    center_x : ndarray, [n,]
        Smoothed array of x-coordinates.
    center_y_adj : ndarray, [n,]
        Smoothed array of adjusted y-coordinates.
    height_adj : ndarray, [n,]
        Smoothed array of adjusted heights.
    """
    
    center_x = medfilt(center_x, SMOOTH_SPAN)
    center_y_adj = medfilt(center_y_adj, SMOOTH_SPAN)
    height_adj = medfilt(height_adj, SMOOTH_SPAN)

    return center_x, center_y_adj, height_adj

def adjust_for_blinks(center_x, center_y_adj, height_adj, width, blinks, plot=True):
    
    """

    This function processes the given eye tracking data to account for blinks. During blinks, the x and y 
    coordinates and diameter values are set to NaN to indicate missing data. The function creates a data array with adjusted 
    center and diameter values and optionally generates plots to visualize these adjustments.

    Parameters
    ----------
    center_x : ndarray
        Array of x-coordinates of the eye center.
    center_y_adj : ndarray
        Array of adjusted y-coordinates of the eye center.
    height_adj : ndarray
        Array of adjusted heights (diameters) of the eye.
    width : ndarray
        Array of widths of the eye.
    blinks : ndarray
        Boolean array indicating blink occurrences (True if a blink occurred).
    plot : bool, optional
        If True, scatter plots are generated to visualize the data after adjustment.

    Returns
    -------
    data : ndarray
        Array combining center_x, center_y_adj, and height_adj into a single data structure.
    center : list of ndarray
        List containing the adjusted x and y coordinates as two separate arrays.
    diameter : ndarray
        Array of adjusted diameters, with NaNs where blinks are detected.
    """
    data = np.column_stack((center_x, center_y_adj, height_adj))

    center = [center_x, center_y_adj]
    for i in range(len(center)):
        if blinks[i]:
            center[i] = (np.nan, np.nan)
    diameter = np.where(blinks, np.nan, height_adj)
    if plot:
        plt.figure()
        plt.scatter(center_x, width, c = diameter, cmap='jet')
        plt.colorbar(label = 'Diameter')
        plt.xlabel('CenterX')
        plt.ylabel('Width')
        plt.title('After medfilt')
        plt.show() 
        
        plt.figure()
        plt.scatter(center_y_adj, width, c = diameter, cmap='jet')
        plt.colorbar(label = 'Diameter')
        plt.xlabel('CenterY')
        plt.ylabel('Width')
        plt.title('After medfilt')
        plt.show() 
    return data, center, diameter 


#%% Plot trace of diameter and center coordinates with blinks 

def plot_and_save_data(file_path, data, blinks, bl_starts, bl_stops, destBaseFolder):
    
    """
    Generates plots for eye tracking data and saves them along with the data in specified formats.

    This function creates plots from eye tracking data (center x, center y, diameter) and highlights the 
    periods where blinks are detected. It also saves the eye tracking data and blinks information in both 
    .npy and .csv formats in a structured directory.

    Parameters
    ----------
    file_path : str
        Path to the file containing the raw data used for generating eye tracking information.
    data : ndarray
        Array containing eye tracking data, specifically center x, center y, and diameter values.
    blinks : ndarray
        Boolean array indicating the frames where blinks are detected.
    bl_starts : list
        List containing the start frames of each blink.
    bl_stops : list
        List containing the stop frames of each blink.
    destBaseFolder : str
        Base directory where the output files and plots will be saved.

    Notes
    -----
    - The function creates a new directory within the destination folder to store the output files and plots.
    - The plots generated show the eye tracking data over time and mark the periods of blinks with a gray overlay.
    - The eye tracking data is saved in .npy format, and the data including blink information is saved in a .csv file.
    - The plots are saved as a single .png file showing all three metrics (center x, center y, diameter) across different subplots.
    """

    # Create the subplots
    fig, ax = plt.subplots(3, 1, figsize=(15, 12))

    # Combine the data into a single array and give it column names
    names = ['center x', 'center y', 'diameter']
    
    # Create destination folder for pupil diameter files: 
    
    session_folder = os.path.basename(os.path.dirname(file_path))
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
    
    # Create the new directory
    measure_pupil_folder = os.path.join(base_dir, "xyPos_diameter", session_folder)
    if not os.path.exists(measure_pupil_folder):
        os.makedirs(measure_pupil_folder)
    
    # Create a copy of the data accounting for blinks
    data_with_blinks = np.copy(data)
    for i in range(len(data_with_blinks)):
        if blinks[i]:
            data_with_blinks[i, :] = np.nan
    # Create two separate arrays for xyPos and diameter
    xyPos = data_with_blinks[:, :2]
    diameter = data_with_blinks[:, 2]
    
    # Save xyPos and diameter as .npy files
    np.save(os.path.join(measure_pupil_folder, 'eye.xyPos.npy'), xyPos)
    np.save(os.path.join(measure_pupil_folder, 'eye.diameter.npy'), diameter)        
    # # Save the data with blinks to a CSV file
    # df_with_blinks = pd.DataFrame(data_with_blinks)
    # df_with_blinks.to_csv(os.path.join(measure_pupil_folder, 'diameter_blinks.npy'), index=False)

    # Create the plots
    for s in range(3):
        mini = np.min(data[blinks == False, s])
        maxi = np.max(data[blinks == False, s])
        rng = maxi - mini
        mini = mini - 0.02 * rng
        maxi = maxi + 0.02 * rng

        ax[s].plot(data[:, s], 'k')
        ax[s].set_ylim([mini, maxi])
        ax[s].set_ylabel(names[s])

        if len(bl_starts) > 0:
            for start, stop in zip(bl_starts, bl_stops):
                ax[s].fill_betweenx([mini, maxi], start, stop, color='gray', alpha=0.5)
    
    ax[-1].set_xlabel('frames')

    # Save the plot
    plt.savefig(os.path.join(measure_pupil_folder, "xyPos_diameter_blinks.png"))


