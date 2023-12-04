# EyeVideoAnalysis
Analysis output of trained DeepLabCut (DLC) model to detect pupil position, size, and blinks.

This code will analyse the videos with the DeepLabCut model which gives an output of coordinates of each marker around the pupil (4) and the eye (4) + nose (1). Then those markers are used to estimate the pupil diameter throughout the video accounting for blinks and cases where DLC output is not certain enough. 
Optionally, DLC can provide plots and labelled video that can be used to evaluate the quality of video analysis. 
Additionally, cropped versions of labelled videos can be generated to focus more on the eye area. 

## Installation and Use

1) Create a conda environment supplied for DeepLabCut following the instructions here: https://deeplabcut.github.io/DeepLabCut/docs/installation.html
2) Download the DeepLabCut model from here: 
3) Clone this repository
4) Fill out details of videos to be analysed in the pupil_config file
5) Set 4 paths in user_defs (for raw database, processed database, DLC model path, pupil_config file path)
6) Run main_pupil in the DLC environment
7) Set ROIs if crop_videos is set to TRUE in pupil_config file
8) Check videos and start data analysis!


## Credits
The code for measure_pupil was adapted from Sylvia Schroeder's work: https://github.com/sylviaschroeder/PupilDetection_DLC 


