# SpeakingFaces
This repository contains the source code developed for collecting and preprocessing SpeakingFaces dataset. The SpeakingFaces consists of well-aligned high-resolution thermal and visual spectra image streams of fully-framed faces synchronized with audio recordings of each subject speaking 100 imperative phrases. Data were collected from 142 subjects, yielding over 14,000 instances of synchronized data (7.5 TB).

## Dependencies
1. Ubuntu 16.04
2. MATLAB 2019.x with ROS Toolbox
3. Python 3.x.x
4. OpenCV 4.x.x
5. NumPy -> **pip install numpy**
6. Pandas -> **pip install pandas**
7. SciPy -> **pip install scipy**
8. imutils -> **pip install imutils** 
9. face_recognition -> **pip install face_recognition**

## Data Acquisition
FLIR T540 thermal camera (464×348 pixels, 24◦ FOV) and a Logitech C920 Pro HD web-camera (768×512 pixels, 78◦ FOV) with a built-in dual stereo microphone were used for data collection purpose. Each subject participated in two trials where each trial consisted of **two sessions**. In the **first session**, subjects were silent and still, with the operator capturing the **visual** and **thermal** video streams through the procession of nine collection angles. The **second session** consisted of the subject reading a series of commands as presented one-by-one on the video screens, as the **visual**, **thermal** and **audio** data was collected from the same nine camera positions.
<img src="https://raw.githubusercontent.com/IS2AI/SpeakingFaces/master/figures/nine_positions_v5.png" width="600">
### Data recording for the first session:
1. Launch the MATLAB and start the global ROS node via MATLAB's terminal: **rosinit**
2. Open **/record_matlab/record_only_video.m** and initialize the following parameters:
- **sub_id**: a subject ID.
- **trial_id**: a trial ID.
- **numOfPosit**: a total number of positions.
- **numOfFrames**: a total number of frames per position. 
- **pauseBtwPos**: a pause (sec) that is necessary for moving cameras from one position to another.
- **path**: a path to save the recorded video files.
3. Launch **/record_matlab/record_only_video.m**
### Data recording for the second session:
1. Launch the MATLAB and start the global ROS node via MATLAB's terminal: **rosinit**
2. Open **/record_matlab/record_audio_video.m** and initialize the following parameters:
- **sub_id**: a subject ID.
- **trial_id**: a trial ID.
- **numOfPosit**: a total number of positions.
- **fpc**: a number of frames necessary for reading one character. 
- **pauseBtwPos**: a pause (sec) that is necessary for moving cameras from one position to another.
- **path**: a path to save the recorded audio and video files.
3. Launch **/record_matlab/record_audio_video.m**

## Data preprocessing
### Preprocessing data from the first session 
1. To extract frames from the recorded visual and thermal video streams:
- Open **record_matlab/extract_images_from_videos.m** file. 
- Initialize the following parameters: **sub_id** - a subject ID, **trial_id** - a trial ID, **path_vid** - a path to the recorded video files, **path_rgb** - a path to save the extracted visual frames, **path_thr** - a path to save the extracted thermal frames.
- Launch **record_matlab/extract_images_from_videos.m**
2. To align the extracted visual frames with corresponding thermal frames, run **align_session_one.py** script with the following arguments:
- **dataset**: a path to the SpeakingFaces dataset.
- **sub_info**: subjectID, trialID, positionID.
- **dy**: a list of shifts (pixels) between streams in y-axis.
- **dx**: a list of shifts (pixels) between streams in x-axis.
- **show**: visualize (1) or not (0) a preliminary result of the alignment.

<img src="https://raw.githubusercontent.com/IS2AI/SpeakingFaces/master/figures/aligned_session_one.png" width="900">

### Preprocessing data from the second session
1. To split audio and video files based on commands:
- Open **record_matlab/extract_video_audio_by_commands.m** file.
- Initialize the following parameters: **sub_id** - a subject ID, **trial_id** - a trial ID, **path_vid** - a path to the recorded video files, **path_rgb** - a path to save the extracted visual videos, **path_thr** - a path to save the extracted thermal videos by commands, **path_mic1** - a path to save the trimmed audio files by commands from the microphone #1, **path_mic2** - a path to save the trimmed audio files by commands from the microphone #2.
- Launch **record_matlab/extract_video_audio_by_commands.m**
2. To extract, align and save frames, and also to crop the region of interest (ROI) from the splitted video streams, run **align_crop_session_two.py** script with the following arguments:
- **dataset**: a path to the SpeakingFaces dataset.
- **sub_info**: subjectID, trialID, positionID.
- **dy**: a list of shifts (pixels) between streams in y-axis.
- **dx**: a list of shifts (pixels) between streams in x-axis.
- **model**: a face detection model.
- **show**: visualize (1) or not (0) a preliminary result of the alignment.
If the face detection is selected then cropping the ROI is done automatically based on the facial landmarks:
<img src="https://raw.githubusercontent.com/IS2AI/SpeakingFaces/master/figures/aligned_session_two.png" width="900">

Otherwise the ROI is defined manually:
<img src="https://raw.githubusercontent.com/IS2AI/SpeakingFaces/master/figures/aligned_session_two_manual.png" width="900">


