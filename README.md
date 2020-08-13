<img src="https://raw.githubusercontent.com/IS2AI/SpeakingFaces/master/figures/speakingface.jpg" width="1280">
This repository contains the source code developed for collecting and preprocessing SpeakingFaces dataset. The SpeakingFaces consists of well-aligned high-resolution thermal and visual spectra image streams of fully-framed faces synchronized with audio recordings of each subject speaking 100 imperative phrases. Data were collected from 142 subjects, yielding over 14,000 instances of synchronized data (7.5 TB).

## Dependencies
### For recording data
1. OS Windows 10
2. FLIR Atlas SDK for MATLAB
3. MATLAB 2019.x 
- Computer Vision Toolbox
- ROS Toolbox
### For pre-processing data
1. Ubuntu 16.04
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
2. To align the extracted visual frames with their thermal pairs, run **align_session_one.py** script with the following arguments:
- **dataset**: a path to the SpeakingFaces dataset.
- **sub_info**: subjectID, trialID, positionID.
- **dy**: a list of shifts (pixels) between streams in y-axis.
- **dx**: a list of shifts (pixels) between streams in x-axis.
- **show**: visualize (1) or not (0) a preliminary result of the alignment.

<img src="https://raw.githubusercontent.com/IS2AI/SpeakingFaces/master/figures/aligned_session_one.png" width="900">

3. Alignment for all subjects can be done using **align_session_one_all.py** script and metadata stored in **metadata/session_1/** directory. 

### Preprocessing data from the second session
1. To split audio and video files based on commands:
- Open **record_matlab/extract_video_audio_by_commands.m** file.
- Initialize the following parameters: **sub_id** - a subject ID, **trial_id** - a trial ID, **path_vid** - a path to the recorded video files, **path_rgb** - a path to save the extracted visual videos, **path_thr** - a path to save the extracted thermal videos by commands, **path_mic1** - a path to save the extracted audio files by commands from the microphone #1, **path_mic2** - a path to save the extracted audio files by commands from the microphone #2.
- Launch **record_matlab/extract_video_audio_by_commands.m**
2. Note, the recordings from the two microphones are identical. The extracted audio files from the first microphone were manually trimmed and validated. Based on this processed data, the extracted recordings from the second microphone were automatically trimmed using **trim_audio.py** script.
3. To extract frames from visual and thermal video files based on the trimmed audio files, run **extract_images_by_commands.py** script. It estimates a length of the trimmed audio file and extracts a necessary number of frames accordingly. 
4. To align the extracted frames, and also to crop the mouth region of interest (ROI), run **align_crop_session_two.py** script with the following arguments:
- **dataset**: a path to the SpeakingFaces dataset.
- **sub_info**: subjectID, trialID, positionID.
- **dy**: a list of shifts (pixels) between streams in y-axis.
- **dx**: a list of shifts (pixels) between streams in x-axis.
- **model**: a face detection model.
- **show**: visualize (1) or not (0) a preliminary result of the alignment.

If the face detection is selected, then the ROI is cropped automatically using facial landmarks:

<img src="https://raw.githubusercontent.com/IS2AI/SpeakingFaces/master/figures/aligned_session_two.png" width="700">

Otherwise the ROI is defined manually:

<img src="https://raw.githubusercontent.com/IS2AI/SpeakingFaces/master/figures/aligned_session_two_manual.png" width="700">

5. Alignment for all subjects can be done using **align_session_two_all.py** scripts based on the metadata stored in **metadata/session_2/** directory.


