# SpeakingFaces
This repository contains the source code for preprocessing SpeakingFaces dataset. The SpeakingFaces consists of well-aligned high-resolution thermal and visual spectra image streams of fully-framed faces synchronized with audio recordings of each subject speaking 100 imperative phrases. Data was collected from 142 subjects, yielding over 14,000 instances of synchronized data (7.5 TB).

## Dependencies
1. Ubuntu 16.04
2. MATLAB 2019.x
3. Python 3.x.x
4. OpenCV 4.x.x
5. NumPy -> **pip install numpy**
6. Pandas -> **pip install pandas**
7. SciPy -> **pip install scipy**
8. imutils -> **pip install imutils** 
9. face_recognition -> **pip install face_recognition**

## Data Acquisition
FLIR T540 thermal camera (resolution 464×348 pixels, wave band 7.5μm- 14μm, the field of view 24◦) and a Logitech C920 Pro HD web-camera (768×512 pixels, the field of view 78◦) with a builtin dual stereo microphone was used for data collection purpose.  

![setup](https://raw.githubusercontent.com/IS2AI/SpeakingFaces/master/figures/setup.png)
