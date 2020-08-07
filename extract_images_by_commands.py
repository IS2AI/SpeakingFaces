from speakingfacespy.imtools import make_dir
from imutils import paths
import scipy.io.wavfile
import argparse 
import cv2
import os

def extract_frame(video_path, duration, path_to_image, sub_id, trial_id, pos_id, cmd_id, stream_id):
    # initialize the video stream and 
    # pointer to output video file
    vs =  cv2.VideoCapture(video_path)

    # initialize the maximum number of 
    # frames need to be extracted
    max_num_frames = int(duration * 28)

    # initialize a frame counter
    frame_id = 1

    while(vs.isOpened() and frame_id <= max_num_frames):
        #read the next frame from the file
        grabbed, frame = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # construct frame name and path
        # to save the frame
        frame_name = "{}_{}_2_{}_{}_{}_{}.png".format(sub_id, trial_id, pos_id, cmd_id, frame_id, stream_id)
        path_to_save_image = os.path.join(path_to_image, frame_name)

        #cv2.imshow("Frame", frame)
        #cv2.waitKey(0)

        # save the frame
        print(path_to_save_image)
        cv2.imwrite(path_to_save_image, frame)
        
        # increment the frame counter by one
        frame_id += 1
    
    # do a bit cleanup
    vs.release()
    cv2.destroyAllWindows()
    

def extract_frames_by_sub_trial(path_to_dataset, sub_id, trial_id):
    # construct a path to the data for the given subject and trial
    path_to_dataset = os.path.join(path_to_dataset, "sub_{}/trial_{}".format(sub_id, trial_id))

    # create a directory to save extracted rgb images
    rgb_image_path = os.path.join(path_to_dataset, "rgb_image_cmd")
    make_dir(rgb_image_path)

    # and thermal images
    thr_image_path = os.path.join(path_to_dataset, "thr_image_cmd") 
    make_dir(thr_image_path)

    # construct paths to thermal and rgb videos
    rgb_video_path = os.path.join(path_to_dataset, "rgb_video_cmd")
    thr_video_path = os.path.join(path_to_dataset, "thr_video_cmd")
    
    # construct a path to the trimmed audiofiles
    path_to_audiofiles = os.path.join(path_to_dataset, "mic1_audio_cmd_trim") 

    # grab the list of audiofiles 
    audio_file_paths = list(paths.list_files(path_to_audiofiles))

    # loop over the audiofiles
    for audio_file_path in audio_file_paths:
        # extract audio filename
        audio_filename = audio_file_path.split(os.path.sep)[-1]
        
        # extract the position and command ID from 
        # the audio file name
        pos_id, cmd_id = audio_filename.split("_")[3:5]
 
        # estimate the duration of the audio file in seconds 
        sample_rate, audio_data = scipy.io.wavfile.read(audio_file_path)
        duration = audio_data.shape[0] / sample_rate

        # construct filenames to thermal and rgb videos
        rgb_video_filename = "{}_{}_2_{}_{}_2.avi".format(sub_id, trial_id, pos_id, cmd_id)
        thr_video_filename = "{}_{}_2_{}_{}_1.avi".format(sub_id, trial_id, pos_id, cmd_id)

        # construct paths to the rgb and thermal videos
        rgb_video = os.path.join(rgb_video_path, rgb_video_filename)
        thr_video = os.path.join(thr_video_path, thr_video_filename)

        # extract thermal and rgb frames
        extract_frame(rgb_video, duration, rgb_image_path, sub_id, trial_id, pos_id, cmd_id, 2)
        extract_frame(thr_video, duration, thr_image_path, sub_id, trial_id, pos_id, cmd_id, 1)  

      
def extract_frames_by_range(path_to_dataset, sub_id_i, sub_id_l):
    # loop over the subjects and trials
    for sub_id in range(sub_id_i, sub_id_l + 1):
        for trial_id in range(1, 3):
            extract_frames_by_sub_trial(path_to_dataset, sub_id, trial_id)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
        help="path to the SpeakingFaces dataset")
ap.add_argument("-i", "--sub_info", nargs='+', type=int, default = (0,0),
        help="subID(1,...,142) trialID (1,2)")
ap.add_argument("-r", "--sub_range", nargs='+', type=int, default = (0,0), 
        help="process more than one subject (1...142)")
args = vars(ap.parse_args())

# initialize a path to the dataset
path_to_dataset = args["dataset"]

# if we are going to extract frames for a specific subject 
if args["sub_info"][0] > 0 and args["sub_info"][0] <= 142:
    # initialize the subject ID and 
    # trial ID
    sub_id = args["sub_info"][0]
    trial_id = args["sub_info"][1]
    
    if trial_id >0 and trial_id <= 2:
        # extract frames for the given subject and trial
        extract_frames_by_sub_trial(path_to_dataset, sub_id, trial_id)
    else:
        print("[INFO] Number of trials are invalid!")

# if we want to process more than one subject 
elif args["sub_range"][0] > 0 and args["sub_range"][0] <= 142:
    # initialize IDs of the initial
    # and the last subjects
    sub_id_i = args["sub_range"][0]
    sub_id_l = args["sub_range"][1]

    # extract frames
    extract_frames_by_range(path_to_dataset, sub_id_i, sub_id_l)

# otherwise ask to enter proper arguments
else:
    print("[INFO] --sub_info or/and --sub_range was/were not defined properly!")
