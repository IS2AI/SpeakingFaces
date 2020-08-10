from speakingfacespy.imtools import make_dir
from imutils import paths
import scipy.io.wavfile
import argparse
import glob
import os

def trim_audio(audio_trim_filename, length_audio_trim, to_trim_mic_id, to_trim_new_dir):
    # construct a filename for the audio which we wanna trim
    audio_to_trim_filename  = '_'.join(audio_trim_filename.split('_')[:-1])+'_{}.wav'.format(to_trim_mic_id)

    # construct a path to save the audio file
    audio_to_trim_new_filepath = os.path.join(to_trim_new_dir, audio_to_trim_filename)
    
    # extract the sample rate and audio data
    print("[INFO] Reading an audio file to trim: {}".format(audio_to_trim_filename))
    sample_rate_to_trim, audio_to_trim = scipy.io.wavfile.read(os.path.join(to_trim_new_dir[:-5], audio_to_trim_filename))

    # estimate the duration of the file
    duration_to_trim = audio_to_trim.shape[0] / sample_rate_to_trim
    
    print("[INFO] Duration of the original file = {} sec".format(duration_to_trim))
    print("[INFO] Writing trimmed audio to {}".format(audio_to_trim_new_filepath))
    scipy.io.wavfile.write(audio_to_trim_new_filepath, sample_rate_to_trim, audio_to_trim[:length_audio_trim])
        

def trim_audio_by_sub_trial(path_to_dataset, sub_id, trial_id):
    # construct a path to the data for the given subject and trial
    path_to_dataset = os.path.join(path_to_dataset, "sub_{}/trial_{}".format(sub_id, trial_id))

    # find a folder with audio files
    for mic_id in [1, 2]:
        # construct a path to the trimmed audiofiles
        path_to_audiofiles = os.path.join(path_to_dataset, "mic{}_audio_cmd_trim".format(mic_id))

        # grab all audiofiles for the given directory  
        audio_file_paths = list(paths.list_files(path_to_audiofiles))

        # if the folder is not empty
        # then exit from the loop
        if len(audio_file_paths) > 0:
            break

    # createa a directory to save new trimmed audio files
    if mic_id == 1:
        to_trim_dir = os.path.join(path_to_dataset, "mic{}_audio_cmd_trim".format(2)) 
        to_trim_mic_id = 2
    else:
        to_trim_dir = os.path.join(path_to_dataset, "mic{}_audio_cmd_trim".format(1))
        to_trim_mic_id = 1
    
    make_dir(to_trim_dir)

    # loop over the trimmed audio files
    for audio_file_path in audio_file_paths:
        # extract the filename
        audio_trim_filename = audio_file_path.split(os.path.sep)[-1]

        print("[INFO] Reading already trimmed audio file: {}".format(audio_trim_filename))
        # extract the sample rate and audio data from
        # the already trimmed file
        sample_rate_trim, audio_trim = scipy.io.wavfile.read(audio_file_path)

        # estimate the duration of the already trimmed file
        duration_trim = audio_trim.shape[0] / sample_rate_trim
        print("[INFO] Duration of the already trimmed file = {} sec".format(duration_trim))
            
        #trim the corresponding audio file from the other
        # mic accordingly
        trim_audio(audio_trim_filename, audio_trim.shape[0], to_trim_mic_id, to_trim_dir)


def trim_audio_by_range(path_to_dataset, sub_id_i, sub_id_l):
    print("[INFO] trim all audio files in the range of sub_id [{} ... {}]".format(sub_id_i, sub_id_l))
    for sub_id in range(sub_id_i, sub_id_l + 1):
        for trial_id in range(1, 3):
            trim_audio_by_sub_trial(path_to_dataset, sub_id, trial_id)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
        help="path to the SpeakingFaces dataset")
ap.add_argument("-i", "--sub_info", nargs='+', type=int, default = (0,0),
        help="subID(1,...,142) trialID (1...2)")
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
        trim_audio_by_sub_trial(path_to_dataset, sub_id, trial_id)
    else:
        print("[INFO] Number of trials are invalid!")

# if we want to process more than one subject 
elif args["sub_range"][0] > 0 and args["sub_range"][0] <= 142:
    # initialize IDs of the initial
    # and the last subjects
    sub_id_i = args["sub_range"][0]
    sub_id_l = args["sub_range"][1]

    # extract frames
    trim_audio_by_range(path_to_dataset, sub_id_i, sub_id_l)

# otherwise ask to enter proper arguments
else:
    print("[INFO] --sub_info or/and --sub_range was/were not defined properly!")
