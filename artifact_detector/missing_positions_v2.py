# import the necessary packages
import csv
#from imutils import paths
#from ffprobe import FFProbe
import pandas as pd
import scipy.io.wavfile as wv
import cv2
import argparse
import os


def append_to_csv(filename, list_of_artifacts):
    with open(filename, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(list_of_artifacts)

# parse the provided arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--dataset", required=True, help="path to the dataset")
ap.add_argument("-i", "--sub_info", nargs='+', type=int, help="subID(1,...,142) trialID (1,2) posID(1,...,9)")

args = vars(ap.parse_args())

sub_id = args["sub_info"][0]
trial_id = args["sub_info"][1]

# grab the path to our dataset
sub_trial_dataset_path = "{}sub_{}".format(args["dataset"], args["sub_info"][0]) + os.path.sep + "trial_{}".format(
    args["sub_info"][1])
thr_dirname = sub_trial_dataset_path + os.path.sep + "thr_video_cmd" + os.path.sep
rgb_dirname = sub_trial_dataset_path + os.path.sep + "rgb_video_cmd" + os.path.sep

# get the path to audio files
audio_mic1 = sub_trial_dataset_path + os.path.sep + "mic1_audio_cmd" + os.path.sep
audio_mic2 = sub_trial_dataset_path + os.path.sep + "mic2_audio_cmd" + os.path.sep

# get csv file for the trial number
csv_file_path = "{}{}csvs{}{}{}commands{}commands_sub{}_trial{}.csv".format(
    os.path.dirname(os.path.dirname(args["dataset"])), os.path.sep, os.path.sep,
    args["dataset"].split(os.path.sep)[-2], os.path.sep, os.path.sep,
    sub_id, trial_id)

# get the command id, position id, start and end frame ids from the csv file
df = pd.read_csv(csv_file_path)
command_pos_char_list = df[['command_id', 'pos_id', 'start_fr_id', 'end_fr_id']].values.astype(int)


h = ['sub_id', 'filename', 'trial_id', 'command_id', 'stream_id', 'comment']
missing_files_csv = 'missing_files.csv'
missing_files_csv_audio = 'missing_files_audio.csv'

if not os.path.exists(missing_files_csv):
    append_to_csv(missing_files_csv, h)
    append_to_csv(missing_files_csv_audio, h)

for command in command_pos_char_list:
    command_id, pos_id, start_frame_id, end_frame_id = command
    # grab video files
    thr_video_filepath = "{}{}_{}_2_{}_{}_1.avi".format(thr_dirname, sub_id, trial_id, pos_id, command_id)
    rgb_video_filepath = "{}{}_{}_2_{}_{}_2.avi".format(rgb_dirname, sub_id, trial_id, pos_id, command_id)

    # grab audio files
    audio_mic1_filepath = "{}{}_{}_2_{}_{}_1.wav".format(audio_mic1, sub_id, trial_id, pos_id, command_id)
    audio_mic2_filepath = "{}{}_{}_2_{}_{}_2.wav".format(audio_mic2, sub_id, trial_id, pos_id, command_id)

    # if files exist check the number of frames in the files if less than the expected number and append to csv if so
    if os.path.exists(thr_video_filepath) and os.path.exists(rgb_video_filepath) and os.path.exists(audio_mic1_filepath) and os.path.exists(audio_mic2_filepath):
        #num_of_frames = FFProbe(thr_video_filepath).streams[0].frames()
        thr_cap = cv2.VideoCapture(thr_video_filepath)
        thr_num_of_frames = int(thr_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_frame_num = (int(end_frame_id)-int(start_frame_id)) + 1

        rgb_cap = cv2.VideoCapture(rgb_video_filepath)
        rgb_num_of_frames = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # get signal rate and length of audio files
        audio_mic1_sr, audio_mic1_file = wv.read(audio_mic1_filepath)
        audio_mic1_duration = len(audio_mic1_file)/audio_mic1_sr

        audio_mic2_sr, audio_mic2_file = wv.read(audio_mic2_filepath)
        audio_mic2_duration = len(audio_mic2_file)/audio_mic2_sr

        if thr_num_of_frames < expected_frame_num:
            file_name = thr_video_filepath.split(os.path.sep)[-1]
            append_to_csv(missing_files_csv, [sub_id, file_name, trial_id, pos_id, command_id, 1, 'Less frame number than expected'])
            print('[INFO] The file {} has less frame number than expected'.format(file_name))

        if rgb_num_of_frames < expected_frame_num:
            file_name = rgb_video_filepath.split(os.path.sep)[-1]
            append_to_csv(missing_files_csv, [sub_id, file_name, trial_id, pos_id, command_id, 2, 'Less frame number than expected'])
            print('[INFO] The file {} has less frame number than expected'.format(file_name))

        if audio_mic1_duration < audio_mic2_duration:
            file_name = audio_mic1_filepath.split(os.path.sep)[-1]
            append_to_csv(missing_files_csv_audio, [sub_id, file_name, trial_id, pos_id, command_id, 1, 'Less audio length'])
            print('[INFO] The file {} has less audio length than expected'.format(file_name))

        elif audio_mic2_duration < audio_mic1_duration:
            file_name = audio_mic2_filepath.split(os.path.sep)[-1]
            append_to_csv(missing_files_csv_audio, [sub_id, file_name, trial_id, pos_id, command_id, 2, 'Less audio length'])
            print('[INFO] The file {} has less audio length than expected {} {}'.format(file_name, expected_frame_num, audio_mic2_duration))


    # else look for which one is missing (rgb or thermal) and append to cvs file
    else:
        if not os.path.exists(thr_video_filepath):
            file_name = thr_video_filepath.split(os.path.sep)[-1]
            append_to_csv(missing_files_csv, [sub_id, file_name, trial_id, pos_id, command_id, 1, 'file missing'])
            print('[INFO] The file {} doesn\'t exist'.format(file_name))

        if not os.path.exists(rgb_video_filepath):
            file_name = rgb_video_filepath.split(os.path.sep)[-1]
            append_to_csv(missing_files_csv, [sub_id, file_name, trial_id, pos_id, command_id, 2, 'file missing'])
            print('[INFO] The file {} doesn\'t exist'.format(file_name))

        if not os.path.exists(audio_mic1_filepath):
            file_name = audio_mic1_filepath.split(os.path.sep)[-1]
            append_to_csv(missing_files_csv_audio, [sub_id, file_name, trial_id, pos_id, command_id, 1, 'file missing'])
            print('[INFO] The file {} doesn\'t exist'.format(file_name))

        if not os.path.exists(audio_mic2_filepath):
            file_name = audio_mic2_filepath.split(os.path.sep)[-1]
            append_to_csv(missing_files_csv_audio, [sub_id, file_name, trial_id, pos_id, command_id, 2, 'file missing'])
            print('[INFO] The file {} doesn\'t exist'.format(file_name))

    print("[INFO] The processed files: {} and {}. ".format(rgb_video_filepath.split(os.path.sep)[-1],
                                                           thr_video_filepath.split(os.path.sep)[-1]))