# The code reads a csv file row by row and creates an updated video (or none) file for each row
# This file is for thermal videos

import os
import argparse
import pandas as pd
import shutil
import cv2
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def make_dir(dirName):
    # create target directory & all intermediate directories if don't exist
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("[INFO] Directory ", dirName, " created")
    else:
        print("[INFO] Directory ", dirName, " already exists")


def get_input_filepath(input_name, dataset_path, stream_id):
    raw_name_list = input_name.split("_")
    sub_id = int(raw_name_list[0])
    trial_id = raw_name_list[1]
    if sub_id in range(121, 143):
        data_folder = 'test_data'
    elif sub_id in range(101, 121):
        data_folder = 'valid_data'
    else:
        data_folder = 'train_data'
    opt = 'thr' if stream_id == 1 else 'rgb'
    input_filepath = '{}/{}/sub_{}/trial_{}/{}_video_cmd/{}.avi'.format(dataset_path, data_folder, sub_id, trial_id, opt, input_name)
    return input_filepath


def get_output_dir(input_name, dataset_path, stream_id):
    raw_name_list = input_name.split("_")
    sub_id = int(raw_name_list[0])
    trial_id = raw_name_list[1]
    if sub_id in range(121, 143):
        data_folder = 'test_data'
    elif sub_id in range(101, 121):
        data_folder = 'valid_data'
    else:
        data_folder = 'train_data'
    opt = 'thr' if stream_id == 1 else 'rgb'
    output_dir = '{}/salvaged_files/{}/sub_{}/trial_{}/{}_image_cmd'.format(dataset_path, data_folder, sub_id, trial_id, opt)
    make_dir(output_dir)
    return output_dir


def get_output_filepath(input_name, output_dir):
    # rename an audio file if needed
    if pd.notna(df.new_name[i]):
        old_name = input_name
        name_list = input_name.split("_")
        name_list[4] = str(int(df.new_name[i]))
        input_name = "_".join(name_list)
        print('[INFO] {}.avi renamed as {}.avi'.format(old_name, input_name))
    return '{}/{}.avi'.format(output_dir, input_name)


# function is the same as copy_audio(df, output_folder)
def copy_video(input_filepath, output_filepath):
    shutil.copy(input_filepath, output_filepath)


def extract_frame(filename, duration, output_dir, frame_id=1):
    cap = cv2.VideoCapture(filename)
    max_num_frames = int(duration * 28)
    while (cap.isOpened() and frame_id <= max_num_frames):
        ret, frame = cap.read()
        if ret == False:
            break
        extracted_image_filename = '_'.join(input_name.split('_')[:-1]) + '_{}_1.png'.format(frame_id)
        extracted_image_file = output_dir + extracted_image_filename
        # prints every 10 extracted frames
        if frame_id % 10 == 0:
            print('[INFO] saving an extracted frame: ' + extracted_image_file)
        cv2.imwrite(extracted_image_file, frame)
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print('[INFO] done extracting frames from {}.avi'.format(input_name))
    return frame_id


def trim_video(df, video_duration, output_video_file, output_dir):
    # use video duration for end timestamp, if NaN
    if pd.isna(df.end_timestamp[i]):
        df.end_timestamp[i] = video_duration
        print('[INFO] done updating end timestamp for {}.avi'.format(input_name))

    # trim a video file based on timestamps and save it
    ffmpeg_extract_subclip(input_name + '.avi', df.begin_timestamp[i], df.end_timestamp[i],
                           targetname=output_video_file)

    # extract frames from a video file
    extract_frame(output_video_file, video_duration, output_dir)


def merge_and_trim_video(df, video_duration, output_dir):
    # trim the first video file based on begin_timestamp
    ffmpeg_extract_subclip(input_name + '.avi', df.begin_timestamp[i], video_duration,
                           targetname='{}/{}_begin.avi'.format(output_dir, input_name))

    # trim the second video file based on end_timestamp
    ffmpeg_extract_subclip(df.raw_audio_name[i+1] + '.avi', 0, df.end_timestamp[i],
                           targetname='{}/{}_end.avi'.format(output_dir, input_name))

    # extract frames from the first video
    duration = (video_duration - df.begin_timestamp[i])
    frame_id = extract_frame('{}/{}_begin.avi'.format(output_dir, input_name), duration, output_dir)

    # extract frames from the second video
    duration = (df.end_timestamp[i])
    extract_frame('{}/{}_end.avi'.format(output_dir, input_name), duration, output_dir, frame_id)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset")

args = vars(ap.parse_args())

dataset_path = args["dataset"]
print(dataset_path)

# read a csv file without the last column
df = pd.read_csv(dataset_path + '/csvs/salvage_mission_train_data.csv')
df = df.iloc[:, :-1]
print(df.head())

for stream_id in range(2):
    for i in range(len(df)):
        input_name = df.raw_audio_name[i]
        input_video_file = get_input_filepath(input_name, dataset_path, stream_id)
        # check if a file exists
        if os.path.isfile(input_video_file):
            video_duration = VideoFileClip(input_video_file).duration
            output_dir = get_output_dir(input_name, dataset_path, stream_id)
            output_video_file = get_output_filepath(input_name, output_dir)

            # 4 categories: copy, trim, merge and trim, skip
            if df.comment[i] == 'copy':
                copy_video(input_video_file, output_video_file)
                print('[INFO] {}.avi is copied'.format(input_name))

            if df.comment[i] == 'trim':
                trim_video(df, video_duration, output_video_file)
                print('[INFO] done trimming {}.avi'.format(input_name))

            if df.comment[i] == 'merge and trim':
                merge_and_trim_video(df, video_duration, output_dir)
                print('[INFO] done merging and trimming {}.avi'.format(input_name))

            if 'skip' in df.comment[i]:
                print('[INFO] {}.avi is skipped'.format(input_name))

        else:
            print("[ERROR] {}.avi doesn't exist".format(input_name))