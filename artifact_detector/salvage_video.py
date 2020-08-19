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


def get_output_filepath(df, input_name, output_dir):
    # rename a video file if needed
    if pd.notna(df.new_name[i]):
        old_name = input_name
        name_list = input_name.split("_")
        name_list[4] = str(int(df.new_name[i]))
        input_name = "_".join(name_list)
        print('[INFO] {}.avi renamed as {}.avi'.format(old_name, input_name))

    output_filepath = '{}/{}.avi'.format(output_dir, input_name)
    return output_filepath


def extract_frame(input_name, input_filepath, duration, output_dir, frame_id=1):
    cap = cv2.VideoCapture(input_filepath)
    max_num_frames = int(duration * 28)
    while (cap.isOpened() and frame_id <= max_num_frames):
        ret, frame = cap.read()
        if ret == False:
            break
        extracted_image_filename = '_'.join(input_name.split('_')[:-1]) + '_{}_1.png'.format(frame_id)
        extracted_image_file = output_dir + extracted_image_filename
        # print every 10 extracted frames
        if frame_id % 10 == 0:
            print('[INFO] saving an extracted frame: ' + extracted_image_file)
        cv2.imwrite(extracted_image_file, frame)
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    return frame_id


def remove_avi_extension(output_dir):
    test = os.listdir(output_dir)
    for item in test:
        if item.endswith('.avi'):
            os.remove(os.path.join(output_dir, item))


# construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset")

args = vars(ap.parse_args())

dataset_path = args["dataset"]
print(dataset_path)

# read a csv file without the last column
df = pd.read_csv(dataset_path + '/csvs/salvage_mission_all_data.csv')
df = df.iloc[:, :-1]
print(df.head())

for stream_id in range(1, 3):
    for i in range(len(df)):
        # read a video file
        input_name = df.raw_audio_name[i][:-1] + str(stream_id)

        # get full information about the current trial
        raw_name_list = input_name.split("_")
        sub_id = int(raw_name_list[0])
        trial_id = raw_name_list[1]
        if sub_id in range(121, 143):
            data_folder = 'test_data'
        elif sub_id in range(101, 121):
            data_folder = 'valid_data'
        else:
            data_folder = 'train_data'
        trial_info = "{}/sub_{}/trial_{}".format(data_folder, sub_id, trial_id)

        opt = 'thr' if stream_id == 1 else 'rgb'
        input_filepath = '{}/{}/{}_video_cmd/{}.wav'.format(dataset_path, trial_info, opt, input_name)

        # check if a file exists
        if os.path.isfile(input_filepath):
            # read video duration and timestamps
            video_duration = VideoFileClip(input_filepath).duration
            begin_time = df.begin_timestamp[i]
            end_time = df.end_timestamp[i]

            # make the directory for the output file and get the updated filepath
            output_dir = '{}/salvaged_files/{}/mic{}_image_cmd'.format(dataset_path, trial_info, opt)
            make_dir(output_dir)
            output_filepath = get_output_filepath(df, input_name, output_dir)

            # 4 categories: copy, trim, merge and trim, skip
            if df.comment[i] == 'copy':
                shutil.copy(input_filepath, output_filepath)
                print('[INFO] {}.avi is copied'.format(input_name))

            elif df.comment[i] == 'trim':
                # use video duration for end timestamp, if NaN
                if pd.isna(end_time):
                    end_time = video_duration

                # trim a video file based on timestamps and save it
                ffmpeg_extract_subclip(input_filepath, begin_time, end_time, targetname=output_filepath)

                # extract frames from a video file
                extract_frame(input_name, output_filepath, video_duration, output_dir)
                
                print('[INFO] done trimming {}.avi'.format(input_name))

            elif df.comment[i] == 'merge and trim':
                # read a video file on the next row
                input_filepath_2 = '{}/{}/{}_video_cmd/{}.avi'.format(dataset_path, trial_info, opt, 
                                                                      df.raw_audio_name[i + 1])
                
                # trim the first video file based on begin_timestamp
                ffmpeg_extract_subclip(input_filepath, begin_time, video_duration, 
                                       targetname=output_filepath[:-4] + '_begin.avi')

                # trim the second video file based on end_timestamp
                ffmpeg_extract_subclip(input_filepath_2, 0, end_time, targetname=output_filepath[:-4] + '_end.avi')

                # extract frames from the first video
                duration = (video_duration - begin_time)
                frame_id = extract_frame(input_name, output_filepath[:-4] + '_begin.avi', duration, output_dir)

                # extract frames from the second video
                duration = end_time
                extract_frame(input_name, output_filepath[:-4] + '_end.avi', duration, output_dir, frame_id)

                print('[INFO] done merging and trimming {}.avi'.format(input_name))

            else:
                print('[INFO] {}.avi is skipped'.format(input_name))

            remove_avi_extension(output_dir)

        else:
            print("[ERROR] {}.avi doesn't exist".format(input_name))
