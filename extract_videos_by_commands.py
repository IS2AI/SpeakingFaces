import os
import numpy as np
import glob
import argparse
import cv2
from speakingfacespy.imtools import make_dir
import scipy.io.wavfile
import pandas as pd

def extract_frame(audio_trim_filename, duration, data_path, stream_id):
    
    video_to_extract_filename  = '_'.join(audio_trim_filename.split('_')[:-1])+'_{}.avi'.format(stream_id)
    opt = 'thr' if stream_id == 1 else 'rgb'  
    video_to_extract_filepath = data_path + opt+ '_video_cmd' + os.path.sep + video_to_extract_filename
    print('[INFO] accessing a video file:'+ video_to_extract_filepath)
    extracted_images_dir = data_path + opt+ '_image_cmd'+ os.path.sep
    
    cap =  cv2.VideoCapture(video_to_extract_filepath)
    max_num_frames = int(duration*28)
    frame_id=1
    while(cap.isOpened() and frame_id <= max_num_frames):
        ret, frame = cap.read()
        if ret == False:
            break
        extracted_image_filename = '_'.join(audio_trim_filename.split('_')[:-1])+'_{}_{}.png'.format(frame_id, stream_id)
        extracted_image_filepath = extracted_images_dir+extracted_image_filename
        if not os.path.exists(extracted_image_filepath):
            if frame_id%10 == 0:
                print('[INFO] saving an extracted frame: '+ extracted_image_filepath)
            cv2.imwrite(extracted_image_filepath, frame)
        frame_id+=1
    
    cap.release()
    cv2.destroyAllWindows()
    print('[INFO] finished extracting frames')
    
def extract_audio_by_command(command, data_path, sub_id, trial_id, mic_id):
    fps = 28.0
    pos_id = command.pos_id
    start_fr_id = command.start_fr_id-1
    end_fr_id = command.end_fr_id
    command_id  = command.command_id
    print("[INFO] accessing command: pos_id {}, command_id {}, start_fr_id {}, end_fr_id {}".format(pos_id, command_id, start_fr_id, end_fr_id))

    mic_raw_audio_filepath  = data_path+'video_audio/{}_{}_2_{}_{}.wav'.format(sub_id, trial_id, pos_id, mic_id) 
    sample_rate, data = scipy.io.wavfile.read(mic_raw_audio_filepath)
    print("sample rate {}".format(sample_rate))
    print("audio_mic1_shape {}".format(data.shape))
    x_start = int(1/fps*sample_rate*start_fr_id)
    x_end = int(1/fps*sample_rate*end_fr_id)
    print("x_start {} x_end {}".format(x_start, x_end))
    new_filepath =  data_path+'mic{}_audio_cmd/{}_{}_2_{}_{}_{}.wav'.format(mic_id, sub_id, trial_id, pos_id, command_id,  mic_id)
    print(new_filepath)
    if(x_end > data.shape[0]):
        x_end = data.shape[0]
        print("x_end > data.shape[0]")
    scipy.io.wavfile.write(new_filepath, sample_rate, data[x_start:x_end]) 

def extract_data_by_sub_trial(dataset_path, commands_path,  sub_id, trial_id):
    #assuming every trial has mic1_audio_cmd_trim folder
    print("[INFO] extract video and audio from raw data by commands for sub_id = {}, trial_id = {}".format(sub_id, trial_id))
    data_path = '{}sub_{}{}trial_{}{}'.format(
        dataset_path, sub_id, os.path.sep, trial_id, os.path.sep)
    
    #audio_trim_filepaths = glob.glob(data_path +'mic1_audio_cmd_trim'+os.path.sep+'*.wav')
    make_dir(data_path+'thr_video_cmd/')
    make_dir(data_path+'rgb_video_cmd/')
    make_dir(data_path+'mic1_audio_cmd/')
    make_dir(data_path+'mic2_audio_cmd/')
    commands_filepath = commands_path +  'commands_sub{}_trial{}.csv'.format(sub_id, trial_id)

    #retreiving the raw data    
    commands = pd.read_csv(commands_filepath)
    print(commands.columns)
    for i in range(len(commands)):
        command = commands.iloc[i]
        #extract_audio_by_command(command, data_path,sub_id, trial_id, 1)
       
        #thr_raw_video_filepath = data_path+'video_audio/{}_{}_2_{}_1.avi'.format(sub_id, trial_id, command.pos_id)
        rgb_raw_video_filepath = data_path+'video_audio/{}_{}_2_{}_2.avi'.format(sub_id, trial_id, command.pos_id)
        cap =  cv2.VideoCapture(rgb_raw_video_filepath)
        # using cap.set start at the start_fr_id and then do it for a number frames needed
        print(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        rgb_video_out_filepath = data_path+'video_audio/{}_{}_2_{}_{}_2.avi'.format(sub_id, trial_id, command.pos_id, command.command_id)
        rgb_video_out = cv2.VideoWriter(rgb_video_out_filepath, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        frame_id = 1
        while(cap.isOpened() and frame_id <= (end_fr_id - start_fr_id)):
            ret, frame = cap.read()
            if ret == False:
                break
            rgb_video_out.write(frame)
            frame_id += 1

        cap.release()
        
    '''     
    for audio_trim_filepath in audio_trim_filepaths:
        
        audio_trim_filename = audio_trim_filepath.split(os.path.sep)[-1]
        print('[INFO] reading already trimmed audio file: '+audio_trim_filename)
        duration_trim = audio_trim.shape[0] / sample_rate_trim
        print('[INFO] duration of the already trimmed file = {}'.format(duration_trim))
        
        extract_frame(audio_trim_filename, duration_trim, data_path, 1)
        extract_frame(audio_trim_filename, duration_trim, data_path, 2)   
    '''
def extract_data_by_range(dataset_path, commands_path, sub_id_str, sub_id_end):
    print('[INFO] extract frames from videos by commands for the range of sub_id [{} ... {}]'.format(sub_id_str, sub_id_end))
    for sub_id in range(sub_id_str, sub_id_end):
        for trial_id in [1,2]:
            extract_databy_sub_trial(dataset_path, commands_path, sub_id, trial_id)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
        help="path to dataset")
ap.add_argument("-c", "--commands", required=True,
        help="path to commands")
ap.add_argument("-i", "--sub_info",  nargs='+', type=int,
        default = (0,0),
        help="subject info: ID, trial #")
ap.add_argument("-r", "--sub_range",  nargs='+', type=int,
        default = (0,0))
args = vars(ap.parse_args())

sub_id_in = args["sub_info"][0]
trial_id_in = args["sub_info"][1]
dataset_path = args["dataset"]
sub_id_str = args["sub_range"][0]
sub_id_end = args["sub_range"][1]
commands_path = args["commands"]

if (sub_id_str!=0 and sub_id_end!=0):
    extract_data_by_range(dataset_path, commands_path, sub_id_str, sub_id_end)
elif (sub_id_in!=0 and trial_id_in!=0):
    extract_data_by_sub_trial(dataset_path, commands_path, sub_id_in, trial_id_in)
