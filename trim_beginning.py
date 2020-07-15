import os
import glob
from speakingfacespy.imtools import make_dir
import scipy.io.wavfile
import shutil
import argparse


def my_func(f):
  return int(f.split(os.path.sep)[-1].split("_")[-2])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
        help="path to data")
ap.add_argument("-s", "--set_name", required=True,
        help="path to commands")
args = vars(ap.parse_args())

path_to_data = args["path"]
set_name = args["set_name"]

dst_dir = os.path.join(path_to_data, "retrimmed_corrupted_files", set_name+"_data")
make_dir(dst_dir)

cor_path = os.path.join(path_to_data,"corrupted_files",set_name+"_data", "*.png")
cor_filepaths = glob.glob(cor_path)

img_dirs = ["thr_image_cmd", "rgb_image_cmd","rgb_image_cmd_aligned", "thr_roi_cmd", "rgb_roi_cmd"]
audio_dirs = ["mic1_audio_cmd_trim", "mic2_audio_cmd_trim"] 


for f in cor_filepaths:
    sub_id, trial_id, session_id, pos_id, command_id, frame_id = f.split(os.path.sep)[-1].split("_")[-7:-1]   
    
    for audio_dir_id in range(len(audio_dirs)):
        audio_filename = '_'.join([sub_id, trial_id, session_id, pos_id, command_id, "{}.wav".format(audio_dir_id+1)])
        audio_filepath = os.path.join(path_to_data,"{}_data".format(set_name), "sub_{}".format(sub_id), "trial_{}".format(trial_id), audio_dirs[audio_dir_id], audio_filename)
        audio_dst_dir = os.path.join(dst_dir, "sub_{}".format(sub_id), "trial_{}".format(trial_id), audio_dirs[audio_dir_id])
        make_dir(audio_dst_dir)
        if os.path.exists(audio_filepath):
            
            print("[INFO] extracting the audio file {}".format(audio_filepath))
            sample_rate, data = scipy.io.wavfile.read(audio_filepath)
            
            fps = 28.0
            x_start = int(1/fps*sample_rate*int(frame_id))-1
            new_filepath = os.path.join(audio_dst_dir, audio_filename)
            print("[INFO] writing the audio file {}".format(new_filepath))
            scipy.io.wavfile.write(new_filepath, sample_rate, data[x_start:])      

    for img_dir_id in range(len(img_dirs)):
        img_dirpath =  os.path.join(path_to_data, "{}_data".format(set_name), "sub_{}".format(sub_id), "trial_{}".format(trial_id), img_dirs[img_dir_id])
        img_dst_dir = os.path.join(dst_dir, "sub_{}".format(sub_id), "trial_{}".format(trial_id), img_dirs[img_dir_id])
        make_dir(img_dst_dir)
        if(os.path.exists(img_dirpath)):
            img_filepaths = glob.glob(os.path.join(img_dirpath, '_'.join([sub_id, trial_id, session_id, pos_id, command_id, "*.png"])))
            img_filepaths.sort(key = my_func)
            
            print("INFO: the number of files before the transfer: {}".format(len(img_filepaths)))
            
            for img_filepath in img_filepaths:
                img_filename = img_filepath.split(os.path.sep)[-1]
                curr_frame_id = img_filename.split("_")[-2]
                print(frame_id, curr_frame_id) 
                if int(curr_frame_id) > int(frame_id):
                    shutil.copy(img_filepath, img_dst_dir+os.path.sep)
                    
                    dst_filepath = os.path.join(img_dst_dir, img_filename)
                    new_dst_filename = '_'.join([sub_id, trial_id, session_id, pos_id, command_id, str(int(curr_frame_id)-int(frame_id)), img_filename.split("_")[-1]])
                    new_dst_filepath = os.path.join(img_dst_dir, new_dst_filename)
                    os.rename(dst_filepath, new_dst_filepath)
            
            onlyfiles = next(os.walk(img_dst_dir))[2] #dir is your directory path as string
            print("INFO: the number of files after the transfer: {}".format(len(onlyfiles)))

