import os
import glob
from speakingfacespy.imtools import make_dir
import scipy.io.wavfile
import shutil

path_to_data =  "/Users/madinaabdrakhmanova/Documents/data/"
cor_filepaths = glob.glob(path_to_data+"corrupted_files/*.png")
img_dirs = ["thr_image_cmd", "rgb_image_cmd","rgb_image_cmd_aligned", "thr_roi_cmd", "rgb_roi_cmd"]

for f in cor_filepaths:
    sub_id, trial_id, session_id, pos_id, command_id, frame_id = f.split(os.path.sep)[-1].split("_")[-7:-1]   
    if int(sub_id)<101:
        set_name = "train"
    elif int(sub_id)<121:
        set_name = "valid"
    else:
        set_name = "test"
    audio_filename = '_'.join([sub_id, trial_id, session_id, pos_id, command_id, "1"])
    audio_filepath = path_to_data + "{}_data/sub_{}/trial_{}/mic1_audio_cmd_trim/{}.wav".format(set_name, sub_id, trial_id, audio_filename)
    if os.path.exists(audio_filepath):
        
        dst_dir = path_to_data + "{}_data/sub_{}/trial_{}/retrimmed".format(set_name, sub_id, trial_id)
        make_dir(dst_dir)

        print("[INFO] extracting the audio file {}".format(audio_filepath))
        
        sample_rate, data = scipy.io.wavfile.read(audio_filepath)
        
        fps = 28.0
        x_start = int(1/fps*sample_rate*int(frame_id))-1
        print(frame_id, x_start)
        new_filepath = os.path.join(dst_dir, audio_filename)
        scipy.io.wavfile.write(new_filepath, sample_rate, data[x_start:])      
        '''
        for img_dir in img_dirs:
            if(os.path.exists(path_to_data + "{}_data/sub_{}/trial_{}/{}/".format(set_name, sub_id, trial_id, img_dir))):
                img_filepaths = glob.glob(path_to_data + "{}_data/sub_{}/trial_{}/{}/{}_{}_{}_{}_{}_*.png".format(
                    set_name, sub_id, trial_id, img_dir, sub_id, trial_id, session_id, pos_id, command_id))
                for filepath in img_filepaths:
                    filename = filepath.split(os.path.sep)[-1]
                    curr_frame_id = filename.split("_")[-2]

                    if int(curr_frame_id) > int(frame_id):
                        print(curr_frame_id)
                        shutil.copy(filepath, dst_dir)
                        dst_file = os.path.join(dst_dir, filename)
                        new_file_name = '_'.join([sub_id, trial_id, session_id, pos_id, command_id, str(int(curr_frame_id)-int(frame_id)), filename.split("_")[-1]])
                        print(new_file_name)
                        new_dst_file_name = os.path.join(dst_dir, new_file_name)
                        os.rename(dst_file, new_dst_file_name)
                    #copy and rename image frames 
        '''