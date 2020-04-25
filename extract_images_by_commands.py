import os
import glob
import argparse
import cv2
from speakingfacespy.imtools import make_dir

def extract_frame(audio_trim_filepath, stream_id):
    print('[INFO] reading already trimmed audio file: '+audio_trim_filepath)
    sample_rate_trim, audio_trim = scipy.io.wavfile.read(audio_trim_filepath)
    duration_trim = audio_trim.shape[0] / sample_rate_trim
    print('[INFO] duration of the already trimmed file = {}'.format(duration_trim))
      
    audio_trim_filepath_split = audio_trim_filepath.split('/')
    audio_trim_filename_split = audio_trim_filepath_split[-1].split('_')
    video_to_extract_filename  = '_'.join(audio_trim_filename_split[:-1])+'_{}.avi'.format(stream_id)
    
    opt = 'thr' if stream_id == 1 else 'rgb'  
    video_to_extract_filepath = '/'.join(audio_trim_filepath_split[:-2])+'/'+opt+'_video_cmd/'+video_to_extract_filename
    print('[INFO] accessing a video file:'+ video_to_extract_filepath)
    cap =  cv2.VideoCapture(video_to_extract_filepath)
    
    extracted_images_dir = '/'.join(audio_trim_filepath_split[:-2])+'/'+opt+'_image_cmd/'
    fps = 28
    max_num_frames = int(duration_trim*28)
    frameID=1
    while(cap.isOpened() and frameID <= max_num_frames):
        ret, frame = cap.read()
        if ret == False:
            break
        extracted_image_filename = '_'.join(audio_trim_filename_split[:-1])+'_{}_{}.png'.format(frameID, stream_id)
        extracted_image_filepath = extracted_images_dir+extracted_image_filename
        if not os.path.exists(extracted_image_filepath):
            if frameID%10 == 0:
                print('[INFO] saving an extracted frame: '+ extracted_image_filepath)
            cv2.imwrite(extracted_image_filepath, frame)
        frameID+=1
    
    cap.release()
    cv2.destroyAllWindows()
    print('[INFO] finished extracting frames')


def extract_frame_new(audio_trim_filepath, dataset_path, set_name, sub_id, trial_id, stream_id):
    print('[INFO] reading already trimmed audio file: '+audio_trim_filepath)
    sample_rate_trim, audio_trim = scipy.io.wavfile.read(audio_trim_filepath)
    duration_trim = audio_trim.shape[0] / sample_rate_trim
    print('[INFO] duration of the already trimmed file = {}'.format(duration_trim))
      
    audio_trim_filepath_split = audio_trim_filepath.split('/')
    audio_trim_filename_split = audio_trim_filepath_split[-1].split('_')
    video_to_extract_filename  = '_'.join(audio_trim_filename_split[:-1])+'_{}.avi'.format(stream_id)
    
    opt = 'thr' if stream_id == 1 else 'rgb'  
    video_to_extract_filepath = '/'.join(audio_trim_filepath_split[:-2])+'/'+opt+'_video_cmd/'+video_to_extract_filename
    print('[INFO] accessing a video file:'+ video_to_extract_filepath)
    cap =  cv2.VideoCapture(video_to_extract_filepath)
    
    extracted_images_dir = '/'.join(audio_trim_filepath_split[:-2])+'/'+opt+'_image_cmd/'
    fps = 28
    max_num_frames = int(duration_trim*28)
    frameID=1
    while(cap.isOpened() and frameID <= max_num_frames):
        ret, frame = cap.read()
        if ret == False:
            break
        extracted_image_filename = '_'.join(audio_trim_filename_split[:-1])+'_{}_{}.png'.format(frameID, stream_id)
        extracted_image_filepath = extracted_images_dir+extracted_image_filename
        if not os.path.exists(extracted_image_filepath):
            if frameID%10 == 0:
                print('[INFO] saving an extracted frame: '+ extracted_image_filepath)
            cv2.imwrite(extracted_image_filepath, frame)
        frameID+=1
    
    cap.release()
    cv2.destroyAllWindows()
    print('[INFO] finished extracting frames')

def extract_frames_by_sub_trial(dataset_path, set_name, sub_id, trial_id):
    #assuming every trial has mic1_audio_cmd_trim folder
    print("[INFO] extract frames from videos by commands for sub_id = {}, trial_id = {}".format(sub_id, trial_id))
    audio_trim_filepaths = glob.glob('{}{}_data/sub_{}/trial_{}/mic1_audio_cmd_trim/*.wav'.format(dataset_path, set_name, sub_id, trial_id))
    make_dir('{}{}_data/sub_{}/trial_{}/thr_image_cmd'.format(dataset_path, set_name, sub_id, trial_id))
    make_dir('{}{}_data/sub_{}/trial_{}/rgb_image_cmd'.format(dataset_path, set_name, sub_id, trial_id))

    for audio_trim_filepath in audio_trim_filepaths:
        extract_frame(audio_trim_filepath, 1)
        extract_frame(audio_trim_filepath, 2)    

def extract_frames_by_set(dataset_path, set_name):
    csv_paths = glob.glob('{}csvs/{}_set/fpc5/*1.csv'.format(dataset_path, set_name))
    csv_paths = csv_paths+ glob.glob('{}csvs/{}_set/fpc5/*2.csv'.format(dataset_path, set_name))
    for csv_path in csv_paths:
        csv_filename_split = csv_path.split('/')[-1].split('_')
        trial_id = int(csv_filename_split[-1][-5])
        sub_id = int(csv_filename_split[1][3:])
        extract_frames_by_sub_trial(dataset_path, set_name, sub_id, trial_id)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
        help="path to dataset")
ap.add_argument("-s", "--split", required = True,
        help="name the split train/valid/test")
ap.add_argument("-i", "--sub_info",  nargs='+', type=int,
        default = (0,0),
        help="subject info: ID, trial #")
args = vars(ap.parse_args())

sub_id_in = args["sub_info"][0]
trial_id_in = args["sub_info"][1]
set_name = args["split"]
dataset_path = args["dataset"]
print(dataset_path)
print(set_name)
print(sub_id_in, trial_id_in)
if (sub_id_in!=0 and trial_id_in!=0):
    extract_frames_by_sub_trial(dataset_path, set_name, sub_id_in, trial_id_in)
else:
    extract_frames_by_set(dataset_path, set_name)
