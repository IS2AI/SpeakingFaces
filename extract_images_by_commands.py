import os
import glob
import scipy.io.wavfile
import argparse
import cv2

def mk_dir_to_extract(dataset_path, set_name, sub_id, trial_id, streamID):
    opt = 'thr' if streamID == 1 else 'rgb'
    #:new_dir = '{}sub_{}/trial_{}/{}_image_cmd'.format(dataset_path, sub_id, trial_id, opt)
    new_dir = '{}Drive/thermal_db/{}_data/sub_{}/trial_{}/{}_image_cmd'.format(dataset_path, set_name, sub_id, trial_id, opt)
    new_dir_exists = False
    
    if not os.path.exists(new_dir):
        os.system('sudo mkdir {}'.format(new_dir))
        print('sudo mkdir '+ new_dir)
    else:
        new_dir_exists = True
        print('the directory {} already exists'.format(new_dir))
    return new_dir, new_dir_exists

def extract_frame(audio_trim_path, streamID):
    opt = 'thr' if streamID == 1 else 'rgb'  

    audio_trim_path_split = audio_trim_path.split('/')
    audio_trim_filename = audio_trim_path_split[-1]
    print('reading already trimmed audio file: '+audio_trim_filename)
    sample_rate_trim, audio_trim = scipy.io.wavfile.read(audio_trim_path)
    duration_trim = audio_trim.shape[0] / sample_rate_trim
    print('duration of the already trimmed file = {}'.format(duration_trim))
        
    audio_trim_filename_split = audio_trim_filename.split('_')
    video_to_extract_filename  = '_'.join(audio_trim_filename_split[:-1])+'_{}.avi'.format(streamID)
    video_to_extract_filepath = '/'.join(audio_trim_path_split[:-2])+'/'+opt+'_video_cmd/'+video_to_extract_filename
    print('accessing a video file:'+ video_to_extract_filename)
    cap =  cv2.VideoCapture(video_to_extract_filepath)
    
    extracted_images_dir = '/'.join(audio_trim_path_split[:-2])+'/'+opt+'_image_cmd/'
    fps = 28
    max_num_frames = int(duration_trim*28)
    frameID=1
    while(cap.isOpened() and frameID <= max_num_frames):
        ret, frame = cap.read()
        if ret == False:
            break
        extracted_image_filename = '_'.join(audio_trim_filename_split[:-1])+'_{}_{}.png'.format(frameID, streamID)
        extracted_image_filepath = extracted_images_dir+extracted_image_filename
        if not os.path.exists(extracted_image_filepath):
            if frameID%10 == 0:
                print('saving an extracted frame: '+ extracted_image_filename)
            cv2.imwrite(extracted_image_filepath,frame)
        frameID+=1
    
    cap.release()
    cv2.destroyAllWindows()
    print('finished extracting frames')

def extract_frames_by_sub_trial(dataset_path, set_name, sub_id, trial_id):
    #assuming every trial has mic1_audio_cmd_trim folder
    print("extract frames from videos by commands for sub_id = {}, trial_id = {}".format(sub_id, trial_id))
    audio_trim_paths = glob.glob('{}Drive/thermal_db/{}_data/sub_{}/trial_{}/mic1_audio_cmd_trim/*.wav'.format(dataset_path, set_name, sub_id, trial_id))
    #audio_trim_paths = glob.glob('{}sub_{}/trial_{}/mic1_audio_cmd_trim/*.wav'.format(dataset_path, sub_id, trial_id))
    thr_image_cmd_path, thr_image_cmd_exists = mk_dir_to_extract(dataset_path, set_name, sub_id, trial_id, 1)
    rgb_image_cmd_path, rgb_image_cmd_exists = mk_dir_to_extract(dataset_path, set_name, sub_id, trial_id, 2)
    
    for audio_trim_path in audio_trim_paths:
        extract_frame(audio_trim_path, 1)
        extract_frame(audio_trim_path, 2)    

def extract_frames_by_set(dataset_path, set_name):
    # get the csv files
    # fix these paths for my mac machine
    csv_paths = glob.glob('{}Drive/thermal_db/csvs/{}_set/fpc5/*1.csv'.format(dataset_path, set_name))
    csv_paths = csv_paths+ glob.glob('{}Drive/thermal_db/csvs/{}_set/fpc5/*2.csv'.format(dataset_path, set_name))
    for csv_path in csv_paths:
        #for each path get the subjectID and trialID
        csv_path_split = csv_path.split('/')
        csv_filename = csv_path_split[-1]
        csv_filename_split = csv_filename.split('_')
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
