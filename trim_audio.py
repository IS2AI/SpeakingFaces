import os
import glob
import scipy.io.wavfile
import argparse
from speakingfacespy.imtools import make_dir

def trim_audio(audio_trim_path, length_audio_trim,to_trim_mic_id, to_trim_new_dir):
    #trim the corresponding other mic audio
    audio_trim_filename = audio_trim_path.split(os.path.sep)[-1] 
    audio_trim_filename_split = audio_trim_filename.split('_') 
    audio_to_trim_filename  = '_'.join(audio_trim_filename_split[:-1])+'_{}.wav'.format(to_trim_mic_id)
    audio_to_trim_new_path = to_trim_new_dir+os.path.sep+audio_to_trim_filename
    
    if not os.path.exists(audio_to_trim_new_path):
        print('[INFO] reading an audio file to trim: '+audio_to_trim_filename)
        sample_rate_to_trim, audio_to_trim = scipy.io.wavfile.read(to_trim_new_dir[:-5]+os.path.sep+audio_to_trim_filename)
        duration_to_trim = audio_to_trim.shape[0] / sample_rate_to_trim
        print('[INFO] duration of the original file = {}'.format(duration_to_trim))
        print('[INFO] writing trimmed audio to {}'.format(audio_to_trim_new_path))
        scipy.io.wavfile.write(audio_to_trim_new_path, sample_rate_to_trim, audio_to_trim[:length_audio_trim])
        print('[INFO] done trimming')
    else:
        print('[INFO] the file {} already exists'.format(audio_to_trim_new_path))
        

def trim_audio_by_sub_trial(dataset_path, set_name, sub_id, trial_id):
    print("[INFO] trim all audio files given sub_id = {}, trial_id = {}".format(sub_id, trial_id))
    audio_trim_paths = glob.glob('{}{}_data{}sub_{}{}trial_{}{}*_trim{}*.wav'.format(
        dataset_path, set_name, os.path.sep, sub_id, os.path.sep, trial_id, os.path.sep, os.path.sep))        
    is_mic1_audio_trim = (audio_trim_paths[0].split(os.path.sep)[-2].find('1')!=-1)
    to_trim_mic_id = 2 if is_mic1_audio_trim else 1
    to_trim_dir = '{}{}_data{}sub_{}{}trial_{}{}mic{}_audio_cmd_trim'.format(
        dataset_path, set_name, os.path.sep, sub_id, os.path.sep, trial_id, os.path.sep, to_trim_mic_id)
    if not os.path.exists(to_trim_dir):
        make_dir(to_trim_dir)
        for audio_trim_path in audio_trim_paths:
            print('[INFO] reading already trimmed audio file: '+audio_trim_path)
            sample_rate_trim, audio_trim = scipy.io.wavfile.read(audio_trim_path)
            length_audio_trim = audio_trim.shape[0]
            duration_trim = audio_trim.shape[0] / sample_rate_trim
            print('[INFO] duration of the already trimmed file = {}'.format(duration_trim))
            #trim the corresponding other mic audio
            trim_audio(audio_trim_path, length_audio_trim, to_trim_mic_id, to_trim_dir)


def trim_audio_by_set(dataset_path, set_name):
    csv_paths = glob.glob('{}csvs{}{}_set{}fpc5{}*1.csv'.format(dataset_path,os.path.sep,set_name,os.path.sep,os.path.sep))
    csv_paths = csv_paths+ glob.glob('{}csvs{}{}_set{}fpc5{}*2.csv'.format(dataset_path,os.path.sep,set_name,os.path.sep,os.path.sep))
    for csv_path in csv_paths:
        csv_filename_split = csv_path.split(os.path.sep)[-1].split('_')
        trial_id = int(csv_filename_split[-1][-5])
        sub_id = int(csv_filename_split[1][3:])
        trim_audio_by_sub_trial(dataset_path, set_name, sub_id, trial_id)    


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
    trim_audio_by_sub_trial(dataset_path, set_name, sub_id_in, trial_id_in)
else:
    trim_audio_by_set(dataset_path, set_name)
