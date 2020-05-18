import os
import glob
import scipy.io.wavfile
import argparse
from speakingfacespy.imtools import make_dir

def trim_audio(audio_trim_filename, length_audio_trim,to_trim_mic_id, to_trim_new_dir):
    #trim the corresponding other mic audio
    audio_to_trim_filename  = '_'.join(audio_trim_filename.split('_')[:-1])+'_{}.wav'.format(to_trim_mic_id)
    audio_to_trim_new_filepath = to_trim_new_dir+os.path.sep+audio_to_trim_filename
    
    if not os.path.exists(audio_to_trim_new_filepath):
        print('[INFO] reading an audio file to trim: '+audio_to_trim_filename)
        sample_rate_to_trim, audio_to_trim = scipy.io.wavfile.read(to_trim_new_dir[:-5]+os.path.sep+audio_to_trim_filename)
        duration_to_trim = audio_to_trim.shape[0] / sample_rate_to_trim
        print('[INFO] duration of the original file = {}'.format(duration_to_trim))
        print('[INFO] writing trimmed audio to {}'.format(audio_to_trim_new_filepath))
        scipy.io.wavfile.write(audio_to_trim_new_filepath, sample_rate_to_trim, audio_to_trim[:length_audio_trim])
        print('[INFO] done trimming')
    else:
        print('[INFO] the file {} already exists'.format(audio_to_trim_new_filepath))
        

def trim_audio_by_sub_trial(dataset_path, sub_id, trial_id):
    print("[INFO] trim all audio files given sub_id = {}, trial_id = {}".format(sub_id, trial_id))
    data_path = '{}sub_{}{}trial_{}{}'.format(
        dataset_path, sub_id, os.path.sep, trial_id, os.path.sep)

    audio_trim_filepaths = glob.glob(data_path +'*_trim'+os.path.sep+'*.wav')
     
    is_mic1_audio_trim = (audio_trim_filepaths[0].split(os.path.sep)[-2].find('1')!=-1)
    to_trim_mic_id = 2 if is_mic1_audio_trim else 1
    to_trim_dir = data_path + 'mic{}_audio_cmd_trim'.format(to_trim_mic_id)
    
    if not os.path.exists(to_trim_dir):
        make_dir(to_trim_dir)
        for audio_trim_filepath in audio_trim_filepaths:
            audio_trim_filename = audio_trim_filepath.split(os.path.sep)[-1]
            print('[INFO] reading already trimmed audio file: '+audio_trim_filename)
            sample_rate_trim, audio_trim = scipy.io.wavfile.read(audio_trim_filepath)
            duration_trim = audio_trim.shape[0] / sample_rate_trim
            print('[INFO] duration of the already trimmed file = {}'.format(duration_trim))
            #trim the corresponding other mic audio
            trim_audio(audio_trim_filename, audio_trim.shape[0], to_trim_mic_id, to_trim_dir)

def trim_audio_by_range(dataset_path, sub_id_str, sub_id_end):
    print('[INFO] trim all audio files in the range of sub_id [{} ... {}]'.format(sub_id_str, sub_id_end))
    for sub_id in range(sub_id_str, sub_id_end):
        for trial_id in [1,2]:
            trim_audio_by_sub_trial(dataset_path, sub_id, trial_id)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to dataset")
ap.add_argument("-i", "--sub_info",  nargs='+', type=int,
    default = (0,0), help="subject info: ID, trial #")
ap.add_argument("-r", "--sub_range",  nargs='+', type=int,
    default = (0,0))
args = vars(ap.parse_args())

sub_id_in = args["sub_info"][0]
trial_id_in = args["sub_info"][1]
dataset_path = args["dataset"] 
sub_id_str = args["sub_range"][0]
sub_id_end = args["sub_range"][1]

if (sub_id_str!=0 and sub_id_end!=0):
    trim_audio_by_range(dataset_path, sub_id_str, sub_id_end):
elif (sub_id_in!=0 and trial_id_in!=0):
    trim_audio_by_sub_trial(dataset_path, sub_id_in, trial_id_in)
