gpu = '0'
random_seed = 10
predict = 'gender'     # 'age', 'gender'
# 1-rgb, 2-thermal, 3-audio, 4-rgb and thermal, 5-rgb and audio, 6-thermal and audio, 7-all
mode = 3

###############################################################################################
#dataset options
sub_path = 'data/subjects.csv'
train_path = 'datasets/SpeakingFaces/train_data'
valid_path = 'datasets/SpeakingFaces/valid_data'
test_path = 'datasets/SpeakingFaces/test_data'
train_noise = False
valid_noise = False
test_noise = False

###############################################################################################
#image file options
num_frames = 3

add_rgb_noise = False
rgb_noise = 'gauss'     # 'gauss', 'blur'
rnoise_value = 100      # 'gauss' -> STD, 'blur' -> kernel size

add_thr_noise = False
thr_noise = 'gauss'     # 'gauss', 
tnoise_value = 100      # 'gauss' -> STD

#audio file options
segment_len = 0.4       # seconds
sample_rate = 44100     # audio file sampling rate

add_audio_noise = False
audio_noise = 'gauss'   # 'gauss',
anoise_value = 5        # STD

###############################################################################################
#model options
drop = 0.1
max_epoch = 200
batch_size = 256
base_lr = 1e-1
warmup_lr = 1e-3
warmup_epochs = 0
is_clip = True
print_errors = True
clip = 10
weight_decay = 0e-0
patience = 20
num_workers = 0
save_prefix = f'models/mmnet'
data_shuffle = False

###############################################################################################
#LOAD PRETRAINED MODELS
#weights=f'models/mmnet_mode7_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed0_clip10_rfr3_tfr3_len0.4_bestEpoch61.torch'
