import pandas as pd
import os
split = 'train'
dataset_path = '/mnt/sdb/SpeakingFaces/{}_data/'.format(split)
alignment_filepath = '/home/test/Documents/Github/SpeakingFaces/metadata/alignment_info_{}.csv'.format(split)
alignment_data = pd.read_csv(alignment_filepath)

for i in range(40, 60):
    dx_str = alignment_data.iloc[i].dx
    dy_str = alignment_data.iloc[i].dy
    #can be anything lets set the pos_id to 1
    sub_info = "{} {} 1".format(alignment_data.iloc[i].sub_id, alignment_data.iloc[i].trial_id)
    final_command = 'python image_alignment.py --dataset {} --dy {} --dx {} --sub_info {} --show 0'.format(dataset_path, dy_str, dx_str, sub_info)
    print(final_command)
    os.system(final_command)
