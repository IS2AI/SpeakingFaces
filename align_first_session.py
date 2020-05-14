import pandas as pd
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
        help="path to dataset")
ap.add_argument("-s", "--split", required = True,
	help="name the split train/valid/test")
ap.add_argument("-i", "--sub_range",  nargs='+', type=int,
        default = (0,0))

args = vars(ap.parse_args())

set_name = args["split"]
dataset_path = '{}{}_data/'.format(args["dataset"], set_name)
alignment_filepath = '/home/test/Documents/Github/SpeakingFaces/metadata/alignment_info_{}.csv'.format(set_name)
alignment_data = pd.read_csv(alignment_filepath)
sub_id_str = args["sub_range"][0] - 1
sub_id_end = args["sub_range"][1] - 1
print(sub_id_str*2, sub_id_end*2)

for i in range(sub_id_str*2, sub_id_end*2): 
    dx_str = alignment_data.iloc[i].dx
    dy_str = alignment_data.iloc[i].dy
    #can be anything lets set the pos_id to 1
    sub_info = "{} {} 1".format(alignment_data.iloc[i].sub_id, alignment_data.iloc[i].trial_id)
    final_command = 'python image_alignment.py --dataset {} --dy {} --dx {} --sub_info {} --show 0'.format(dataset_path, dy_str, dx_str, sub_info)

    print(final_command)
    os.system(final_command)
