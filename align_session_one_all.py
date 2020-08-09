# import the necessary packages
import pandas as pd
import argparse
import os

# create argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
        help="path to dataset")
ap.add_argument("-s", "--split", required = True,
	help="name the split train/valid/test")
ap.add_argument("-i", "--sub_range",  nargs='+', type=int, default = (0,0), 
    help="range of subjects")
args = vars(ap.parse_args())

# initialize the set name 
set_name = args["split"]

# construct a path to the dataset
dataset_path = os.path.join(args["dataset"], set_name)

# construct the filename based on the set name 
alignment_filepath = "metadata/session_1/alignment_info_{}.csv".format(set_name)

# read the alignment info to the Pandas table
alignment_data = pd.read_csv(alignment_filepath)

# initialize the starting and ending subjects IDs
sub_id_str = args["sub_range"][0] - 1
sub_id_end = args["sub_range"][1] - 1

# loop over the subjects
for i in range(sub_id_str * 2, sub_id_end * 2):
    # extract shifts for x and y axis
    # for the given subject 
    dx_str = alignment_data.iloc[i].dx
    dy_str = alignment_data.iloc[i].dy

    print(dy_str, dx_str)

    # can be anything lets set the pos_id to 1
    sub_info = "{} {} 1".format(alignment_data.iloc[i].sub_id, alignment_data.iloc[i].trial_id)
    final_command = 'python align_session_one.py --dataset {} --dy {} --dx {} --sub_info {}'.format(dataset_path, dy_str, dx_str, sub_info)

    print(final_command)
    os.system(final_command)
