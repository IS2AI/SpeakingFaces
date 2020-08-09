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
alignment_filepath = "metadata/session_2/alignment_info_{}.csv".format(set_name)

# read the alignment info to the Pandas table
alignment_data = pd.read_csv(alignment_filepath)

# initialize the starting and ending subjects IDs
sub_id_str = args["sub_range"][0] - 1
sub_id_end = args["sub_range"][1] - 1

ap_temp = argparse.ArgumentParser()
ap_temp.add_argument("-d")
ap_temp.add_argument("-i", nargs='+')
ap_temp.add_argument("-y", nargs='+')
ap_temp.add_argument("-x", nargs='+')
ap_temp.add_argument("-m", type=str)
ap_temp.add_argument("-u", nargs='+')
ap_temp.add_argument("-s", type=int)

# loop over the subjects
for row_idx in range(sub_id_str * 2, sub_id_end * 2):
    # extract the sequence of arguments 
    sequences = alignment_data.iloc[row_idx,2]
    sequences = sequences.split("\n")
    sequences =  filter(None, sequences)

    # loop over the sequences
    for sequence in sequences:
        # parse the sequence
        params_str = sequence[sequence.find('-d'):]

        # parse the arguments
        params = ap_temp.parse_args(params_str.split())
        
        # initialize input arguments
        i = ' '.join(params.i)
        y = ' '.join(params.y)
        x = ' '.join(params.x)
        u = ' '.join(params.u)

        # initialize the command
        command = "python align_crop_session_two.py -d {} -s 1 -i {} -y {} -x {} -l {}".format(dataset_path, i, y, x, l)

        # run the command
        print(command)
        os.system(command)

