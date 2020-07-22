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
alignment_filepath = 'metadata/session_2/{}.csv'.format(set_name)
alignment_data = pd.read_csv(alignment_filepath)

sub_id_str = args["sub_range"][0] - 1
sub_id_end = args["sub_range"][1] - 1

ap_temp = argparse.ArgumentParser()
ap_temp.add_argument("-p")
ap_temp.add_argument("-i", nargs='+')
ap_temp.add_argument("-y", nargs='+')
ap_temp.add_argument("-x", nargs='+')
ap_temp.add_argument("-s", type=int)
ap_temp.add_argument("-l", nargs='+')
ap_temp.add_argument("-f", type=str)
ap_temp.add_argument("-r", nargs='+')

print(alignment_data.shape)
for row_idx in range(sub_id_str*2, sub_id_end*2): 
    sequences = alignment_data.iloc[row_idx,2]
    sequences = sequences.split("\n")
    sequences =  filter(None, sequences)
    for sequence in sequences:

        print(sequence)
        params_str = sequence[sequence.find('-p'):]
        params = ap_temp.parse_args(params_str.split())
        #print(params)
        i = ' '.join(params.i)
        y =  ' '.join(params.y)
        x =  ' '.join(params.x)
        l =  ' '.join(params.l)
        if params.f == "none":
            r =  ' '.join(params.r)
        command = "/home/madina_abdrakhmanova/miniconda3/bin/python align_crop_image.py -p {} -s 0 -i {} -y {} -x {} -l {} ".format(dataset_path, i, y,x,l)
        if params.f != None:
            command = command + "-f {}".format(params.f)
           # print(command)
        if params.f == "none":
            command = command + " -r {}".format(' '.join(params.r))
        print(command)
        os.system(command)

