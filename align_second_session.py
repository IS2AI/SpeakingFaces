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
alignment_filepath = 'metadata/{}.csv'.format(set_name)
alignment_data = pd.read_csv(alignment_filepath)

sub_id_str = args["sub_range"][0] - 1
sub_id_end = args["sub_range"][1] - 1


ap = argparse.ArgumentParser()
ap.add_argument("-p",  required=True,
    help="path to the dataset")
ap.add_argument("-i",  nargs='+', type=int,
    help="subID(1,...,142) trialID (1,2) posID(1,...,9)")
ap.add_argument("-y", "--dy",  nargs='+', type=int,
    help="a list of shifts in y axis for each position")
ap.add_argument("-x", "--dx",  nargs='+', type=int,
    help="a list of shifts in x axis for each position")
ap.add_argument("-s", "--show", type=int, default=0,
    help="visualize or not a preliminary result of alignment")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
    help="the minimum probability to filter out weak face detections")
ap.add_argument("-l", "--landmark", nargs='+', type=int,
    help="a list of landmark ids that should server as the upper bound to crop for each position")
ap.add_argument("-z", "--size",  nargs='+', type=int, default = (128,64), 
    help="W H")
ap.add_argument("-f", "--face_detector", type=str, default="hog",
    help="model for face detection: hog/cnn/dnn/none")
ap.add_argument("-r", "--roi",  nargs='+', type=int,
    help="a list of coordinates to bound the RoI: startX, startY, endX, endY")


for i in range(sub_id_str*2, sub_id_end*2): 
    commands = df.iloc[i,2]
	commands = commands.split("\n")
	for command in commands:
		print(command)
		params_str = command[command.find('-p'):]
		params = ap.parse_args(params_str.split())
		print(params)
		'''
        command = "/home/madina_abdrakhmanova/miniconda3/bin/python align_crop_image.py -p /mnt/sharefolder/Drive/thermal_db/train_data/ "+p
    	print(command)
    	os.system(command)
		'''
