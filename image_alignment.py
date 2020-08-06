# import the necessary packages
from speakingfacespy.imtools import make_dir
import numpy as np
import pandas as pd
import cv2 
import argparse
import os

# create the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the SpeakingFaces dataset")
ap.add_argument("-i", "--sub_info",  nargs='+', type=int,
	help="subID(1,...,142) trialID (1,2) posID(1,...,9)")
ap.add_argument("-y", "--dy",  nargs='+', type=int,
	help="a list of shifts in y axis for each position")
ap.add_argument("-x", "--dx",  nargs='+', type=int,
	help="a list of shifts in x axis for each position")
ap.add_argument("-s", "--show", type=int, default=0,
	help="visualize (1) or not (0) a preliminary result of alignment")
args = vars(ap.parse_args())

# load the matched features from the .xlsx file to a Pandas table
# then convert it to a NumPy array 
df = pd.read_excel (r'calibration/matched_features.xlsx')
M = df.to_numpy()

# initialize subject ID and trial ID
sub_id = args["sub_info"][0]
trial_id = args["sub_info"][1]

# initialize number of positions per trial
# and number of frames per position
pos_ids = 9
frame_ids = 900

# initialize a path to the dataset
path_to_dataset = args["dataset"]

# construct a path to the unaligned rgb images 
rgb_image_path = os.path.join(path_to_dataset, "sub_{}/trial_{}/rgb_image".format(sub_id, trial_id)) 

# construct a path to the thermal images 
thr_image_path = os.path.join(path_to_dataset, "sub_{}/trial_{}/thr_image".format(sub_id, trial_id))   

# create a directory to save the aligned rgb images
rgb_image_aligned_path = os.path.join(path_to_dataset, "sub_{}/trial_{}/rgb_image_aligned".format(sub_id, trial_id)) 
make_dir(rgb_image_aligned_path)

# loop over the positions 1...9
for pos_id in range(1, pos_ids + 1):
    # consider a position that we are going to
    # align only if the "show" mode is enabled
    if args["sub_info"][2] != pos_id and args["show"]:
        continue

    # grab dy and dx shifts for the given position
    dy = args["dy"][pos_id - 1]
    dx = args["dx"][pos_id - 1]

    # construct arrays of matched features
    # for the given position
    ptsA = np.array([[399 + dx, 345 + dy], [423 + dx, 293 + dy], [293 + dx, 316 + dy], [269 + dx, 368 + dy]])
    ptsB = np.array([[249, 237], [267, 196], [169, 214], [151, 254]])
    
    # estimate a homography matrix
    # for the given position 
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 2.0)

    # loop over the frames 1...900
    for frame_id in range(1, frame_ids + 1):
        print("[INFO] processing sub_{}, trial_{}, pos_{}, frame_{}".format(sub_id, trial_id, pos_id, frame_id))

        # construct filenames for rgb and corresponding thermal images  
        rgb_file = "{}_{}_1_{}_{}_{}.png".format(sub_id, trial_id, pos_id, frame_id, 2)
        thr_file = "{}_{}_1_{}_{}_{}.png".format(sub_id, trial_id, pos_id, frame_id, 1)

        # construct the final paths to grab rgb and thermal images
        rgb_file = os.path.join(rgb_image_path, rgb_file)
        thr_file = os.path.join(thr_image_path, thr_file)
        #print(rgb_file)
        #print(thr_file)

        # load rgb and thermal images
        rgb_image = cv2.imread(rgb_file)        
        thr_image = cv2.imread(thr_file)
    
        # grab height and width of the thermal image 
        (h_thr, w_thr) = thr_image.shape[:2]
        
        # align rgb image with the thermal one
        rgb_aligned = cv2.warpPerspective(rgb_image, H, (w_thr, h_thr), 
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # show results of the alignment 
        if args["show"]:
            # make a copy of the rgb image and replace its RED channel 
            # with the RED channel of the thermal image
            rgb_copy = rgb_aligned.copy()
            rgb_copy[:, :, 2] = thr_image[:, :, 2]

            # show images
            cv2.imshow("Output".format(
                sub_id, trial_id, pos_id, frame_id), np.hstack([rgb_aligned, thr_image, rgb_copy]))
            key = cv2.waitKey(0) & 0xFF

            # if the 'q' key is pressed, stop the loop, 
            # exit from the program
            if key == ord("q"):
                    break
        else:
            # construct a path to save the aligned rgb image
            rgb_aligned_file = "{}_{}_1_{}_{}_{}.png".format(sub_id, trial_id, pos_id, frame_id, 3)
            rgb_aligned_file = os.path.join(rgb_image_aligned_path, rgb_aligned_file)
            cv2.imwrite(rgb_aligned_file, rgb_aligned)
