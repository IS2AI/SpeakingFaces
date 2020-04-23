# import the necessary packages
from imutils import paths
from speakingfacespy.imtools import path_to_thermal_image
from speakingfacespy.imtools import make_dir
import imutils
import numpy as np
import pandas as pd
import cv2 
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to dataset")
ap.add_argument("-n", "--frame", type=int, default=1,
	help="process every n'th frame")
ap.add_argument("-i", "--sub_info",  nargs='+', type=int,
	help="subject info: ID, trial #")
ap.add_argument("-y", "--dy",  nargs='+', type=int,
	help="dy shifts based on positions")
ap.add_argument("-x", "--dx",  nargs='+', type=int,
	help="x shifts based on positions")
ap.add_argument("-s", "--show", type=int, default=0,
	help="visualize ext racted faces")
ap.add_argument("-e", "--session", type =int, default = 1,
        help="session_id, 1 for nonspeaking, 2 for speaking")
args = vars(ap.parse_args())

# load matched features from xlsx file and convert it numpy array 
df = pd.read_excel (r'calibration'+os.path.sep+'matched_features.xlsx')
M = df.to_numpy()

# set the option in accordance with the session_id
opt = "_cmd" if args["session"] == 2 else ""

# grab the path to the visual images in our dataset
dataset_path = "{}sub_{}".format(args["dataset"], args["sub_info"][0])+os.path.sep+"trial_{}".format(args["sub_info"][1])  
rgb_image_filepaths = list(paths.list_images(dataset_path + os.path.sep+"rgb_image{}".format(opt)))

# create a directory to save images
rgb_image_aligned_path = dataset_path + os.path.sep+"rgb_image{}_aligned".format(opt)+os.path.sep
make_dir(rgb_image_aligned_path)

# loop over images in the folders
for rgb_image_filepath in rgb_image_filepaths:

    # extract the current image info
    if args["session"] == 1:
        sub_id, trial_id, session_id, pos_id, frame_id = rgb_image_filepath.split(os.path.sep)[-1].split("_")[-6:-1]
        command_id = -1
        rgb_image_aligned_filepath = "{}{}_{}_{}_{}_{}_3.png".format(rgb_image_aligned_path, sub_id, trial_id, session_id, pos_id, frame_id)
    else:
        sub_id, trial_id, session_id, pos_id, command_id, frame_id = rgb_image_filepath.split(os.path.sep)[-1].split("_")[-7:-1]
        rgb_image_aligned_filepath = "{}{}_{}_{}_{}_{}_{}_3.png".format(rgb_image_aligned_path, sub_id, trial_id, session_id, pos_id, command_id, frame_id)
    #print(rgb_image_aligned_filepath)
    #print(rgb_image_filepath)
    # given by the argument only if "show" mode
    # is enabled

    if args["sub_info"][2] != int(pos_id) and args["show"]:
        cv2.destroyAllWindows()
        continue

    # initialize lists of shifts
    dy = args["dy"][int(pos_id) - 1]
    dx = args["dx"][int(pos_id) - 1]

    ptsA = np.array([[399 + dx, 345 + dy], [423 + dx, 293 + dy], [293 + dx, 316 + dy], [269 + dx, 368 + dy]])
    ptsB = np.array([[249, 237], [267, 196], [169, 214], [151, 254]])
    
    # estimate a homography matrix to warp the visible image
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 2.0)

    # process only n'th frames  
    if int(frame_id) % args["frame"] == 0:
        print("[INFO] processing image {}".format(rgb_image_filepath.split(os.path.sep)[-1]))

        # construct the thermal image path using the rgb image path
        thr_image_path = path_to_thermal_image(rgb_image_filepath, dataset_path, os.path.sep+"thr_image{}".format(opt)+os.path.sep)
        # load rgb and corresponding thermal image 
        rgb = cv2.imread(rgb_image_filepath)
        thr = cv2.imread(thr_image_path)

        # grab a size of the thermal image 
        (H_thr, W_thr) = thr.shape[:2]
        
        # warp the rgb image
        # to align with the thermal image
        rgb = cv2.warpPerspective(rgb, H, (W_thr, H_thr), 
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        if args["show"]:
            # make a copy of the rgb image
            # then replace its RED channel with 
            # the RED channel of the thermal image
            rgb_copy = rgb.copy()
            rgb_copy[:, :, 2] = thr[:, :, 2]

            # show the images
            cv2.imshow("Sub:{} Trial:{} Session:{} Pos:{} Command:{} Frame:{}".format(sub_id, trial_id, session_id, pos_id, command_id, frame_id), np.hstack([rgb, thr, rgb_copy]))
            key = cv2.waitKey(0) & 0xFF

            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                    break
    
        cv2.imwrite(rgb_image_aligned_filepath, rgb)
