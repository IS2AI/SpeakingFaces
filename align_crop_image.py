# import the necessary packages
from imutils import paths
from speakingfacespy.imtools import path_to_thermal_image
from speakingfacespy.imtools import make_dir
from speakingfacespy.imtools import lip_region_extractor
import imutils
import numpy as np
import pandas as pd
import cv2 
import argparse
import os

# parse the provided arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the dataset")
ap.add_argument("-n", "--frame", type=int, default=1,
	help="process every n'th frame")
ap.add_argument("-i", "--sub_info",  nargs='+', type=int,
	help="subID(1,...,142) trialID (1,2) posID(1,...,9)")
ap.add_argument("-y", "--dy",  nargs='+', type=int,
	help="a list of shifts in y axis for each position")
ap.add_argument("-x", "--dx",  nargs='+', type=int,
	help="a list of shifts in x axis for each position")
ap.add_argument("-s", "--show", type=int, default=0,
	help="visualize or not a preliminary result of alignment")
ap.add_argument("-e", "--session", type =int, default = 1,
        help="sessionID, 1 for nonspeaking, 2 for speaking")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
    help="minimum probability to filter out weak detections")
args = vars(ap.parse_args())

# load matched features from xlsx file and convert it numpy array 
df = pd.read_excel (r'calibration'+os.path.sep+'matched_features.xlsx')
M = df.to_numpy()

# set the option in accordance with the session_id
opt = "_cmd" if args["session"] == 2 else ""

# grab the path to the visual images in our dataset
sub_trial_dataset_path = "{}sub_{}".format(args["dataset"], args["sub_info"][0])+os.path.sep+"trial_{}".format(args["sub_info"][1])  
rgb_image_filepaths = list(paths.list_images(sub_trial_dataset_path + os.path.sep+"rgb_image{}".format(opt)))

# create a directory to save aligned images
rgb_image_aligned_path = sub_trial_dataset_path + os.path.sep+"rgb_image{}_aligned".format(opt)+os.path.sep
make_dir(rgb_image_aligned_path)

if args["session"] == 2:
    # load the serialized models from disk
    print("[INFO] loading the face and landmark predictors ...")
    face_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", 
        "models/res10_300x300_ssd_iter_140000.caffemodel")                      
    
    # create directories for extracted lip region
    lip_rgb_path = sub_trial_dataset_path + os.path.sep+"rgb_roi{}".format(opt)+os.path.sep
    make_dir(lip_rgb_path)
    lip_thr_path = sub_trial_dataset_path + os.path.sep+"thr_roi{}".format(opt)+os.path.sep
    make_dir(lip_thr_path)

# loop over original visible images in the folders
for rgb_image_filepath in rgb_image_filepaths:

    # extract the current image info
    if args["session"] == 1:
        sub_id, trial_id, session_id, pos_id, frame_id = rgb_image_filepath.split(os.path.sep)[-1].split("_")[-6:-1]
        command_id = -1
        rgb_image_aligned_filepath = "{}{}_{}_{}_{}_{}_3.png".format(rgb_image_aligned_path, sub_id, trial_id, session_id, pos_id, frame_id)
    else:
        sub_id, trial_id, session_id, pos_id, command_id, frame_id = rgb_image_filepath.split(os.path.sep)[-1].split("_")[-7:-1]
        rgb_image_aligned_filepath = "{}{}_{}_{}_{}_{}_{}_3.png".format(rgb_image_aligned_path, sub_id, trial_id, session_id, pos_id, command_id, frame_id)
        #rgb_lip_aligned_filepath = "{}{}_{}_{}_{}_{}_{}_4.png".format(lip_rgb_path, sub_id, trial_id, session_id, pos_id, command_id, frame_id) 
        #thr_lip_aligned_filepath = "{}{}_{}_{}_{}_{}_{}_5.png".format(lip_thr_path, sub_id, trial_id, session_id, pos_id, command_id, frame_id)
    
    # process only the files for the given position if "show" mode is enabled. 
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
        thr_image_filepath = path_to_thermal_image(rgb_image_filepath, sub_trial_dataset_path, os.path.sep+"thr_image{}".format(opt)+os.path.sep)
        
        # load rgb and corresponding thermal image 
        rgb = cv2.imread(rgb_image_filepath)
        thr = cv2.imread(thr_image_filepath)

        # grab a size of the thermal image 
        (H_thr, W_thr) = thr.shape[:2]
        
        # warp the rgb image to align with the thermal image
        rgb = cv2.warpPerspective(rgb, H, (W_thr, H_thr), 
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        if args["session"] == 2:
            #get the boundaries of a boundin box for the lip region from the warped rgb image
            shape = lip_region_extractor(face_net, rgb, thr, args["confidence"], dnn_mode=False)
            (startX, startY, endX, endY) = (shape[2][0], shape[2][1], shape[14][0], shape[8][1])
        
            # if the lip region was not detected properly then save and skip this frame
            if startX is None or startX < 0 or startY < 0 or endX < 0 or endY < 0:
                print("[INFO] Can't extract the lip region!!!")
                continue
            print(startX, startY, endX, endY)   
            # otherwise crop out the lip regions 
            lip_rgb = rgb[startY:endY, startX:endX]
            lip_thr = thr[startY:endY, startX:endX]
      
        if args["show"]:
            # make a copy of the rgb image
            # then replace its RED channel with the RED channel of the thermal image
            # show the images
            rgb_copy = rgb.copy()
            rgb_copy[:, :, 2] = thr[:, :, 2]
            cv2.imshow(" Face Sub:{} Trial:{} Session:{} Pos:{} Command:{} Frame:{}".format(sub_id, trial_id, session_id, pos_id, command_id, frame_id), np.hstack([rgb, thr, rgb_copy]))
            if args["session"] == 2:
                lip_rgb_copy = lip_rgb.copy()
                lip_rgb_copy[:, :, 2] = lip_thr[:, :, 2]    
                cv2.imshow("Lip Sub:{} Trial:{} Session:{} Pos:{} Command:{} Frame:{}".format(sub_id, trial_id, session_id, pos_id, command_id, frame_id), np.hstack([lip_rgb, lip_thr, lip_rgb_copy]))
            key = cv2.waitKey(0) & 0xFF

            # if the 'q' key is pressed, stop the loop, exit from the program
            if key == ord("q"):
                    break
        # note the images are written if only show is disabled 
        # all positions are processed at this point assuming that 
        # the optimal shifts have been found and provided
        # write the aligned rgb image
        cv2.imwrite(rgb_image_aligned_filepath, rgb)
        # write the extracted lip region
        #cv2.imwrite(rgb_lip_aligned_filepath, lip_rgb)
        #cv2.imwrite(thr_lip_aligned_filepath, lip_thr)
