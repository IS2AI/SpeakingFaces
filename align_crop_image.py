# import the necessary packages
from imutils import paths
from speakingfacespy.imtools import make_dir
from speakingfacespy.imtools import get_face_location_landmarks
from speakingfacespy.imtools import align_rgb_image

import imutils
import numpy as np
import pandas as pd
import cv2 
import argparse
import os
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--dataset", required=True,
	help="path to the dataset")
ap.add_argument("-i", "--sub_info",  nargs='+', type=int,
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

args = vars(ap.parse_args())


# grab the path to the visual images in our dataset
sub_trial_dataset_path = "{}sub_{}".format(args["dataset"], args["sub_info"][0])+os.path.sep+"trial_{}".format(args["sub_info"][1])  
rgb_image_filepaths = list(paths.list_images(sub_trial_dataset_path + os.path.sep+"rgb_image_cmd"))

# get the path to the thermal images in our dataset
thr_image_path = sub_trial_dataset_path + os.path.sep+"thr_image_cmd"+ os.path.sep
# create a directory to save aligned images
rgb_image_aligned_path = sub_trial_dataset_path + os.path.sep+"rgb_image_cmd_aligned"+os.path.sep
make_dir(rgb_image_aligned_path)
# create directories for extracted lip region
lip_rgb_path = sub_trial_dataset_path + os.path.sep+"rgb_roi_cmd"+os.path.sep
make_dir(lip_rgb_path)
lip_thr_path = sub_trial_dataset_path + os.path.sep+"thr_roi_cmd"+os.path.sep
make_dir(lip_thr_path)

# loop over original visible images in the folders
for rgb_image_filepath in rgb_image_filepaths:
    
    # extract the current image info
    sub_id, trial_id, session_id, pos_id, command_id, frame_id = rgb_image_filepath.split(os.path.sep)[-1].split("_")[-7:-1]    
    # process only the files for the given position if "show" mode is enabled. 
    if args["sub_info"][2] != int(pos_id):
        cv2.destroyAllWindows()
        continue

    print("[INFO] processing image {}".format(rgb_image_filepath.split(os.path.sep)[-1]))
    
    # construct the thermal image path using the rgb image path
    thr_image_filepath = "{}{}_{}_{}_{}_{}_{}_1.png".format(thr_image_path, sub_id, trial_id, session_id, pos_id, command_id, frame_id)
    rgb_image_aligned_filepath = "{}{}_{}_{}_{}_{}_{}_3.png".format(rgb_image_aligned_path, sub_id, trial_id, session_id, pos_id, command_id, frame_id)
    rgb_lip_aligned_filepath = "{}{}_{}_{}_{}_{}_{}_4.png".format(lip_rgb_path, sub_id, trial_id, session_id, pos_id, command_id, frame_id) 
    thr_lip_aligned_filepath = "{}{}_{}_{}_{}_{}_{}_5.png".format(lip_thr_path, sub_id, trial_id, session_id, pos_id, command_id, frame_id)

    # load rgb and corresponding thermal image 
    rgb = cv2.imread(rgb_image_filepath)
    thr = cv2.imread(thr_image_filepath)

    # initialize lists of shifts
    dy = args["dy"][int(pos_id) - 1]
    dx = args["dx"][int(pos_id) - 1]
    rgb = align_rgb_image(dy, dx, thr, rgb)
    
    if (args["face_detector"]!= "none"): 
        #get the boundaries of a boundin box for the lip region from the warped rgb image
        face_landmarks, face_locations = get_face_location_landmarks(rgb, args["confidence"], model=args["face_detector"])
        # compare the max of upper lip and third landmark, if the later one is lower, then take the second landmark as the top boundary
        if not (face_locations):
            print("[INFO] CANT GET THE LOCATION OF THE FACE!!!! STOP AND CHANGE THE FACE DETECTOR")
            break
        if not (face_landmarks):
            print("[INFO] CANT GET THE FACE LANDMARKS!!!! STOP AND CHANGE THE FACE DETECTOR")
            break
    
        face_location = face_locations[0]
        shape_chin, shape_mouth = face_landmarks[0]['chin'], face_landmarks[0]['top_lip']    
        upper_landmark_id = args["landmark"][int(pos_id) - 1]
        (startX, startY, endX, endY) = (shape_chin[2][0], shape_chin[upper_landmark_id][1], shape_chin[14][0],shape_chin[6][1])
        
        # if the lip region was not detected properly then save and skip this frame
        if startX < 0 or startY < 0 or endX < 0 or endY < 0:
            print("[INFO] CANT EXTRACT THE LIP REGION. NEGATIVE COORDINATES FOR THE BOUNDING BOX!!!")
            break

        # otherwise crop out the lip regions 
        if int(pos_id) > 6:
            startX = startX - 20 
            endX = endX - 20
        elif int(pos_id)> 3:
            startX = startX + 20
            endX = endX + 20
    else:
        startX, startY, endX, endY = args['roi']
    
    lip_rgb = rgb[startY:endY, startX:endX]
    lip_thr = thr[startY:endY, startX:endX]

    if args["show"]:
        # make a copy of the rgb image
        # then replace its RED channel with the RED channel of the thermal image
        # show the images
        rgb_copy = rgb.copy() 
        thr_copy = thr.copy()      
        hybrid = rgb.copy()
        hybrid[:, :, 2] = thr[:, :, 2]

        face_window_name = "Face Sub:{} Trial:{} Session:{} Pos:{} Command:{} Frame:{} startX: {} startY: {} endX: {} endY: {}".format(sub_id, trial_id, session_id, pos_id, command_id, frame_id, startX, startY, endX, endY)
        cv2.namedWindow(face_window_name,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(face_window_name, 1200, 350)
        cv2.moveWindow(face_window_name, 20, 20);
        cv2.rectangle(rgb_copy, (startX, startY), (endX, endY), (255, 0, 0), 2)
        
        if (args["face_detector"]!= "none"):  
            cv2.rectangle(rgb_copy, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (0, 255, 0), 2)
            for (x, y) in shape_chin:
                cv2.circle(rgb_copy, (x, y), 1, (0, 0, 255), -1)
            for (x, y) in shape_mouth:
                cv2.circle(rgb_copy, (x, y), 1, (0, 0, 255), -1) 
                cv2.circle(thr_copy, (x, y), 1, (0, 0, 255), -1) 
            cv2.circle(rgb_copy, (shape_chin[upper_landmark_id][0], shape_chin[upper_landmark_id][1]), 1, (255, 0, 0), -1)
            
        cv2.imshow(face_window_name, np.hstack([rgb_copy, thr_copy, hybrid]))
        

        lip_hybrid = lip_rgb.copy()
        lip_hybrid[:, :, 2] = lip_thr[:, :, 2]
        
        lip_window_name = "Lip Sub:{} Trial:{} Session:{} Pos:{} Command:{} Frame:{}".format(sub_id, trial_id, session_id, pos_id, command_id, frame_id)
        cv2.namedWindow(lip_window_name)
        cv2.moveWindow(lip_window_name, 20, 400);
        cv2.imshow(lip_window_name, np.hstack([lip_rgb, lip_thr, lip_hybrid]))

        
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
    lip_rgb_resized = cv2.resize(lip_rgb, args["size"], interpolation = cv2.INTER_AREA)
    lip_thr_resized = cv2.resize(lip_thr, args["size"], interpolation = cv2.INTER_AREA)

    cv2.imwrite(rgb_lip_aligned_filepath, lip_rgb_resized)
    cv2.imwrite(thr_lip_aligned_filepath, lip_thr_resized)
