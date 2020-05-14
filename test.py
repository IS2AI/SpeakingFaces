# import the necessary packages
from imutils import paths
from speakingfacespy.imtools import path_to_thermal_image
from speakingfacespy.imtools import make_dir
from speakingfacespy.imtools import lip_region_extractor
from speakingfacespy.imtools import face_region_extractor

import imutils
import numpy as np
import pandas as pd
import cv2 
import argparse
import os

# parse the provided arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path to the image")
ap.add_argument("-y", "--dy",  nargs='+', type=int,
	help="a list of shifts in y axis for each position")
ap.add_argument("-x", "--dx",  nargs='+', type=int,
	help="a list of shifts in x axis for each position")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
    help="a list of minimum probabilities for each position to filter out weak detections")
ap.add_argument("-l", "--landmark", type=int,
    help="a list of landmark ids that should server as the upper bound to crop for each position")

args = vars(ap.parse_args())

# load matched features from xlsx file and convert it numpy array 
df = pd.read_excel (r'calibration'+os.path.sep+'matched_features.xlsx')
M = df.to_numpy()


print("[INFO] loading the face and landmark predictors ...")
face_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", 
    "models/res10_300x300_ssd_iter_140000.caffemodel")                      

rgb_image_filepath = args["path"]
sub_id, trial_id, session_id, pos_id, command_id, frame_id = rgb_image_filepath.split(os.path.sep)[-1].split("_")[-7:-1]


# initialize lists of shifts
dy = args["dy"][0]
dx = args["dx"][0]

ptsA = np.array([[399 + dx, 345 + dy], [423 + dx, 293 + dy], [293 + dx, 316 + dy], [269 + dx, 368 + dy]])
ptsB = np.array([[249, 237], [267, 196], [169, 214], [151, 254]])

# estimate a homography matrix to warp the visible image
(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 2.0)

# process only n'th frames  
print("[INFO] processing image {}".format(rgb_image_filepath.split(os.path.sep)[-1]))

# construct the thermal image path using the rgb image path
thr_image_filepath = path_to_thermal_image(rgb_image_filepath, os.path.sep + "thr_image_cmd" + os.path.sep)

# load rgb and corresponding thermal image 
rgb = cv2.imread(rgb_image_filepath)
thr = cv2.imread(thr_image_filepath)

# grab a size of the thermal image 
(H_thr, W_thr) = thr.shape[:2]

# warp the rgb image to align with the thermal image
rgb = cv2.warpPerspective(rgb, H, (W_thr, H_thr), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

shape_chin, shape_mouth, bb  = lip_region_extractor(face_net, rgb, args["confidence"], dnn_mode=)
if shape_chin is None:
    print("[INFO] Can't extract the lip region!!!")
else:
	upper_landmark_id = args["landmark"]
	(startX, startY, endX, endY) = (shape_chin[2][0], shape_chin[upper_landmark_id][1], shape_chin[14][0],shape_chin[6][1])

	# if the lip region was not detected properly then save and skip this frame
	if startX < 0 or startY < 0 or endX < 0 or endY < 0:
	    print("[INFO] Can't extract the lip region!!!")
	else:
		if int(pos_id) > 6:
			startX = startX - 20 
			endX = endX - 20
		elif int(pos_id)> 3:
			startX = startX + 20
			endX = endX + 20

		#print(startX, startY, endX, endY)   
		# otherwise crop out the lip regions 
		lip_rgb = rgb[startY:endY, startX:endX]
		lip_thr = thr[startY:endY, startX:endX]

		rgb_copy = rgb.copy() 
		thr_copy = thr.copy()      
		hybrid = rgb.copy()
		hybrid[:, :, 2] = thr[:, :, 2]

		face_window_name = "Face Sub:{} Trial:{} Session:{} Pos:{} Command:{} Frame:{}".format(sub_id, trial_id, session_id, pos_id, command_id, frame_id)
		cv2.namedWindow(face_window_name,cv2.WINDOW_NORMAL)
		cv2.resizeWindow(face_window_name, 1200, 350)
		cv2.moveWindow(face_window_name, 20, 20);

		lip_window_name = "Lip Sub:{} Trial:{} Session:{} Pos:{} Command:{} Frame:{}".format(sub_id, trial_id, session_id, pos_id, command_id, frame_id)
		cv2.namedWindow(lip_window_name)
		cv2.moveWindow(lip_window_name, 20, 400);


		cv2.rectangle(rgb_copy, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
		for (x, y) in shape_chin:
		    cv2.circle(rgb_copy, (x, y), 1, (0, 0, 255), -1)
		for (x, y) in shape_mouth:
		    cv2.circle(rgb_copy, (x, y), 1, (0, 0, 255), -1) 
		    cv2.circle(thr_copy, (x, y), 1, (0, 0, 255), -1) 
		           
		cv2.circle(rgb_copy, (shape_chin[upper_landmark_id][0], shape_chin[upper_landmark_id][1]), 1, (255, 0, 0), -1)
		lip_hybrid = lip_rgb.copy()
		lip_hybrid[:, :, 2] = lip_thr[:, :, 2]
		cv2.imshow(face_window_name, np.hstack([rgb_copy, thr_copy, hybrid]))
		cv2.imshow(lip_window_name, np.hstack([lip_rgb, lip_thr, lip_hybrid]))


		key = cv2.waitKey(0) & 0xFF

		  