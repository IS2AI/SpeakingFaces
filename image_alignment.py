# import the necessary packages
from imutils import paths
from speakingfacespy.imtools import pathToThermalImage
from speakingfacespy.imtools import homography_matrix
from speakingfacespy.imtools import createDirectory
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
	help="visualize extracted faces")
args = vars(ap.parse_args())

# load matched features from xlsx file
# and convert it numpy array 
df = pd.read_excel (r'calibration/matched_features.xlsx')
M = df.to_numpy()

# grab the path to the visual images in our dataset
dataset_path = "{}sub_{}/trial_{}".format(args["dataset"], args["sub_info"][0],
					args["sub_info"][1])  
rgbImagePaths = list(paths.list_images(dataset_path + "/rgb_image"))

# create a directory to save images
path = dataset_path + "/rgb_image_aligned/"
createDirectory(path)

# loop over images in the folders
for rgbImagePath in rgbImagePaths:

	# extract the current image info
	sub, trial, pos, image_id = rgbImagePath.split("/")[-1].split("_")[-5:-1]

	# process images for the position
	# given by the argument only if "show" mode
	# is enabled
	if args["sub_info"][2] != int(pos) and args["show"]:
		cv2.destroyAllWindows()
		continue

	# initialize lists of shifts
	dy = args["dy"][int(pos) - 1]
	dx = args["dx"][int(pos) - 1]

	ptsA = np.array([[399 + dx, 345 + dy], [423 + dx, 293 + dy], [293 + dx, 316 + dy], [269 + dx, 368 + dy]])
	ptsB = np.array([[249, 237], [267, 196], [169, 214], [151, 254]])

	# estimate a homography matrix to warp the visible image
	(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 2.0)
	
	# initialize lists of shifts
	#dy = args["dy"][int(pos) - 1]
	#dx = args["dx"][int(pos) - 1]

	# estimate a homoghraphy matrix
	# which will be used to align visible 
	# and thermal frames
	#H = homography_matrix(M, dx, dy, N=10)

	# process only n'th frames  
	if int(image_id) % args["frame"] == 0:
		print("[INFO] processing image {}".format(rgbImagePath.split("/")[-1]))

		# construct the thermal 
		# image path using the rgb image path
		thrImagePath = pathToThermalImage(rgbImagePath, dataset_path)
		
		# load rgb and corresponding thermal image 
		rgb = cv2.imread(rgbImagePath)
		thr = cv2.imread(thrImagePath)

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
			cv2.imshow("Sub:{} Trial:{} Pos:{} Frame:{}".format(sub, trial, pos, image_id), np.hstack([rgb, thr, rgb_copy]))
			#cv2.imwrite("Sub:{} Trial:{} Pos:{} Frame:{}.png".format(sub, trial, pos, image_id), np.hstack([rgb, thr, rgb_copy]))
			key = cv2.waitKey(0) & 0xFF

			# if the 'q' key is pressed, stop the loop
			if key == ord("q"):
				break

		cv2.imwrite("{}{}_{}_{}_{}_2.png".format(path, sub, trial, pos, image_id), rgb)
