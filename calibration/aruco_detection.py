import numpy as np
import cv2
from cv2 import aruco
import pandas as pd
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--start", type=int, default=1, required=True,
	help = "ID of the first frame")
ap.add_argument("-e", "--end", type=int, default=1, required=True,
	help = "ID of the last frame")
args = vars(ap.parse_args())

# set some parameters of the aruco detection
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
parameters =  aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 10
parameters.polygonalApproxAccuracyRate = 0.012

# create a containter to 
# store corners of the detected markers
F = []

# loop over the list of images 
for i in range(args["start"], args["end"]):
	# print the current processing image ID
	print("[INFO] Processing image [ID]:", i)

	# load the visible and thermal image
	image_thr = cv2.imread("images/{}_0.png".format(i))
	image_rgb = cv2.imread("images/{}_1.png".format(i))

	# convert images to grayscale
	gray_thr = cv2.cvtColor(image_thr, cv2.COLOR_BGR2GRAY)
	gray_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

	# invert the grayscale image of thermal image
	gray_thr = 255 - gray_thr

	# detect markers
	corners_thr, ids_thr, _ = aruco.detectMarkers(gray_thr, aruco_dict, parameters=parameters)
	corners_rgb, ids_rgb, _ = aruco.detectMarkers(gray_rgb, aruco_dict, parameters=parameters)

	# draw the detected markers
	aruco.drawDetectedMarkers(image_thr, corners_thr, ids_thr)
	aruco.drawDetectedMarkers(image_rgb, corners_rgb, ids_rgb)

	# show the detected markers
	cv2.imshow("visible_{}".format(i), image_rgb)
	cv2.imshow("thermal_{}".format(i), image_thr)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	# if markers were detected in both images
	# then loop over the detections to find
	# markers with same IDs
	if corners_rgb and corners_thr:
		for ind1 in range(len(ids_thr)):
			for ind2 in range(len(ids_rgb)):
				if ids_thr[ind1] == ids_rgb[ind2]:
					F.append(np.concatenate((corners_thr[ind1][0,0], corners_rgb[ind2][0,0]), axis = 0))
					F.append(np.concatenate((corners_thr[ind1][0,1], corners_rgb[ind2][0,1]), axis = 0))
					F.append(np.concatenate((corners_thr[ind1][0,2], corners_rgb[ind2][0,2]), axis = 0))
					F.append(np.concatenate((corners_thr[ind1][0,3], corners_rgb[ind2][0,3]), axis = 0))
				else:
					continue

# save the list of matched features
df = pd.DataFrame(F)
df.to_excel("matched_features.xlsx")

