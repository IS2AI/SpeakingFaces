# import the necessary packages
from imutils import paths
from speakingfacespy.imtools import homography_matrix
from speakingfacespy.imtools import createDirectory
import imutils
import numpy as np
import cv2 
import dlib
import argparse
import os
import time
import pandas as pd

# create argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to dataset")
ap.add_argument("-v", "--video", type=str, default="video_only",
	help="path to video files: video_only/video_audio")
ap.add_argument("-x", "--dx", type=int, default=0,
	help="shift in x axis to dx pixels")
ap.add_argument("-y", "--dy", type=int, default=0,
	help="shift in y axis to dy pixels")
ap.add_argument("-i", "--sub_info",  nargs='+', type=int,
	help="subject info: ID, trial #")
ap.add_argument("-s", "--show", type=int, default=0,
	help="visualize extracted faces")
args = vars(ap.parse_args())

# load matched features from xlsx file
# and convert it numpy array 
df = pd.read_excel (r'calibration/matched_features.xlsx')
M = df.to_numpy()

# estimate a homoghraphy matrix
# which will be used to align visible 
# and thermal frames
H = homography_matrix(M, N=1300)

# shifts in x and y axises
dx = args["dx"]
dy = args["dy"]

# grab the path to the visual videos in our dataset
dataset_path = "{}sub_{}/trial_{}/".format(args["dataset"], args["sub_info"][0],
					args["sub_info"][1])  
rgbVideoPaths = list(paths.list_files(dataset_path + args["video"]))

# create a directory to 
# save aligned videos 
rgbDir = dataset_path + "/" + args["video"] + "_aligned/" 
createDirectory(rgbDir)

# loop over videos in the folders
for rgbVideoPath in rgbVideoPaths:
	# extract the current video info
	sub, trial, pos, id = rgbVideoPath.split("/")[-1].split("_")

	# skip thermal videos
	if id.split(".")[-2] == '0':
		continue

	# construct a path to the thermal video file
	thrVideoPath = "/{}_{}_{}_0.avi".format(sub, trial, pos, 0)
	print("[INFO]: Processing video file: {}".format(rgbVideoPath.split("/")[-1]))
	thrVideoPath = dataset_path + args["video"] + thrVideoPath

	# initialize the video streams, pointers to output video files, and
	# a dimension of the thermal frame 
	vsRGB = cv2.VideoCapture(rgbVideoPath)
	vsThr = cv2.VideoCapture(thrVideoPath)
	
	# initialize a video writer
	writer = None

	# get a dimension 
	# of the thermal frame
	(W_thr, H_thr) = (None, None)

	while True:
		# read the next frames from the files
		(grabbedRGB, frameRGB) = vsRGB.read()
		(grabbedThr, frameThr) = vsThr.read()

		# if the frames were not grabbed, then we have reached the end
		# of the streams
		if not grabbedRGB or not grabbedThr:
			break

		# if the thermal frame dimension is empty, https://x.company/careers-at-x/
		# grab them
		if W_thr is None or H_thr is None:
			(H_thr, W_thr) = frameThr.shape[:2]

		# warp the rgb frame
		# to align it with the thermal frame
		warpedRGB = cv2.warpPerspective(frameRGB, H, (W_thr, H_thr), 
			flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

		# adjust the alignment if there is still 
		# some misalignment among x and y axises
		warpedRGB = warpedRGB[dy:H_thr, dx:W_thr]
		frameThr = frameThr[0:H_thr - dy, 0:W_thr - dx]

		# check if the video writer is None
		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			filename = rgbDir + "/{}_{}_{}_1.avi".format(sub, trial, pos, 1) 
			writer = cv2.VideoWriter(filename, fourcc, 28,
				(warpedRGB.shape[1], warpedRGB.shape[0]), True)

		# write the output frame to disk
		writer.write(warpedRGB)

		# show the aligned frames 
		# and extracted lip regions
		if args["show"]:
			# make a copy of the warpedRGB image
			# then replace its RED channel by the 
			# RED channel of the thermal image
			rgb_copy = warpedRGB.copy()
			rgb_copy[:, :, 2] = frameThr[:, :, 2]

			# show the images
			cv2.imshow("Output", np.hstack([warpedRGB, frameThr, rgb_copy]))
			key = cv2.waitKey(1) & 0xFF

			# if the 'q' key is pressed, 
			# then break from the loop
			if key == ord("q"):
				break


	# do a bit cleanup 
	cv2.destroyAllWindows()
	writer.release()
	vsRGB.release()
	vsThr.release()
	time.sleep(1)