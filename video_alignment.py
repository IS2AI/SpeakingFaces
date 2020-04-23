# import the necessary packages
from imutils import paths
from speakingfacespy.imtools import make_dir
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
ap.add_argument("-k", "--stack", type=int, default=4,
	help ="choose which stream to add to the stack, 0 - all, 1 - only aligned, 2 - only red ")
args = vars(ap.parse_args())

# initialize shifts in x and y axises
dx = args["dx"]
dy = args["dy"]

ptsA = np.array([[399 + dx, 345 + dy], [423 + dx, 293 + dy], [293 + dx, 316 + dy], [269 + dx, 368 + dy]])
ptsB = np.array([[249, 237], [267, 196], [169, 214], [151, 254]])
# estimate a homography matrix to warp the visible image
(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 2.0)

# grab the path to the visual videos in our dataset
dataset_path = "{}sub_{}/trial_{}/".format(args["dataset"], args["sub_info"][0],
					args["sub_info"][1])  
rgbVideoPaths = list(paths.list_files(dataset_path + args["video"], validExts=('.avi',)))
print(rgbVideoPaths)
# create a directory to 
# save aligned videos 
rgbDir = dataset_path + "/" + args["video"] + "_aligned/" 
make_dir(rgbDir)

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

		# create a stack of images
		rgb_copy = warpedRGB.copy()
		rgb_copy[:, :, 2] = frameThr[:, :, 2]
		if args["stack"] == 2:
			stack_image = np.hstack([warpedRGB])
		elif args["stack"] == 3:
			stack_image = np.hstack([rgb_copy])
		else:
			stack_image = np.hstack([frameThr, warpedRGB, rgb_copy])

		# check if the video writer is None
		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			filename = rgbDir + "/{}_{}_{}_{}.avi".format(sub, trial, pos, args["stack"])
			writer = cv2.VideoWriter(filename, fourcc, 28,
				(stack_image.shape[1], stack_image.shape[0]), True)

		# write the output frame to disk
		writer.write(stack_image)

		# show the aligned frames 
		# and extracted lip regions
		if args["show"]:
			# show the images
			cv2.imshow("Output", stack_image)
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
