# import the necessary packages
from imutils import paths
from speakingfacespy.imtools import lip_region_extractor
from speakingfacespy.imtools import face_region_extractor
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
ap.add_argument("-a", "--height", type=int, default=64,
	help="height of the cropped lip region")
ap.add_argument("-w", "--width", type=int, default=128,
	help="width of the cropped lip region")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
	help="minimum probability to filter out weak detections")
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

# load the serialized models from disk
print("[INFO] loading the face and landmark predictors ...")
face_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", 
	"models/res10_300x300_ssd_iter_140000.caffemodel")
#landmark_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# grab the path to the visual videos in our dataset
dataset_path = "{}sub_{}/trial_{}".format(args["dataset"], args["sub_info"][0],
					args["sub_info"][1])  
rgbVideoPaths = list(paths.list_files(dataset_path + "/rgb_video_cmd"))

# loop over videos in the folders
for rgbVideoPath in rgbVideoPaths:
	# extract the current video info
	sub, trial, pos, id, cmd = rgbVideoPath.split("/")[-1].split("_")
	cmd2 = cmd.split(".")[-2]

	print("[INFO]: Processing video file: {}".format(rgbVideoPath.split("/")[-1]))

	# construct a path to the thermal video file
	thrVideoPath = "/{}_{}_{}_{}_{}".format(sub, trial, pos, 0, cmd)
	thrVideoPath = dataset_path + "/thr_video_cmd" + thrVideoPath

	# initialize the video streams, pointers to output video files, and
	# a dimension of the thermal frame 
	vsRGB = cv2.VideoCapture(rgbVideoPath)
	vsThr = cv2.VideoCapture(thrVideoPath)
	(W_thr, H_thr) = (None, None)

	# define a frame counter
	frameCounter = 1

	# create directories to 
	# save extracted lips 
	rgbDir = dataset_path + "/rgb_image_cmd/" + cmd2
	thrDir = dataset_path + "/thr_image_cmd/" + cmd2
	createDirectory(rgbDir)
	createDirectory(thrDir)

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

		# give the rgb frame to the landmark detector
		# as an input and get a bouding box of the lip
		# as an output
		shape = lip_region_extractor(face_net, warpedRGB, frameThr, args["confidence"], dnn_mode=False)
		(startX, startY, endX, endY) = (shape[2][0], shape[2][1], shape[14][0], shape[8][1])

		# if the lip region was not detected properly
		# then save and skip this frame
		if startX is None or startX < 0 or startY < 0 or endX < 0 or endY < 0:
			print("[INFO] Can't extract the lip region!!!")
			cv2.imwrite("{}_{}.png".format(rgbVideoPath.split("/")[-1], frameCounter), frameVis)
			cv2.imwrite("{}_{}.png".format(rgbVideoPath.split("/")[-1], frameCounter), warpedVis)
			continue

		# crop the leap regions 
		lipRGB = warpedRGB[startY:endY, startX:endX]
		lipThr = frameThr[startY:endY, startX:endX]

		# save extracted lips
		cv2.imwrite(rgbDir + "/{}.png".format(frameCounter), lipRGB)
		cv2.imwrite(thrDir + "/{}.png".format(frameCounter), lipThr)

		frameCounter = frameCounter + 1

		# show the aligned frames 
		# and extracted lip regions
		if args["show"]:
			# make a copy of the warpedRGB image
			# then replace its RED channel by the 
			# RED channel of the thermal image
			rgb_copy = warpedRGB.copy()
			# draw facial landmarks
			for (x, y) in shape:
				cv2.circle(rgb_copy, (x, y), 2, (0, 255, 0), -1)

			rgb_copy[:, :, 2] = q

			# show the images
			cv2.imshow("Output", np.hstack([warpedRGB, frameThr, rgb_copy]))
			cv2.imshow("Lips", np.hstack([lipRGB, lipThr]))
			key = cv2.waitKey(1) & 0xFF

			# if the 'q' key is pressed, 
			# then break from the loop
			if key == ord("q"):
				break


	# do a bit cleanup 
	cv2.destroyAllWindows()
	vsRGB.release()
	vsThr.release()
	time.sleep(1)