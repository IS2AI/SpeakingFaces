# import the necessary packages
from imutils import paths
from speakingfacespy.imtools import path_to_thermal_image
from speakingfacespy.imtools import face_region_extractor
from speakingfacespy.imtools import get_homography_matrix
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
ap.add_argument("-t", "--type", type=str, default="train",
	help="type of data: train/test")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
	help="minimum probability to filter weak detections")
ap.add_argument("-n", "--frame", type=int, default=1,
	help="process every n'th frame")
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
H = get_homography_matrix(M, N=1000)

# shifts in x and y axises
dx = args["dx"]
dy = args["dy"]

# load serialized face detector from disk
print("[INFO] loading face detector ...")
face_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", 
	"models/res10_300x300_ssd_iter_140000.caffemodel")

# grab the path to the visual images in our dataset
dataset_path = "{}sub_{}/trial_{}".format(args["dataset"], args["sub_info"][0],
					args["sub_info"][1])  
rgbImagePaths = list(paths.list_images(dataset_path + "/rgb_image"))

# create directories to save data
if args["type"] == "train":
	pathA = dataset_path + "/cycleGAN/trainA/"
	pathB = dataset_path + "/cycleGAN/trainB/"
else:
	pathA = dataset_path + "/cycleGAN/testA/"
	pathB = dataset_path + "/cycleGAN/testB/"

make_dir(pathA)
make_dir(pathB)

# loop over images in the folders
for rgbImagePath in rgbImagePaths:

	# extract the current image info
	sub, trial, pos, image_id = rgbImagePath.split("/")[-1].split("_")[-5:-1]

	# process only n'th frames  
	if int(image_id) % args["frame"] == 0:
		print("[INFO] processing image {}".format(rgbImagePath.split("/")[-1]))

		# construct the thermal 
		# image path using the rgb image path
		thrImagePath = path_to_thermal_image(rgbImagePath, dataset_path)
		
		# load rgb and corresponding thermal image 
		rgb = cv2.imread(rgbImagePath)
		thr = cv2.imread(thrImagePath)

		# grab a size of the thermal image 
		(H_thr, W_thr) = thr.shape[:2]

		# warp the rgb image
		# to align with the thermal image
		rgb = cv2.warpPerspective(rgb, H, (W_thr, H_thr), 
			flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

		# adjust the alignment if there is still 
		# some misalignment among x and y axises
		rgb = rgb[dy:H_thr, dx:W_thr]
		thr = thr[0:H_thr - dy, 0:W_thr - dx]

		# give the rgb image to the face detector
		# as an input and get a bouding box of a face
		# as an output 
		(startX, startY, endX, endY) = face_region_extractor(face_net, rgb, thr, args["confidence"])

		# if the face is not detected
		# then skip the current image 
		if startX is None:
			continue

		# crop the detected faces 
		rgb_face = rgb[startY:endY, startX:endX]
		thr_face = thr[startY:endY, startX:endX]

		if args["show"]:
			# make a copy of the rgb image
			# then replace its RED channel with 
			# the RED channel of the thermal image
			rgb_copy = rgb.copy()
			rgb_copy[:, :, 2] = thr[:, :, 2]

			# the same for the faces
			rgb_face_copy = rgb_face.copy()
			rgb_face_copy[:, :, 2] = thr_face[:, :, 2]

			# show the images
			cv2.imshow("Output", np.hstack([rgb, thr, rgb_copy]))
			cv2.imshow("Faces", np.hstack([rgb_face, thr_face, rgb_face_copy]))
			key = cv2.waitKey(0) & 0xFF

			# if the 'q' key is pressed, stop the loop
			if key == ord("q"):
				break
		
		cv2.imwrite("{}{}_{}_{}_{}.png".format(pathA, sub, trial, pos, image_id), thr_face)
		cv2.imwrite("{}{}_{}_{}_{}.png".format(pathB, sub, trial, pos, image_id), rgb_face)

	