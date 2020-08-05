# import the necessary packages
from speakingfacespy.imtools import face_region_extractor
from speakingfacespy.imtools import make_dir
import face_recognition
import pandas as pd
import numpy as np
import cv2
import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the SpeakingFaces dataset")
ap.add_argument("-m", "--model", type=str, default="dnn",
	help="face detection model: dnn/hog")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
	help="a minimum probability for dnn to filter weak detections")
ap.add_argument("-n", "--frame", type=int, default=100,
	help="process only every n'th frame")
ap.add_argument("-s", "--show", type=int, default=0,
	help="visualize extracted faces")
args = vars(ap.parse_args())

# load dnn face detector from disk
# if it was selected
if args["model"] == "dnn": 
	print("[INFO] loading dnn face detector...")
	face_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", 
		"models/res10_300x300_ssd_iter_140000.caffemodel")

# initialize a path to dataset
path_to_dataset = args["dataset"]

# create directories to save train data
train_pathA = os.path.join(path_to_dataset, "cycleGAN/trainA")
train_pathB = os.path.join(path_to_dataset, "cycleGAN/trainB")
make_dir(train_pathA)
make_dir(train_pathB)
	
# and testing data
test_pathA = os.path.join(path_to_dataset, "cycleGAN/testA")
test_pathB = os.path.join(path_to_dataset, "cycleGAN/testB")
make_dir(test_pathA)
make_dir(test_pathB)

# initialize the total number trials, positions per 
# trial, and frames per position
sub_ids = 142
trial_ids = 2
pos_ids = 9
frame_ids = 900

# initialize a counter to count 
# a total number of processed images
train_samples = 0
test_samples = 0

# read information about subjects
# from CSV file to NumPy array
# sub_id - 'train/val/test' - age - gender - ...
sub_info = pd.read_csv('metadata/subjects.csv').to_numpy()

# loop over the subjects 1...142
for sub_id in range(1, sub_ids + 1):
	# skip african subjects 
	ethnicity = sub_info[sub_id - 1, 4]
	if ethnicity == "Black":
		continue

	# loop over the trials 1..2
	for trial_id in range(1, trial_ids + 1):
		# skip subjects with accecories
		acc = sub_info[sub_id - 1, trial_id + 4]
		if acc != "None":
			continue

		# construct path to the folders with rgb and thermal images
		if sub_id <= 100:
			rgb_path = os.path.join(args["dataset"], "train_data/sub_{}/trial_{}/rgb_image_aligned".format(sub_id, trial_id))
			thr_path = os.path.join(args["dataset"], "train_data/sub_{}/trial_{}/thr_image".format(sub_id, trial_id))
		elif sub_id > 100 and sub_id <= 120:
			rgb_path = os.path.join(args["dataset"], "valid_data/sub_{}/trial_{}/rgb_image_aligned".format(sub_id, trial_id))
			thr_path = os.path.join(args["dataset"], "valid_data/sub_{}/trial_{}/thr_image".format(sub_id, trial_id))
		else:
			rgb_path = os.path.join(args["dataset"], "test_data/sub_{}/trial_{}/rgb_image_aligned".format(sub_id, trial_id))
			thr_path = os.path.join(args["dataset"], "test_data/sub_{}/trial_{}/thr_image".format(sub_id, trial_id))

		# loop over the positions 1...9
		for pos_id in range(1, pos_ids + 1):

			# loop over the frames 1...900
			for frame_id in range(1, frame_ids + 1):
				# process only n'th frames
				if frame_id % args["frame"] != 0:
					continue

				print("[INFO] processing sub:{}, trial:{}, pos:{}, frame:{}".format(sub_id, trial_id, pos_id, frame_id))
				# construct rgb and thermal  
				rgb_file = "{}_{}_1_{}_{}_{}.png".format(sub_id, trial_id, pos_id, frame_id, 3)
				thr_file = "{}_{}_1_{}_{}_{}.png".format(sub_id, trial_id, pos_id, frame_id, 1)

				# construct the source path
				rgb_file = os.path.join(rgb_path, rgb_file)
				thr_file = os.path.join(thr_path, thr_file)
				#print(rgb_file)
				#print(thr_file)

				# load rgb and thermal images
				rgb_image = cv2.imread(rgb_file)		
				thr_image = cv2.imread(thr_file)

				# detect the (x ,y)-coordinates of the bounding boxes
				# corresponding to each face in the input image 
				if args["model"] != "dnn":
					rgb_boxes = face_recognition.face_locations(rgb_image, model=args["model"])
				else:
					rgb_boxes = face_region_extractor(face_net, rgb_image, args["confidence"])

				# if at least one face is detected
				if len(rgb_boxes):
					# we assume that only one person was detected 
					# extract corners of the bounding box
					(startY, endX, endY, startX) = rgb_boxes[0]

					# crop the detected faces 
					rgb_face = rgb_image[startY:endY, startX:endX]
					thr_face = thr_image[startY:endY, startX:endX]

					if args["show"]:
						# make a copy of the rgb image then replace its RED channel with 
						# the RED channel of the thermal image
						rgb_copy = rgb_image.copy()
						rgb_copy[:, :, 2] = thr_image[:, :, 2]

						# the same for the faces
						rgb_face_copy = rgb_face.copy()
						rgb_face_copy[:, :, 2] = thr_face[:, :, 2]

						# show the images
						cv2.imshow("Output", np.hstack([rgb_image, thr_image, rgb_copy]))
						cv2.imshow("Faces", np.hstack([rgb_face, thr_face, rgb_face_copy]))
						key = cv2.waitKey(0) & 0xFF

						# if the 'q' key is pressed, stop the loop
						if key == ord("q"):
							break

					# save images
					if sub_id <= 100:
						train_samples += 1
						cv2.imwrite(os.path.join(train_pathA, "{}_{}_{}_{}.png".format(sub_id, trial_id, pos_id, frame_id)), thr_face)
						cv2.imwrite(os.path.join(train_pathB, "{}_{}_{}_{}.png".format(sub_id, trial_id, pos_id, frame_id)), rgb_face)
					else:
						test_samples += 1
						cv2.imwrite(os.path.join(test_pathA, "{}_{}_{}_{}.png".format(sub_id, trial_id, pos_id, frame_id)), thr_face)
						cv2.imwrite(os.path.join(test_pathB, "{}_{}_{}_{}.png".format(sub_id, trial_id, pos_id, frame_id)), rgb_face)

print("[INFO] Processing is done! Total number of train samples: {}, and test samples: {}".format(train_samples, test_samples))
	
