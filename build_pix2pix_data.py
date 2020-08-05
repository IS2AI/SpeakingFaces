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

# create directories to save train data
train_path = os.path.join(args["dataset"], "pix2pix/train")
make_dir(train_path)

# validation data
val_path = os.path.join(args["dataset"], "pix2pix/val")
make_dir(val_path)

# and testing data
test_path = os.path.join(args["dataset"], "pix2pix/test")
make_dir(test_path)

# initialize the total number of subjects,
# trials, and positions
sub_ids = 142
trial_ids = 2
pos_ids = 9
frame_ids = 900

# initialize a total number of subjects 
# for training and testing
num_train_subjects = 100
num_val_subjects = 22
num_test_subjects = 20

# initialize counters
train_images = 0
val_images = 0
test_images = 0

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
		# skip subjects with glasses
		glasses = sub_info[sub_id - 1, trial_id + 4]
		if glasses == "Glasses":
			continue

		# construct path to the folders with rgb and thermal images
		rgb_path = os.path.join(args["dataset"], "sub_{}/trial_{}/rgb_image_aligned".format(sub_id, trial_id))
		thr_path = os.path.join(args["dataset"], "sub_{}/trial_{}/thr_image".format(sub_id, trial_id))

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
						#cv2.imshow("Output", np.hstack([rgb_image, thr_image, rgb_copy]))
						cv2.imshow("Faces", np.hstack([rgb_face, thr_face, rgb_face_copy]))
						key = cv2.waitKey(0) & 0xFF

						# if the 'q' key is pressed, stop the loop
						if key == ord("q"):
							break

					if sub_id <= num_train_subjects:
						train_images += 1
						cv2.imwrite(os.path.join(train_path, "{}_{}_{}_{}.png".format(sub_id, trial_id, pos_id, frame_id)), np.hstack([thr_face, rgb_face]))
					elif sub_id > num_train_subjects and sub_id <= num_train_subjects + num_val_subjects:
						val_images += 1
						cv2.imwrite(os.path.join(val_path, "{}_{}_{}_{}.png".format(sub_id, trial_id, pos_id, frame_id)), np.hstack([thr_face, rgb_face]))
					else:
						test_images += 1
						cv2.imwrite(os.path.join(test_path, "{}_{}_{}_{}.png".format(sub_id, trial_id, pos_id, frame_id)), np.hstack([thr_face, rgb_face]))

print("[INFO] Processing is done! Total number of images: train :{}, val :{}, test: {}".format(train_images, val_images, test_images))
	