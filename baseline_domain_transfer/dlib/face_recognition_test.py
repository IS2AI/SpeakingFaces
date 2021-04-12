# import the necessary packages
from imutils import paths
import face_recognition
import numpy as np
import argparse
import pickle
import cv2
import os

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to the images")
ap.add_argument("-t", "--thr", type=float, default=0.6,
	help="threshold for encodings")
ap.add_argument("-s", "--size", type=int, default=128,
	help="image size")
ap.add_argument("-v", "--vis", type=int, default=0,
	help="visualize")
args = vars(ap.parse_args())

# construct a list of paths to the real visual images
imagePaths = list(paths.list_images(args["images"]))

# load the known faces and encodings
print("[INFO] loading facial encodings...")
embeddings = pickle.loads(open("embeddings/embeddings_{}.pickle".format(args["size"]),
							   "rb").read())

# define True Positive, False Positive, 
# and False Negative
TP = 0
FP = 0
FN = 0

# loop over the rgb images
for ind, imagePath in enumerate(imagePaths, 1):
	# extract the image name 
	imageName = imagePath.split("/")[-1]
	
	# extract the subject ID
	sub_id, trial, *rest  = imageName.split("_")

	# skip the first trial images
	if trial == '1':
		continue

	# load the image
	image = cv2.imread(imagePath)		

	# height and width of the image
	(h, w) = image.shape[:2]
		
	# compute the facial embedding for the faces 
	encoding = face_recognition.face_encodings(image, [(0, w, h, 0)], num_jitters=5, model="large")[0]

	# attempt to match each face in the input image to our 
	# known encodings
	matches = face_recognition.compare_faces(embeddings["encodings"],
					encoding, tolerance=args["thr"])
	name = "Unknown"

	# check to see if we have found a match
	if True in matches:
		# find the indexes of all matched faces then initialize a 
		# dictionary to count the total number of times each face
		# was matched 
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}

		# loop over the matched indexes and maintain a count for
		# each recognized face face
		for i in matchedIdxs:
			name = embeddings["names"][i]
			counts[name] = counts.get(name, 0) + 1

		# determine the recognized face with the largest number of 
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)
		name = max(counts, key=counts.get)

	if name == "Unknown":
		FN += 1
		color = (0, 0, 255)
	elif name == sub_id:
		TP += 1
		color = (0, 255, 0)
	else:
		FP += 1
		color = (0, 0, 255)

	print("[INFO] TP: {}, FP: {}, FN: {}. Image {}/{}".format(TP, FP, FN, ind, len(imagePaths)))

	if args["vis"]:
		cv2.putText(image, "{}/{}".format(name, sub_id), (15, 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)
		
		cv2.imshow('Image', image)
		key = cv2.waitKey(0) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break