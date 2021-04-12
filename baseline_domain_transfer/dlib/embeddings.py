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
ap.add_argument("-s", "--size", type=int, default=128,
	help="image size")
args = vars(ap.parse_args())

# construct a list of paths to the real visual images
imagePaths = list(paths.list_images(args["images"]))

# initialize lists of encodings 
# and labels
encodings = []
labels = []

total_images = 0
# loop over the rgb images
for ind, imagePath in enumerate(imagePaths, 1):
	print("[INFO] Accepted/Current Iteration/Total Iterations: {}/{}/{}".format(total_images, ind, len(imagePaths)))

	# extract the image name 
	imageName = imagePath.split("/")[-1]
	
	# extract the subject ID
	sub_id, trial, *rest  = imageName.split("_")

	# skip the second trial images
	if trial == '2':
		continue

	# load the image
	image = cv2.imread(imagePath)		

	# height and width of the image
	(h, w) = image.shape[:2]
		
	# compute the facial embedding for the faces 
	encoding = face_recognition.face_encodings(image, [(0, w, h, 0)], num_jitters=5, model="large")[0]

	# add the facial encoding and 
	# the label to the lists
	encodings.append(encoding)
	labels.append(sub_id)

	total_images += 1

print("[INFO] Total samples: {}".format(total_images))

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")

# visual data encodings
data = {"encodings":encodings, "names":labels}
f = open("embeddings/embeddings_{}.pickle".format(args["size"]), "wb")
f.write(pickle.dumps(data))
f.close()