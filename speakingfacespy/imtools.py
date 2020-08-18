# import the necessary packages
import numpy as np
import cv2 
import os


def face_region_extractor(face_net, visible_image, threshold):
	# grab the size of the visible image 
	(h, w) = visible_image.shape[:2]

	#  create an image blob to pass it through the network
	blob = cv2.dnn.blobFromImage(cv2.resize(visible_image, (300, 300)), 1.0,
		(300, 300), (104.0, 107.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# corresponding predictions 
	face_net.setInput(blob)
	detections = face_net.forward()

	# create a container to store bounding boxes
	boxes = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence associated with the prediction 
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the 'confidence' is
		# greater than the minimum probability
		if confidence > threshold:
			# compute the (x, y)-coordinates of the bounding box
			# for the object and add it to the list 
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

			# extract corners of the bounding box
			(startX, startY, endX, endY) = box.astype("int")

			# add the box to the list according to
			# dlib's rect format
			boxes.append((startY, endX, endY, startX))
	
	return boxes


def make_dir(dirName):
	# Create a target directory & all intermediate 
	# directories if they don't exists
	if not os.path.exists(dirName):
		os.makedirs(dirName, exist_ok = True)
		print("[INFO] Directory " , dirName ,  " created")
	else:
		print("[INFO] Directory " , dirName ,  " already exists") 
