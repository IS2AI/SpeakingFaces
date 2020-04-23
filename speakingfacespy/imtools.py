# import the necessary packages
from imutils import face_utils
import face_recognition
import imutils
import cv2 
import dlib
import os
import numpy as np

# obtain a path to a thermal image
def path_to_thermal_image(rgbImagePath, dataset_path, thermal_image_folder):
	# modify the visible image file name to obtain 
	# the corresponding thermal video file name
	rgb_file = list(rgbImagePath.split(os.path.sep)[-1])
	rgb_file[-5] = '1'

	# return the path to the thermal video 
	return dataset_path + thermal_image_folder + ''.join(rgb_file)


# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


def face_region_extractor(face_net, visible_image, thermal_image, threshold):
	# convert bgr to grayscale image
	gray = cv2.cvtColor(visible_image, cv2.COLOR_BGR2GRAY)

	# get the visible image size
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
		# extract the confidence associated with the
		# prediction 
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the 'confidence' is
		# greater than the minimum probability
		if confidence > threshold:
			# compute the (x, y)-coordinates of the bounding box
			# for the object and add it to the list 
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			boxes.append(box)
	
	# if at least one face was detected
	# then apply non maximum suppression
	if boxes:
		# apply non-maximum suppresion
		[[startX, startY, endX, endY]] = non_max_suppression_fast(np.asarray(boxes, dtype=np.float32), overlapThresh=0.3)

		# return bbox coordinates
		return (startX, startY, endX, endY)

	# otherwise return Nones
	else:
		return (None, None, None, None)


def lip_region_extractor(face_net, visible_image, thermal_image, threshold, dnn_mode=False):
	# convert bgr to grayscale image
	gray = cv2.cvtColor(visible_image, cv2.COLOR_BGR2GRAY)

	# apply the face detector to the visible image 
	(startX, startY, endX, endY) = face_region_extractor(face_net, visible_image, thermal_image, threshold)

	# if at least one was face was detected
	# then extract its facial ladnmarks
	if startX is not None:
		# determine the facial landmark for the face region, then
		# convert the facial landmark (x, y)-coordinates to a Numpy
		# array
		if dnn_mode: 
			rect = dlib.rectangle(startX, startY, endX, endY)
			#shape = landmark_predictor(gray, rect)
			#shape = face_utils.shape_to_np(shape)
			shape = face_recognition.face_landmarks(gray, [(startY, endX, endY, startX)])
		else:	
			#shape = face_recognition.face_landmarks(gray, [(startY, endX, endY, startX)])
			shape = face_recognition.face_landmarks(gray)
		
		shape = shape[0]['chin']

		return shape
	
	# otherwise return Nones
	else:
		return (None, None, None, None)


def get_homography_matrix(M, dx, dy, N=40):
	# define matching point coordinates between two images
	ptsA = M[:N,3:]
	ptsA[:,0] = ptsA[:,0] + dx
	ptsA[:,1] = ptsA[:,1] + dy

	ptsB = M[:N,1:3]

	# estimate a homography matrix to warp the visible image
	(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 2.0)

	return H


def make_dir(dirName):
	# Create target directory & all intermediate directories if don't exists
	if not os.path.exists(dirName):
		os.makedirs(dirName)
		print("Directory " , dirName ,  " Created ")
	else:
		print("Directory " , dirName ,  " already exists") 
