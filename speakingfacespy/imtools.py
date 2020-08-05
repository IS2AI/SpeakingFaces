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


def face_region_extractor(face_net, visible_image, threshold):
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

			# extract corners of the bounding box
			(startX, startY, endX, endY) = box.astype("int")

			# add the box to the list according to
			# dlib's rect format
			boxes.append((startY, endX, endY, startX))
	
	return boxes


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

def align_rgb_image(dy, dx, thr, rgb):
	
    print("[INFO] aligning the visible image with its thermal pair")

    ptsA = np.array([[399 + dx, 345 + dy], [423 + dx, 293 + dy], [293 + dx, 316 + dy], [269 + dx, 368 + dy]])
    ptsB = np.array([[249, 237], [267, 196], [169, 214], [151, 254]])
    
    # estimate a homography matrix to warp the visible image
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 2.0)
    
    # grab a size of the thermal image 
    (H_thr, W_thr) = thr.shape[:2]
    
    # warp the rgb image to align with the thermal image
    rgb = cv2.warpPerspective(rgb, H, (W_thr, H_thr), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
   
    return rgb

def single_face_detector(image, threshold):
	print("[INFO] DNN model is chosen, loading the face and landmark predictors ...")
	face_net = cv2.dnn.readNetFromCaffe(str(os.path.abspath("models/deploy.prototxt.txt")), 
		str(os.path.abspath("models/res10_300x300_ssd_iter_140000.caffemodel")))                      
	
	# get the visible image size
	(h, w) = image.shape[:2]

	#  create an image blob to pass it through the network
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

	# pass the blob through the network and obtain the detections and corresponding predictions 
	face_net.setInput(blob)
	detections = face_net.forward()
	result = []
	# loop over the detections
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
		
		# filter out weak detections by ensuring the 'confidence' is greater than the minimum probability
		if confidence > threshold:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			result = [(startY, endX, endY, startX)]
	return result

def get_face_location_landmarks(image, threshold, model):
	print("[INFO] retrieving the face location and landmarks")

	if model != "dnn":
		face_locations = face_recognition.face_locations(image, model = model)
	else:
		face_locations = single_face_detector(image, threshold)

	if face_locations:
		face_landmarks = face_recognition.face_landmarks(image, face_locations)
		return face_landmarks, face_locations
	else:
		return [], []


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
		os.makedirs(dirName, exist_ok = True)
		print("[INFO] Directory " , dirName ,  " created")
	else:
		print("[INFO] Directory " , dirName ,  " already exists") 
