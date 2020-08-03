# import the necessary packages
from imutils import paths
from speakingfacespy.imtools import get_face_location_landmarks

import imutils
import cv2
import argparse
import os

# parse the provided arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--dataset", required=True,
	help="path to the dataset")
ap.add_argument("-i", "--sub_info",  nargs='+', type=int,
	help="subID(1,...,142) trialID (1,2)")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
    help="the minimum probability to filter out weak face detections")

args = vars(ap.parse_args())

# grab the path to the visual images in our dataset
sub_trial_dataset_path = "{}sub_{}".format(args["dataset"], args["sub_info"][0])+os.path.sep+"trial_{}".format(args["sub_info"][1])
rgb_image_filepaths = list(paths.list_images(sub_trial_dataset_path + os.path.sep+"rgb_image_cmd"))

problems = []
face_detectors = ["hog", "dnn"]
shift = 50

# creating a dictionary to keep track of the previous frame's landmarks
prev = {"hog_startX": -999, "hog_startY": -999,
        "dnn_startX": -999, "dnn_startY": -999,
        "pos": 0}

rgb_image_filepaths = sorted(rgb_image_filepaths, key=lambda i: (int(i.split("_")[-4]), int(i.split("_")[-3]), int(i.split("_")[-2])))

# loop over original visible images in the folders
for rgb_image_filepath in rgb_image_filepaths:
    # extract the current image info
    sub_id, trial_id, session_id, pos_id, command_id, frame_id = rgb_image_filepath.split(os.path.sep)[-1].split("_")[-7:-1]

    print("[INFO] processing image {}".format(rgb_image_filepath.split(os.path.sep)[-1]))

    # load rgb and corresponding thermal image
    rgb = cv2.imread(rgb_image_filepath)

    # initiate shifts to False
    is_shifted = {"hog": False, "dnn": False}

    # loop over HOG and DNN face detectors
    for face_detector in face_detectors:
        # get the boundaries of a bounding box for the lip region from the warped rgb image
        face_landmarks, face_locations = get_face_location_landmarks(rgb, args["confidence"], model=face_detector)
        # if the face detector does not work, record it to the dictionary
        if not (face_locations):
            is_shifted[face_detector] = 0
            continue
        if not (face_landmarks):
            is_shifted[face_detector] = 0
            continue

        face_location = face_locations[0]
        shape_chin, shape_mouth = face_landmarks[0]['chin'], face_landmarks[0]['top_lip']
        upper_landmark_id = 2
        (startX, startY, endX, endY) = (shape_chin[2][0], shape_chin[upper_landmark_id][1], shape_chin[14][0], shape_chin[6][1])

        # if the lip region was not detected properly then skip this frame
        if startX < 0 or startY < 0 or endX < 0 or endY < 0:
            continue

        dict_startX = face_detector + "_startX"
        dict_startY = face_detector + "_startY"

        # skipping the first valid frame of a position
        if prev["pos"] != pos_id or prev[dict_startX] == -999:
            prev["pos"] = pos_id
            prev[dict_startX] = startX
            prev[dict_startY] = startY
            continue

        # if detected, record the shift to the dictionary
        if abs(startX-prev[dict_startX]) > shift:
            is_shifted[face_detector] = True

        # updating the dictionary to keep track of the previous frame's landmarks
        prev["pos"] = pos_id
        prev[dict_startX] = startX
        prev[dict_startY] = startY

    hog = is_shifted["hog"]
    dnn = is_shifted["dnn"]
    # if all working face detectors show shifts, then record the image directory as a problem
    if (hog == True and dnn == True) or (hog == 0 and dnn == True) or (hog == True and dnn==0):
        problems.append("{}_{}_{}_{}_{}_{}_2.png".format(sub_id, trial_id, session_id, pos_id, command_id, frame_id))
        continue

# print all detected cases
print("")
for problem in problems:
    print(problem)








