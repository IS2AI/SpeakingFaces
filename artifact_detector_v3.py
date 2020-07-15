# import the necessary packages
import csv

from imutils import paths
from speakingfacespy.imtools import make_dir
from speakingfacespy.imtools import get_face_location_landmarks
from speakingfacespy.imtools import align_rgb_image
from skimage.measure import compare_ssim
import imutils
import numpy as np
import pandas as pd
import cv2
import argparse
import os


def append_to_csv(filename, list_of_artifacts):
    with open(filename, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(list_of_artifacts)


def get_similarity_score(thr_image_current, thr_image_previous):
    # convert images to grayscale
    previous_thr_image_gray = cv2.cvtColor(thr_image_previous, cv2.COLOR_BGR2GRAY)
    current_thr_image_gray = cv2.cvtColor(thr_image_current, cv2.COLOR_BGR2GRAY)

    # Compute the Structural similarity index (SSI)
    (score, diff) = compare_ssim(previous_thr_image_gray, current_thr_image_gray, full=True)

    # Convert the image differences
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # You can return both score and the thresh (difference image), but for this purpose we only need the score
    # return score, thresh
    return score


def check_for_blurriness(rgb_image, rgb_filename, rgb_artifacts_file):
    # focus measure of the image computed from the variance of the Laplacian
    focus_measure = cv2.Laplacian(rgb_image, cv2.CV_64F).var()

    # print("the processed frame: {}".format(thr_image_filepath.split(os.path.sep)[-1]))
    if focus_measure < 10:
        print("Image: {} is blurry with focus measure of: {}".format(rgb_filename,
                                                                     focus_measure))

        window_name = "Sub:{} Trial:{} Session:{} Pos:{} Command:{} Frame:{}".format(sub_id, trial_id,
                                                                                     session_id, pos_id,
                                                                                     command_id, frame_id)
        if args['show']:
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name, 20, 400)
            cv2.imshow(window_name, rgb_image)

            key = cv2.waitKey(0) & 0xFF

            # if the 'q' key is pressed, stop the loop, exit from the program
            if key == ord("q"):
                pass
        append_to_csv(rgb_artifacts_file, [sub_id, rgb_filename, focus_measure, 'blurry'])


# return true if the chin is not cutoff
def chin_checker(rgb, rgb_filename, rgb_artifacts_file):
    # get the bounding box of the face
    face_landmarks, face_locations = get_face_location_landmarks(rgb, args["confidence"], model=args["face_detector"])

    if not (face_locations):
        print("[INFO] CAN'T GET THE LOCATION OF THE FACE!!!! STOP AND CHANGING THE FACE DETECTOR")
        face_landmarks, face_locations = get_face_location_landmarks(rgb, args["confidence"], model='dnn')
        if not (face_locations):
            print("[INFO] CAN'T GET THE LOCATION OF THE FACE!!!!")
            return 0

    if not (face_landmarks):
        print("[INFO] CAN'T GET THE FACE LANDMARKS!!!! STOP AND CHANGE THE FACE DETECTOR")
        face_landmarks, face_locations = get_face_location_landmarks(rgb, args["confidence"], model='dnn')
        if not (face_landmarks):
            print("[INFO] CAN'T GET THE LANDMARKS OF THE FACE!!!!")
            return 0

    face_location = face_locations[0]
    shape_chin, shape_mouth = face_landmarks[0]['chin'], face_landmarks[0]['top_lip']
    startX, startY, endX, endY = shape_chin[2][0], shape_chin[4][1], shape_chin[16][0], shape_chin[7][1]

    # if the chin region was not detected properly then save and skip this frame
    if startX <= 0 or endX <= 0:
        print("[INFO] CAN'T EXTRACT THE LIP REGION. NEGATIVE COORDINATES FOR THE BOUNDING BOX!!!")
        return 0
    chin = rgb[startY:endY, startX:endX]
    h, _, _ = rgb.shape
    if endX >= h and endY >= h and (endY - h) > 10:
        if args['show']:
            rgb_copy = rgb.copy()
            cv2.rectangle(rgb_copy, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv2.rectangle(rgb_copy, (face_location[3], face_location[0]), (face_location[1], face_location[2]),(0, 255, 0), 2)
            cv2.imshow("", rgb_copy)

            key = cv2.waitKey(0) & 0xFF
            if key== ord('q'):
                return 1
        else:
            append_to_csv(rgb_artifacts_file, [sub_id, rgb_filename, "", 'Chin cut'])
            print("[INFO] The chin is cut in the file: {}".format(rgb_filename))

    return 1


# parse the provided arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--dataset", required=True, help="path to the dataset")
ap.add_argument("-i", "--sub_info", nargs='+', type=int, help="subID(1,...,142) trialID (1,2) posID(1,...,9)")
ap.add_argument("-s", "--show", type=int, default=0, help="visualize or not a preliminary result of alignment")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
                help="the minimum probability to filter out weak face detections")
ap.add_argument("-f", "--face_detector", type=str, default="hog", help="model for face detection: hog/cnn/dnn/none")

args = vars(ap.parse_args())


# grab the path to our dataset
sub_trial_dataset_path = "{}sub_{}".format(args["dataset"], args["sub_info"][0]) + os.path.sep + "trial_{}".format(args["sub_info"][1])

# get the path to the thermal images in our dataset and sort them in order
thr_image_filepaths = list(paths.list_images(sub_trial_dataset_path + os.path.sep + "thr_image_cmd" + os.path.sep))
thr_image_filepaths.sort(key=lambda x: ( int(os.path.split(x)[-1].split("_")[3]), int(os.path.split(x)[-1].split("_")[4]), int(os.path.split(x)[-1].split("_")[5])))

# get path to the directory that holds rgb images
rgb_image_path = sub_trial_dataset_path + os.path.sep + "rgb_image_cmd" + os.path.sep

first_loop = 1
freezing_frame_counter = 0


h1 = ['sub_id', 'freeze_start_filename', 'freeze_end_filename', 'score', 'artifact']
h2 = ['sub_id', 'filename', 'focus_measure', 'artifact']
thr_artifacts_file = 'thr_artifacts.csv'
rgb_artifacts_file = 'rgb_artifacts.csv'
if not os.path.exists(thr_artifacts_file):
    append_to_csv(thr_artifacts_file, h1)
    append_to_csv(rgb_artifacts_file, h2)

frame_diff = 0

# loop over thermal images
for thr_image_filepath in thr_image_filepaths:

    # extract the current image info
    sub_id, trial_id, session_id, pos_id, command_id, frame_id = thr_image_filepath.split(os.path.sep)[-1].split("_")[
                                                                 -7:-1]
    # Get the path to the corresponding rgb image pair
    rgb_image_filepath = "{}{}_{}_{}_{}_{}_{}_2.png".format(rgb_image_path, sub_id, trial_id, session_id, pos_id, command_id, frame_id)
    rgb_filename = rgb_image_filepath.split(os.path.sep)[-1]

    # Artifact #1 blurrines in RGB images.
    '''
    Why do you need it in thermal loop?
    '''
    rgb_image = cv2.imread(rgb_image_filepath)
    check_for_blurriness(rgb_image, rgb_filename, rgb_artifacts_file)
    

    # Artifact #2 cropped chin in RGB images.
    is_chin = chin_checker(rgb_image, rgb_filename, rgb_artifacts_file)

    if not is_chin:
        if not args['show']:
            append_to_csv(rgb_artifacts_file, [sub_id, rgb_filename, '', 'no location or landmarks'])
        print("[INFO]Could not get the location and/or landmarks for the: {}".format(rgb_filename))
    elif is_chin and args['show']:
        break

    # Artifact #3 freezing in thermal images.
    current_thr_image = cv2.imread(thr_image_filepath)

    # making sure we have a pair of thermal images to compare
    if first_loop:
        previous_thr_file = thr_image_filepath
        previous_thr_image = current_thr_image
        previous_frame_id = frame_id
        previous_command = command_id
        previous_pos_id = pos_id
        first_loop = 0
        continue

    score = get_similarity_score(current_thr_image, previous_thr_image)

    if score == 1:
        print("Image {} and {} are similar with a Structural Similarity Index (SSI) of: {}".format(
            previous_thr_file.split(os.path.sep)[-1], thr_image_filepath.split(os.path.sep)[-1], score))

        # print("The subtraction of the pixels: {}".format((previous_thr_image-current_thr_image)))

        image_window_name = "Sub:{} Trial:{} Session:{} Pos:{} Command:{} Frame:{}".format(sub_id, trial_id,
                                                                                           session_id, pos_id,
                                                                                           command_id, frame_id)
        if args['show']:
            cv2.namedWindow(image_window_name)
            cv2.moveWindow(image_window_name, 20, 400)
            cv2.imshow(image_window_name, np.hstack([previous_thr_image, current_thr_image]))

            key = cv2.waitKey(0) & 0xFF

            # if the 'q' key is pressed, stop the loop, exit from the program
            if key == ord("q"):
                break
        else:
            # the frames are similar, check in the current and previous frames 
            # are indeed consecuitive and belong to the same commands
            frame_diff = int(frame_id) - int(previous_frame_id)
            if frame_diff == 1 and int(previous_command) == int(command_id):
                if freezing_frame_counter == 0:
                    freeze_start_filename = previous_thr_file.split(os.path.sep)[-1]
                freezing_frame_counter = freezing_frame_counter + 1
            

    if frame_diff != 1 or int(previous_command) != int(command_id):
        freezing_frame_counter = 0

    if freezing_frame_counter >= 52 and int(command_id) == int(previous_command) and previous_pos_id == pos_id:
        append_to_csv(thr_artifacts_file, [sub_id, freeze_start_filename, thr_image_filepath.split(os.path.sep)[-1], score, 'freezing'])
        freezing_frame_counter = 0

    previous_thr_file = thr_image_filepath
    previous_thr_image = current_thr_image
    previous_frame_id = frame_id
    previous_command = command_id
    previous_pos_id = pos_id

    # print("the processed frame: {}".format(thr_image_filepath.split(os.path.sep)[-1]))
