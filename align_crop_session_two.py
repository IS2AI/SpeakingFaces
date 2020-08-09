# import the necessary packages
from speakingfacespy.imtools import face_region_extractor
from speakingfacespy.imtools import make_dir
from imutils import paths
import face_recognition
import numpy as np
import cv2 
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the dataset")
ap.add_argument("-i", "--sub_info",  nargs='+', type=int,
	help="subID(1,...,142) trialID (1,2) posID(1,...,9)")
ap.add_argument("-y", "--dy",  nargs='+', type=int,
	help="a list of shifts in y axis for each position")
ap.add_argument("-x", "--dx",  nargs='+', type=int,
	help="a list of shifts in x axis for each position")
ap.add_argument("-m", "--model", type=str, default="dnn",
    help="a model for face detection: hog/dnn/cnn/none")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
    help="a minimum probability for dnn to filter out weak face detections")
ap.add_argument("-u", "--upper_bound", nargs='+', type=int,
    help="a list of upper bounds for landmarks to crop the lip for each position")
ap.add_argument("-r", "--resize",  nargs='+', type=int, default = (128,64), 
    help="resize the ROI (width, height)")
ap.add_argument("-s", "--show", type=int, default=0,
    help="visualize (1) or not (0) a preliminary result of alignment")
args = vars(ap.parse_args())

# initialize the subject, trial and 
# position IDs
sub_id = args["sub_info"][0]
trial_id = args["sub_info"][1]
pos_id = args["sub_info"][2]

# initialize a list of the upper bounds for 
# cropping the lip ROI
upper_bound = args["upper_bound"][pos_id - 1]

# initialize the bbox coordinates of the RoI
# for manual extraction 
initBB = None

# load the serialiazed dnn face detector from disk
# in case if it was selected
if args["model"] == "dnn": 
    print("[INFO] loading dnn face detector...")
    face_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", 
        "models/res10_300x300_ssd_iter_140000.caffemodel")

# initialize a path to our dataset
path_to_dataset = args["dataset"]

# construct a path to our visual images
rgb_image_path = os.path.join(path_to_dataset, "sub_{}/trial_{}/rgb_image_cmd".format(sub_id, trial_id))

# construct a path to our thermal images
thr_image_path = os.path.join(path_to_dataset, "sub_{}/trial_{}/thr_image_cmd".format(sub_id, trial_id))

# create a directory to save the aligned visual images
rgb_image_aligned_path = os.path.join(path_to_dataset, "sub_{}/trial_{}/rgb_image_cmd_aligned".format(sub_id, trial_id))
make_dir(rgb_image_aligned_path)

# create directories for extracted lip region
lip_rgb_path = os.path.join(path_to_dataset, "sub_{}/trial_{}/rgb_roi_cmd".format(sub_id, trial_id))
lip_thr_path = os.path.join(path_to_dataset, "sub_{}/trial_{}/thr_roi_cmd".format(sub_id, trial_id))
make_dir(lip_rgb_path)
make_dir(lip_thr_path)

# initialize lists of shifts
dy = args["dy"][pos_id - 1]
dx = args["dx"][pos_id - 1]

# construct arrays of matched features
# for the given position
ptsA = np.array([[399 + dx, 345 + dy], [423 + dx, 293 + dy], [293 + dx, 316 + dy], [269 + dx, 368 + dy]])
ptsB = np.array([[249, 237], [267, 196], [169, 214], [151, 254]])
    
# estimate a homography matrix
# for the given position 
(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 2.0)

# grab the path to the visual images 
rgb_image_filepaths = list(paths.list_images(rgb_image_path))

# loop over the visible images
for rgb_image_filepath in rgb_image_filepaths:
    # extract the current image info
    sub, trial, session, pos, cmd, frame = rgb_image_filepath.split(os.path.sep)[-1].split("_")[-7:-1]

    # process only the files for the given position if "show" mode is enabled. 
    if int(pos) != pos_id and args["show"]:
        continue

    print("[INFO] processing image {}".format(rgb_image_filepath.split(os.path.sep)[-1]))

    # construct a filenames of the corresponding thermal images  
    thr_file = "{}_{}_2_{}_{}_{}_1.png".format(sub, trial, pos, cmd, frame)
    thr_image_filepath = os.path.join(thr_image_path, thr_file)
    
    # load rgb and corresponding thermal image 
    rgb_image = cv2.imread(rgb_image_filepath)
    thr_image = cv2.imread(thr_image_filepath)

    # grab height and width of the thermal image 
    (h_thr, w_thr) = thr_image.shape[:2]
        
    # align rgb image with the thermal one
    rgb_image_aligned = cv2.warpPerspective(rgb_image, H, (w_thr, h_thr), 
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # make a copy of the aligned rgb image
    rgb_image_aligned_copy = rgb_image_aligned.copy()

    # if we want automatically extract RoI
    if args["model"] != "none":
        # detect faces in the rgb image and return 
        # corresponding bounding boxes
        if args["model"] != "dnn":
            # apply dlib's face detector
            rgb_boxes = face_recognition.face_locations(rgb_image_aligned, model=args["model"])
        else:
            # apply OpenCV's face detector
            rgb_boxes = face_region_extractor(face_net, rgb_image_aligned, args["confidence"])

        # if at least one face is detected
        if len(rgb_boxes):
            # assume that only one person was detected and extract (x,y) 
            # coordinates of the bbox
            rgb_box = rgb_boxes[0]
            (face_startY, face_endX, face_endY, face_startX) = rgb_box

            # check if the was detected properly 
            if face_startX < 0 or face_startY < 0:
                print("[INFO] INAPPROPRIATE COORDINATES FOR THE BOUNDING BOX!!!")
                break

            # extract facial landmarks
            rgb_landmark = face_recognition.face_landmarks(rgb_image_aligned, rgb_boxes)[0]

            # extract coordinates for the chin ROI
            chin_roi = rgb_landmark['chin']
            (lip_startX, lip_startY, lip_endX, lip_endY) = (chin_roi[2][0], chin_roi[upper_bound][1], chin_roi[14][0],chin_roi[6][1])

            # slightly change coordinates of the lip ROI
            # for some positions
            if pos > 6:
                lip_startX = lip_startX - 20 
                lip_endX = lip_endX - 20
            elif pos > 3:
                lip_startX = lip_startX + 20
                lip_endX = lip_endX + 20

            # draw landmarks if visualization is enabled
            if args["show"]:
                # draw landmarks
                for (x_l, y_l) in chin_roi:
                    cv2.circle(rgb_image_aligned_copy, (x_l, y_l), 2, (0, 255, 0), -1)
                    cv2.circle(thr_image, (x_l, y_l), 2, (0, 255, 0), -1)

    else:
        # select the RoI manually
        if initBB is None:
            initBB = cv2.selectROI("Frames", rgb_image_aligned, fromCenter=False,
            showCrosshair=True)
            (x, y, w, h) = initBB
            (lip_startX, lip_startY, lip_endX, lip_endY) = (x, y, x + w, y + h)
            
    # crop the detected faces 
    rgb_lip = rgb_image_aligned[lip_startY:lip_endY, lip_startX:lip_endX]
    thr_lip = thr_image[lip_startY:lip_endY, lip_startX:lip_endX]

    # resize the lips
    rgb_lip = cv2.resize(rgb_lip, args["resize"], interpolation = cv2.INTER_AREA)
    thr_lip = cv2.resize(thr_lip, args["resize"], interpolation = cv2.INTER_AREA)

    # if visualization is enabled
    if args["show"]:
        # draw the ROI
        cv2.rectangle(rgb_image_aligned_copy, (lip_startX, lip_startY), (lip_endX, lip_endY), (0, 0, 255), 2)
        cv2.rectangle(thr_image, (lip_startX, lip_startY), (lip_endX, lip_endY), (0, 0, 255), 2)
        
        # show the frames
        cv2.imshow("Frames", np.hstack([rgb_image_aligned_copy, thr_image]))
        cv2.imshow("Lips", np.hstack([rgb_lip, thr_lip]))
        key = cv2.waitKey(0) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break
    else:
        # construct filenames to save the image and the ROIs
        rgb_aligned_filename = "{}_{}_{}_{}_{}_{}_3.png".format(sub, trial, session, pos, cmd, frame) 
        rgb_lip_filename = "{}_{}_{}_{}_{}_{}_4.png".format(sub, trial, session, pos, cmd, frame) 
        thr_lip_filename = "{}_{}_{}_{}_{}_{}_5.png".format(sub, trial, session, pos, cmd, frame)

        # construct paths to save images
        rgb_aligned_path = os.path.join(rgb_image_aligned_path, rgb_aligned_filename)
        rgb_lip_path = os.path.join(lip_rgb_path, rgb_lip_filename)
        thr_lip_path = os.path.join(lip_thr_path, thr_lip_filename)

        # save the images
        cv2.imwrite(rgb_aligned_path, rgb_image_aligned)
        cv2.imwrite(rgb_lip_path, rgb_lip)
        cv2.imwrite(thr_lip_path, thr_lip)


