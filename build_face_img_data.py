# import the necessary packages
from imutils import paths
import glob
import numpy as np
import cv2
import os
import argparse
import insightface
from speakingfacespy.imtools import make_dir

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", required=True,
                help="path to the SpeakingFaces dataset")
ap.add_argument("-v", "--save", required=True,
                help="path to the destination")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
                help="a minimum probability for the model to filter weak detections")
ap.add_argument("-i", "--sub_range",  nargs='+', type=int, default=(1, 142),
                help="range of subjects")
ap.add_argument("-m", "--model", type=str, default="retina",
               help="face detection model: dnn/hog/retina")
ap.add_argument("-n", "--frame", type=int, default=5,
                help="process only n evenly placed frames, if n=-1 then process every frame")
ap.add_argument("-l", "--log", type=str, default="log",
                help="the name of the log file that contains undetected faces")
ap.add_argument("-g", "--missing", type=str, default="missing",
                help="the name of the log file that contains missing frames")
ap.add_argument("-s", "--show", type=int, default=0,
                help="visualize extracted faces")
args = vars(ap.parse_args())

# in case "dnn" was selected, load the serialiazed dnn face detector from disk
if args["model"] == "dnn":
    print("[INFO] Loading dnn face detector...")
    face_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt",
                                        "models/res10_300x300_ssd_iter_140000.caffemodel")

# in case "retina" was selected, load the RetinaFace model
if args["model"] == "retina":
    print("[INFO] Loading RetinaFace face detector...")
    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id=-1, nms=0.4)

# initialize a path to our dataset
path_to_dataset = args["dataset"]

# initialize a path to the destination
path_to_dest = args["save"]

# initialize a total number of subjects, trials per subject,
trial_ids = 2
sub_id_str = args["sub_range"][0]
sub_id_end = args["sub_range"][1]

# initialize counters to count number of training and testing samples
total_num_audio = 0
total_num_rgb = 0
total_num_thr = 0

# open the log files
make_dir(path_to_dest)
f = open(os.path.join(path_to_dest, args["log"]+".txt"), "a+")
m = open(os.path.join(path_to_dest, args["missing"]+".txt"), "a+")

# loop over the subjects 1...142
for sub_id in range(sub_id_str, sub_id_end + 1):
    # loop over the trials 1..2
    for trial_id in range(1, trial_ids + 1):
        # construct paths to the folders with visual and thermal images, and audio files
        if sub_id <= 100:
            set_name = "train"
        elif sub_id > 100 and sub_id <= 120:
            set_name = "valid"
        else:
            set_name = "test"
        rgb_dest_path = os.path.join(path_to_dest
                                     , "{}_data/sub_{}/trial_{}/rgb_face_cmd".format(set_name, sub_id, trial_id))
        thr_dest_path = os.path.join(path_to_dest
                                     , "{}_data/sub_{}/trial_{}/thr_face_cmd".format(set_name, sub_id, trial_id))
        make_dir(rgb_dest_path)
        make_dir(thr_dest_path)

        rgb_org_path = os.path.join(path_to_dataset,
                                "{}_data/sub_{}/trial_{}/rgb_image_cmd_aligned/".format(set_name, sub_id, trial_id))
        thr_org_path = os.path.join(path_to_dataset,
                                "{}_data/sub_{}/trial_{}/thr_image_cmd/".format(set_name, sub_id, trial_id))
        audio_org_path = os.path.join(path_to_dataset,
                                "{}_data/sub_{}/trial_{}/mic1_audio_cmd_trim/".format(set_name, sub_id, trial_id))

        # loop over the all audio commands for the current trial
        audio_file_paths = list(paths.list_files(audio_org_path))

        total_num_audio+=len(audio_file_paths)

        if len(audio_file_paths) == 0:
            print("[INFO] No audio files for sub_id={} trial_id={}".format(sub_id, trial_id))
            break

        for audio_file_path in audio_file_paths:
            # extract the filename
            audio_trim_filename = audio_file_path.split(os.path.sep)[-1]
            command_id = audio_trim_filename.split('_')[:-1][4]
            related_img_filenames = '_'.join(audio_trim_filename.split('_')[:-1])
            
            # find all related rgb images using Unix style pathname pattern matching
            rgb_image_file_paths = glob.glob(os.path.join(rgb_org_path, related_img_filenames + '_*.png'))

            if len(rgb_image_file_paths) == 0:
                print("[INFO] No rgb_image_file_paths for audio_trim_filename={}".format(audio_trim_filename))
                break

            # process on the passed number frames, evenly paced intervals or all existing frames
            num_frames = len(rgb_image_file_paths)

            if args["frame"] != -1:
                iter_str, iter_end = 0, args["frame"]
            else:
                iter_str, iter_end = 1, num_frames+1

            for i in range(iter_str, iter_end):
                if args["frame"] != -1:
                    partition = np.floor((num_frames - 1) / (args["frame"] - 1))
                    idx = int(1+i*partition)
                else:
                    idx = i

                thr_file = related_img_filenames+"_{}_1.png".format(idx)
                rgb_file = related_img_filenames+"_{}_3.png".format(idx)
                thr_file = os.path.join(thr_org_path, thr_file)
                rgb_file = os.path.join(rgb_org_path, rgb_file)

                if os.path.exists(rgb_file) == False:
                    print("[INFO] The rgb image={} was not found!".format(rgb_file.split('/')[-1]))
                    m.write(rgb_file+"\n")
                    continue
                elif os.path.exists(thr_file) == False:
                    print("[INFO] The thr image={} was not found!".format(thr_file.split('/')[-1]))
                    m.write(thr_file + "\n")
                    continue

                thr_image = cv2.imread(thr_file)
                rgb_image = cv2.imread(rgb_file)

                # detect faces in the rgb image and return corresponding bounding boxes
                if args["model"] == "hog":
                    rgb_boxes = face_recognition.face_locations(rgb_image, model=args["model"])
                elif args["model"] == "dnn":
                    rgb_boxes = face_region_extractor(face_net, rgb_image, args["confidence"])
                else:
                    rgb_boxes,_ = model.detect(rgb_image, threshold=args["confidence"], scale=1.0)

                # if at least one face is detected
                if len(rgb_boxes) == 0:
                    print("[INFO] No bounding boxes have been detected for {}.".format(rgb_file))
                    f.write(rgb_file+"\n")
                else:
                    # assume that only one person was detected and extract (x,y)
                    # coordinates of the corners of the bounding box
                    (startX, startY, endX, endY) = [int(rgb_boxes[0][i]) for i in range(4)]
                    print("[INFO] Preprocessing sub_id={} trial_id={} command_id={} images: {}/{}".format(sub_id, trial_id, command_id, idx, num_frames))
                    
                    # crop the detected faces
                    if startX < 0 or startY < 0 or endX < 0 or endY < 0:
                        print("[INFO] on of the coordinates of the bounding box is negative. See below the update")
                        print((startX, startY), (endX, endY))
                        (startY, endX, endY, startX) = (max(0, startY), max(0, endX), max(0, endY), max(0, startX))
                        print((startX, startY), (endX, endY))
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
                        img = np.hstack([rgb_image, thr_image, rgb_copy])
                        img = cv2.rectangle(img, (startX,startY), (endX,endY), (1, 0, 0) , 3)
                        cv2.imshow("Output", img)
                        key = cv2.waitKey(0) & 0xFF

                        # if the 'q' key is pressed, stop the loop
                        if key == ord("q"):
                            break

                    cv2.imwrite(os.path.join(rgb_dest_path
                                             , related_img_filenames + "_{}_7.png".format(idx))
                                             , rgb_face)

                    total_num_rgb+=1
                    cv2.imwrite(os.path.join(thr_dest_path
                                             , related_img_filenames + "_{}_6.png".format(idx))
                                             , thr_face)
                    total_num_thr+=1
f.close()
m.close()
print("[INFO] Processing is done! Total number of audio files: {}, rgb faces: {}, and thr faces: {} ".format(total_num_audio
                                                                                                            , total_num_rgb
                                                                                                            , total_num_thr
                                                                                                            ))

                                                                                                                               

