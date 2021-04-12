# import the necessary packages
import argparse
import insightface
from matplotlib.image import imread
import pandas as pd
from imutils import paths
import os
import numpy as np
import pickle

def calc_cos_similarity(emb_1, emb_2):
    """Calculate cosine similarity between two embeddings

    Parameters:
    emb_1 (numpy.ndarray): Embeddings from test dataset
    emb_2 (numpy.ndarray): Embeddings from true dataset

    Returns:
    int: Returns cosine similarity between emb_1 and emb_2

   """
    return np.dot(emb_1, emb_2) / (np.sqrt(np.dot(emb_1, emb_1)) * np.sqrt(np.dot(emb_2, emb_2)))

# construct an argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	            help="path to the images")
ap.add_argument("-t", "--tol", type=float, default=0.35,
	            help="tolerance for embeddings")
ap.add_argument("-n", "--save", type=str, default="arcface_results",
                help="name of the csv file that will contain the results")
args = vars(ap.parse_args())

# construct a list of names of test images 
image_paths = list(paths.list_images(args["images"]))
test_set = []
for image_name in image_paths:
    name = image_name.split('/')[-1]
    ft = name.split('_')

    if ft[1] == "2":
        test_set.append(name)

# load the ArcFace model (pretrained ResNet-100 model)
model = insightface.model_zoo.get_model('arcface_r100_v1')

# prepare the environment, to use CPU to recognize the incoming face image
model.prepare(ctx_id = -1)

# calculate embeddings of the test images
df_test_data = pd.DataFrame(columns = ["image_name",
                                        "sub_id",
                                        "embedding"])
df_test_data["image_name"] = test_set
df_test_data["sub_id"] = [image_name.split('_')[0] for image_name in test_set]

print("[INFO] Calculating embeddings for testing...")
df_test_data["embedding"] = [(model.get_embedding(imread(os.path.join(args["images"], image_name)))).flatten() for image_name in test_set]

# load the known faces and embeddings
print("[INFO] Loading known faces and embeddings...")
df_true_emb = pd.read_pickle("embeddings/real_trial_1_emb.pkl")

# define True Positive, False Positive, and False Negative
TP = 0
FP = 0
FN = 0

# make a copy of df_test_data and add new column for predicted sub_ids
df_test_data_copy = df_test_data.copy(deep=False)
df_test_data_copy = pd.concat([df_test_data_copy, pd.DataFrame(columns=["pred_sub_id"])])

print("[INFO] Comparing the embeddings...")
for test_data_index, test_data_row in df_test_data_copy.iterrows():
    # make a copy of df_true_emb and add new column for distances
    # between true embeddings and embeddings for testing
    df_true_emb_copy = df_true_emb.copy(deep=False)
    df_true_emb_copy["distance"] = ""

    for true_emb_index, true_emb_row in df_true_emb_copy.iterrows():
        true_emb_row["distance"] = calc_cos_similarity(test_data_row["embedding"], true_emb_row["embedding"])

    # keep only those true embeddings whose
    # distance to test images is bigger than set tolerance
    df_true_emb_copy = df_true_emb_copy[df_true_emb_copy["distance"] > args["tol"]]

    # explore which subject among the remaining ones occur more frequently
    matches = df_true_emb_copy["sub_id"].value_counts().keys().tolist()

    if len(matches) == 0:
        FN += 1
        test_data_row["pred_sub_id"] = "Unknown"
    else:
        # set the most frequently occurring match to be the prediction
        test_data_row["pred_sub_id"] = matches[0]

        if test_data_row["sub_id"] != test_data_row["pred_sub_id"]:
            FP += 1
        else:
            TP += 1

# save df_test_data_copy to csv including image_name, sub_id from test set,
# and matched pred_sub_id from true features
df_test_data_copy.to_csv(args["save"], columns = ["image_name",
                                                    "sub_id",
                                                    "pred_sub_id"])

if TP == 0 & FP == 0:
    precision = "null"
    recall = "null"
    F1 = "null"
else:
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * ((precision * recall) / (precision + recall))

print("[INFO] Comparison is finished!")
print("[INFO] Tolerance={}, TP={}, FN={}, FP={}, Precision={:0.4f}, Recall={:0.4f}, F1={:0.4f}".format(args["tol"],
                                                                                                TP, FN, FP,
                                                                                                precision,
                                                                                                recall, F1))


