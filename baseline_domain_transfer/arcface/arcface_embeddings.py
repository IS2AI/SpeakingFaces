# import the necessary packages
import argparse
import insightface
from matplotlib.image import imread
import pandas as pd
from imutils import paths
import os
import pickle
import numpy as np

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to the images")
args = vars(ap.parse_args())

# construct a list of paths to the real visual images
image_paths = list(paths.list_images(args["images"]))

# construct a list of names of real visual trial 1 images
real_trial_1 = []

for image_name in image_paths:
	name = image_name.split('/')[-1]
	ft = name.split('_')

	if ft[1] == "1":
		real_trial_1.append(name)

# get Arcface model by its name
model = insightface.model_zoo.get_model('arcface_r100_v1')

# prepare the environment, to use CPU to recognize the incoming face image
model.prepare(ctx_id = -1)

# HERE code for detecting and aligning images

# calculate embeddings and save them into a dataframe
df = pd.DataFrame(columns = ["image_name",
                                 "sub_id",
                                 "embedding"])
df["image_name"] = real_trial_1
df["sub_id"] = [image_name.split('_')[0] for image_name in real_trial_1]
print("[INFO] Calculating the embeddings...")
df["embedding"] = [(model.get_embedding(imread(os.path.join(args["images"], image_name)))).flatten() for image_name in real_trial_1]

# create a folder with embeddings
if not os.path.isdir("embeddings"):
	os.mkdir("embeddings")

# save the dataframe to pickle file
df.to_pickle("embeddings/real_trial_1_emb.pkl")
print("[INFO] Embeddings were successfully calculated and saved.")





