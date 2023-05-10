
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from train import load_dataset

""" Global parameters """
H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    for item in ["joint", "mask"]:
        create_dir(f"results/{item}")

    """ Load the model """
    model_path = os.path.join("files", "model.h5")
    model = tf.keras.models.load_model(model_path)

    """ Dataset """
    images = glob("test/*")
    # images = glob("/media/nikhil/New Volume/ML_DATASET/people_segmentation/images/*")
    print(f"Images: {len(images)}")

    """ Prediction """
    for x in tqdm(images, total=len(images)):
        """ Extracting the name """
        name = x.split("/")[-1]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.resize(image, (W, H))
        x = x/255.0
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        pred = model.predict(x, verbose=0)

        line = np.ones((H, 10, 3)) * 255

        """ Joint and save mask """
        pred_list = []
        for item in pred:
            p = item[0] * 255
            p = np.concatenate([p, p, p], axis=-1)

            pred_list.append(p)
            pred_list.append(line)

        save_image_path = os.path.join("results", "mask", name)
        cat_images = np.concatenate(pred_list, axis=1)
        cv2.imwrite(save_image_path, cat_images)

        """ Save final mask """
        image_h, image_w, _ = image.shape

        y0 = pred[0][0]
        y0 = cv2.resize(y0, (image_w, image_h))
        y0 = np.expand_dims(y0, axis=-1)
        y0 = np.concatenate([y0, y0, y0], axis=-1)

        line = line = np.ones((image_h, 10, 3)) * 255

        cat_images = np.concatenate([image, line, y0*255, line, image*y0], axis=1)
        save_image_path = os.path.join("results", "joint", name)
        cv2.imwrite(save_image_path, cat_images)
