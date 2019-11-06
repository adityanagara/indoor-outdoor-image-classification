#!/usr/bin/env python

import argparse
import build_tf_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from PIL import Image
import numpy as np
import download_data
from logging import log, INFO


def train(images_folder="images/"):
    """This function train a classifier given labeled images
    Args:
         images_folder (str) : The directory containing the labeled images
    """
    train_dataset, test_dataset = build_tf_dataset.tf_dataset(images_folder)

    IMG_SHAPE = (244, 244, 3)
    VGG16_MODEL = tf.keras.applications.VGG16(include_top=True,
                                              weights='imagenet')

    VGG16_MODEL.trainable = False
    fc_layer_1 = tf.keras.layers.Dense(256, activation="relu")
    prediction_layer = tf.keras.layers.Dense(2, activation='softmax')
    model = tf.keras.Sequential([
        VGG16_MODEL,
        fc_layer_1,
        prediction_layer
    ])
    model_ckpt = ModelCheckpoint("model.h5",
                    save_best_only=True,
                    monitor="val_accuracy",
                    mode="max",
                    verbose=2)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])

    history = model.fit(train_dataset,
                        validation_data=test_dataset,
                        validation_steps=40,
                        epochs=10,
                        callbacks=[model_ckpt],
                        steps_per_epoch=50)

def load_model(model_file):
    IMG_SHAPE = (244, 244, 3)
    VGG16_MODEL = tf.keras.applications.VGG16(include_top=True,
                                              weights='imagenet')

    VGG16_MODEL.trainable = False
    fc_layer_1 = tf.keras.layers.Dense(256, activation="relu")
    prediction_layer = tf.keras.layers.Dense(2, activation='softmax')
    model = tf.keras.Sequential([
        VGG16_MODEL,
        fc_layer_1,
        prediction_layer
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])

    model.load_weights(model_file)

    return model

def infer(model, file_path):
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img = np.array(img)

    prediction = model.predict(img.reshape((1, 224, 224, 3)).astype("float32"))

    if prediction[0][0] > 0.5:
        print("The image is indoor")
    else:
        print("The image is outdoor")

    print("Indoor probability = {}, Outdoor probability = {}".format(prediction[0][0], prediction[0][1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-videos",
                        help="Use this flag to download videos from youtube. The destination path must be provided with the --videos-path argument",
                        action="store_true",
                        default=False)
    parser.add_argument("--videos-path",
                        help="Path of the videos folder where the videos will be downloaded",
                        type=str,
                        default="videos/")
    parser.add_argument("--build-image-data",
                        help="Build the image dataset once the videos are downloaded.",
                        action="store_true",
                        default=False)
    parser.add_argument("--images-path",
                        help="Path where images are stored",
                        type=str,
                        default="images/")
    parser.add_argument("--train",
                        help="Argumant to train the indoor/outdoor classifier",
                        action="store_true",
                        default=False)
    parser.add_argument("--infer",
                        help="Make predictions for an input image",
                        action="store_true")
    parser.add_argument("--model-path",
                        type=str,
                        help="Path to the model to make predictions")
    parser.add_argument("--input-image",
                        type=str,
                        help="Path to the model to make predictions")
    args = parser.parse_args()
    if args.download_videos:
        log(INFO, "Downloading videos")
        all_video_names = download_data.get_all_video_names()
        download_data.download_all_videos(args.videos_path, all_video_names)
    if args.build_image_data:
        log(INFO, "Extracting image frames from videos")
        download_data.get_all_images(args.videos_path, args.images_path)
    if args.train:
        train(args.images_path)
    if args.infer:
        model = load_model(args.model_path)
        infer(model, args.input_image)
