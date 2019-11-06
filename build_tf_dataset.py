#!/usr/bin/env python

import os
import numpy as np
import tensorflow as tf
import random
from logging import log, INFO

AUTOTUNE = tf.data.experimental.AUTOTUNE


tf.executing_eagerly()

BATCH_SIZE = 32


def get_label(file_path):
    """This function returns the label and the name of a file
    Args:
          file_path (tf.Tensor) :  file path tensor
    Returns:
          class_labels (np.array) : A numpy boolean array
          file_path (str) : The file path
    """
    # convert the path to a list of path components
    class_label = tf.strings.split(file_path, '_')
    #  Image_0_1_8.png
    # The second to last is the class-directory
    print("label")
    print(class_label)
    # class_label = tf.constant(["0"])
    # return class_label[1]
    return class_label[1] == np.array(["0", "1"], dtype='<U10'), file_path

def decode_img(img):
    """This function takes in an image path and returns the image tensor
    Args:
          img (str) : The path of the image
    Returns:
          tf.Tensor : The image resized to 224, 224
    """
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    # resize the image to the desired size.
    return tf.image.resize(img, [224, 224])


def process_path(file_path, return_image_names=False):
    """This function processes a single file path by extracting
    the class labels and converting the image to a tensor
    Args:
          file_path (str) : The input file path
          return_image_names (bool) : The return the file name or not, used for debuging
          purposes
    """
    label, image_name = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    if return_image_names:
        return img, label, image_name
    return img, label


def tf_dataset(images_folder, return_image_name=False):
    all_files = list(map(lambda x: os.path.join(images_folder, x), os.listdir(images_folder)))
    random.shuffle(all_files)
    num_images = len(all_files)
    train_files = all_files[:int(num_images * 0.8)]
    test_files = all_files[int(num_images * 0.8):]

    log(INFO, "Total number of training images {}".format(len(train_files)))
    log(INFO, "Total number of test images {}".format(len(test_files)))
    #print(all_files)
    train_ds = tf.data.Dataset.list_files(train_files)# "{}*.png".format(images_folder)
    test_ds = tf.data.Dataset.list_files(test_files)
    for f in train_ds.take(5):
        print(f.numpy())

    for f in test_ds.take(5):
        print(f.numpy())
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = prepare_for_training(train_ds)

    test_ds = test_ds.map(process_path, return_image_name=return_image_name, num_parallel_calls=AUTOTUNE)
    test_ds = prepare_for_training(test_ds)
    return train_ds, test_ds


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


if __name__ == "__main__":
    tf_dataset()
