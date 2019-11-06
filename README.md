# Indoor/Outdoor classification

This repository containes training and inference scripts to classify
images as indoor or outdoor. The training data is obtained from a subset of videos from the
[youtube-8m](https://research.google.com/youtube8m/) dataset. Select entities are used to download videos
in the indoor (0) and the outdoor (1) category.

## Data preparation

Entities for indoor and outdoor videos are first selected.
Videos from these entities are downloaded.

The following entities are chosen for indoor and outdoor image labels

* indoor: bedroom, bathroom, restaurant, room
* outdoor: ocean, mountains, city, tree


Images are extracted from video frames. Images are sampled at a 5 interval leaving out the first 10 seconds of the video.
The inages are resized to a size of 224, 224, 3 (width, height, channels). This is the input to the classifier for training
and inference.

The images are split randomly into 80% train and 20% test.

The data pipeline is implemented using [tensorflow datasets](https://www.tensorflow.org/tutorials/load_data/images).

## Model

A pretrained VGG-16 model[1] trained on image net is used as a feature extractor. The features are then trained
on a fully connected neural network using 256 units and relu activations.

Training is performed using the Adam optimizer and the final model is chosen based on best accuracy on the test set.

## Usage

```
$ ./main.py --help
usage: main.py [-h] [--download-videos] [--videos-path VIDEOS_PATH]
               [--build-image-data] [--images-path IMAGES_PATH] [--train]
               [--infer] [--model-path MODEL_PATH] [--input-image INPUT_IMAGE]

optional arguments:
  -h, --help            show this help message and exit
  --download-videos     Use this flag to download videos from youtube. The
                        destination path must be provided with the --videos-
                        path argument
  --videos-path VIDEOS_PATH
                        Path of the videos folder where the videos will be
                        downloaded
  --build-image-data    Build the image dataset once the videos are
                        downloaded.
  --images-path IMAGES_PATH
                        Path where images are stored
  --train               Argumant to train the indoor/outdoor classifier
  --infer               Make predictions for an input image
  --model-path MODEL_PATH
                        Path to the model to make predictions
  --input-image INPUT_IMAGE
                        Path to the model to make predictions
```

### Download videos

Videos can be downloaded using the following script. All videos will be downloaded to the
`videos/` folder

```
./main.py --download-videos --videos-path videos/
```

### Build images dataset

Once videos are downloaded images can be extracted into the `images/` folder using the following cmmand.

```
./main.py --build-image-data --images-path images/
```

### Model training

Once the videos and the images are downloaded training can be executed using the following command

```
./main.py --train --images-path images/
```

### Make predictions given a model

```
$ ./main.py --infer --model-path model_final.h5 --input-image images/image_1_6_65.png
The image is outdoor
Indoor probability = 0.3741770386695862, Outdoor probability = 0.625822901725769
```

## References

[1] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).