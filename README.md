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

A pretrained VGG-16 model, trained on image net is used as a feature extractor. The features are then trained
on a fully connected neural network using relu activations.
The final model is chosen based on best accuracy on the test set.

## Script

```python
./main.py --help
```

### Download videos

