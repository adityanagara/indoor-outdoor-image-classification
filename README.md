# Indoor/Outdoor classification

This repository containes training and inference scripts to classify
an image as indoor or outdoor. The dataset is downloaded from [youtube-8m](https://research.google.com/youtube8m/).

## Methods

A pretrained VGG-16 model, trained on image net is used as a feature extractor. The features are trained
on a fully connected neural network.

## Scripts