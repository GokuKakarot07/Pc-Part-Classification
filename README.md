# PC Part Image Classification Project

## Overview

This project aims to build an image classification model using deep learning techniques. The model is trained on a custom dataset with 14 different classes, and the images are organized in the ImageNet structure. The resolution of each image is 256x256 pixels in JPG format. The project utilizes the MobileNetV2 architecture, pre-trained on ImageNet, and fine-tuned for the specific classification task.

## Dataset Details

- **Total number of classes:** 14
- **Total number of images:** 3279
- **Resolution:** 256x256 pixels
- **Image format:** JPG


## Model Architecture

The model architecture is based on MobileNetV2, a lightweight convolutional neural network architecture that has shown excellent performance in image classification tasks.

### Base Model (MobileNetV2)

- Input shape: (224, 224, 3)
- Global Average Pooling layer
- Dense layer with 512 units and ReLU activation
- Dropout layer with a rate of 0.5
- Output Dense layer with 14 units and softmax activation

## Data Preprocessing
The dataset is organized into three subsets:

- `image/train/`: Training dataset
- `image/val/`: Validation dataset
- `image/test/`: Test dataset

### Data Augmentation

Data augmentation is applied to the training set to improve the model's generalization ability. Augmentation techniques include rotation, width and height shifting, shearing, zooming, and horizontal flipping. The `ImageDataGenerator` from TensorFlow is used for this purpose.


