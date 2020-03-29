# Self Driving Car

> This project is mainly focused on End-to-End Deep Learning for Self-Driving-Cars. Uses raw image data, and a convolutional neural network to drive a car autonomously in Need For Speed game.

## Sections:
* [What's New](#changelog)
* [Demo](#demo)
* [Installation](#installation)
* [Working](#working)
* [DriveNet](#DriveNet)
* [Citations](#citations)

---
## Changelog
Version 2.0
* Updated from tflearn to tensorflow 2.0
* Updated from alexnet to new architecture.
* Using minimap also as an input, along with road images.

## Demo
![demo](nfs.gif)

## Installation
Copy and paste the below code in a terminal/cmd open inside the repository folder.

`python -m pip install --user --requirement requirements.txt`

More specifically, all required modules (except the built-ins) are listed below.
```
opencv-python
numpy
psutil
pandas
sklearn
tensorflow
matplotlib
```

## Working
1. Visualizing Region of Interest - to make sure that we are capturing the areas that we want and nothing extra.
    * [Visualize screen](visualize_screen.py) can be used to see the area of the road that is being captured when recording training data.
    * [Visualize map](visualize_map.py) can be used to see the area of the minimap that is located in the bottom-left corner, that is captured when recording training data.

1. Getting Training Data - capturing raw frames along with player's inputs.
    * [Get data](get_data.py) is used to capture the ROI found in step 1. We capture both the road and the minimap per observation as features, along with the player's input as label.
    * The captured road frame is resized to (80, 200, 3)
    * The captured minimap frame is resized to (50, 50, 1)

1. Balancing the data - The raw data is balanced to avoid bias.
    * [Balance data](balance_data.py) removes the unwanted bias in the training data.
    * The raw data has most of the observations with labels for _forward_ with few observations with labels for _left_ or _right_.
    * We thus discard the excess amount of unwanted data that has label as _forward_. Keep in mind that we do __lose a lot of data__.

1. Combining the data - The balanced files can now be joined together to form one final data file.
    * [Combine data](combine_data.py) is used to join all balanced data for easier loading of data during training process.
    * All batches of balanced data is now combined together to form _final\_data.npy_ file.
    * This file has 2 images as features, with shape (80, 200, 3) and (50, 50, 1), respectively, and a one-hot-encoded label with 3 classes.

1. Training the Neural Network - [DriveNet](#drivenet) is used as the convolutional neural network.
    * [Train model](train_model.py) is used to train the network over a max of 100 epochs.
    * Training is regulated by EarlyStopping Callback, monitoring validation_loss with a patience of 3 epochs.
    * Adam optimizer is used, with learning rate set to 0.001
    * No data augmentation is done.

1. Testing the model - Final testing done in the game.
    * [Test model](test_model.py) is used to actually run the trained model and control the car real-time.

## DriveNet
Graph of [DriveNet](drivenet.py), rendered using plot_model function.

![Image](DriveNet.png)

* input_1 is the road image
* input_2 is the minimap image
* dense_2 is the output layer, which returns the probability of the car turning left, right or just go straight ahead.
* **NOTE:** This architecture is heavily inspired by the paper "Variational End-to-End Navigation and Localization" by Alexander Amini and others. Refer to the [citation](#citation) section for more details.

## Citations
1. [MIT - Intro to Deep Learning Course](https://introtodeeplearning.com/ "Go to HomePage")
1. [Variational End-to-End Navigation and Localization](https://arxiv.org/abs/1811.10119v2 "Go to arxiv page")
1. [Sentex's Python Plays GTA-V](https://github.com/Sentdex/pygta5 "Go to GitHub")
1. [Box of Hats](https://github.com/Box-Of-Hats "Github") - [getkeys.py](getkeys.py)

---
Open to suggestions. Feel free to fork this repository. If you would to use some code from here, please do give the required citations and references.
