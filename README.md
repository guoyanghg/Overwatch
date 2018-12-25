# Overwatch

## Prerequisites
* PyCharm:  https://www.jetbrains.com/pycharm/download/#section=windows
* Python 3.5.2:  https://www.python.org/downloads/release/python-352/
* Tensorflow 1.1.0:  https://www.tensorflow.org/install/
## Installing

* Copy this repo directly to an empty PyCharm project folder
* File->open 

## Training

Path parameters are in the ImageReader.py file.

Training Script: OWModel.py

## Test and Result

Testing script: OWExample.py

## Details
* Establish and label the data sets
Using the custom mode and screenshots in the game, we collected 5 heroes: Reaper, Hanzo, Reinhardt, Mercy, and Genji, along with a total of 500 images (100 per hero). We used the player's crosshair as the origin in order to establish a two-dimensional coordinate system. We divided the screen into four quadrants, and used the combination of two space unit vectors to mark the enemy's direction. For example, the following picture shows the location of the Reaper. Is (+1, +1). Similarly, the possible values of the remaining flags are (+1, 0), (+1, -1), (0, 0), (0, -1), (-1, +1), (-1, 0 ), (-1, -1).

![image](http://github.com/guoyanghg/Overwatch/readmeim.bmp)


* Model design
The function of feature extraction is implemented by using three convolutional layers. After the convolution, the two heads are separated, and each head completes the corresponding classification task. For the enemy identification (Classifier-C), since the mark adopts one-hot form, the loss function selects the cross-entropy. For direction identification (Classifier-D), the loss function selects the mean square error (MSE).



