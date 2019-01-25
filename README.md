# OpenCV

Simple OCR implementation using C++ and OpenCV. 
Uses CMake as a build system.

Libraries:
OpenCV

The application works in 2 stages:
1. Generates trainig data from input images. If the training data already exists goes straight to the second stage.
2. Trains the kNN algorithm using the training data obtained in the first stage. Then uses the kNN algorithm implementation in OpenCV to classify the input characters/digits.
