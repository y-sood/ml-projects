# MNIST Digit Classification using Softmax Regression

## Problem Statement
The MNIST dataset is a benchmark dataset consisting of 28x28 pixel grayscale images of handwritten digits (0–9). The goal of this project is to build a simple yet effective digit classifier using multinomial logistic regression, also known as softmax regression. This approach serves as a fundamental baseline for comparison against more complex models, providing insight into how well a linear classifier performs on high-dimensional image data.

## Feature Description
Each image in the dataset contains 784 features, corresponding to the flattened pixel intensities of a 28x28 image. The input data is normalized by scaling pixel values from the original [0, 255] range to [0, 1]. Labels are integers from 0 to 9 representing the digit in the image. The dataset is divided into training, validation, and test sets, with stratified sampling to ensure balanced class distribution.

## Exploratory Data Analysis
The dataset is well balanced across all digit classes. Visual inspection of sample images shows clear differences in shapes and strokes between digits, although variations in handwriting do pose classification challenges. Label distributions were plotted to confirm class balance. Additionally, some training samples were visualized to develop an intuition for model difficulty and intra-class variance.

## Model Development
The model used is a softmax classifier implemented using PyTorch. A single-layer linear transformation is applied to the input features followed by the softmax function to produce class probabilities. Cross-entropy loss is used as the objective function, and the model is trained using stochastic gradient descent with momentum. The model is trained for 50 epochs, and accuracy is tracked for both training and validation sets. Early stopping based on validation performance is used to prevent overfitting.

## Model Comparison
Although only one model is implemented in this notebook, it serves as a strong linear baseline. The final softmax classifier achieves approximately 92.7% accuracy on the test set. While this is lower than more complex models like convolutional neural networks, it demonstrates that even simple linear classifiers can be highly effective on structured datasets like MNIST. The model is also computationally efficient and interpretable, making it suitable for educational and baseline purposes.
