# SVM-Based Tumor Classification (MATLAB)

## Problem Statement  
This project applies Support Vector Machine (SVM) techniques to classify tumors as benign or malignant using the Wisconsin Diagnostic Breast Cancer dataset. The primary objective is to construct a binary classifier that can assist in medical diagnostics by distinguishing tumor types based on measured features such as area, radius, smoothness, and more.

## Feature Description  
The dataset includes multiple features extracted from digitized images of breast tissue, excluding patient ID and label formatting. Each data point contains numeric attributes representing cell characteristics and a corresponding label: benign (1) or malignant (-1). These features are used to define a hyperplane that best separates the two classes.

## Exploratory Data Analysis  
Although detailed visualization was not the focus, performance trends and confusion matrix metrics were analyzed across various regularization parameter (C) values. Results demonstrated that class separation improves with an appropriately tuned C, with sensitivity and specificity offering insight into false negatives and positives.

## Model Development  
The SVM model was formulated as a quadratic programming (QP) optimization problem and solved in MATLAB. Slack variables were introduced to handle non-linearly separable data, and matrix constraints were constructed accordingly. The decision boundary was optimized by minimizing a combination of the margin width and the classification error. Multiple values of the regularization parameter C were tested to observe performance variation.

## Model Comparison  
For C = 1000, the model achieved an accuracy of 94.69%, with perfect sensitivity (no false negatives) and high specificity (93.10%). Further analysis showed that C = 1200 yielded the best balance across accuracy, sensitivity, and specificity. These metrics highlight the effectiveness of the SVM approach, especially in medical applications where false negatives must be minimized.
