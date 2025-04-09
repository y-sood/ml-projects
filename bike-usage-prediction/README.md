# Bikesharing Demand Prediction

## Problem Statement
Capital Bikeshare provides 24-hour bicycle-sharing services in Washington, DC, but experiences demand fluctuations that lead to shortages during peak hours. These shortages can discourage use and increase car dependency, contributing to higher CO2 emissions. This project aims to predict high-demand hours for additional bike stock using temporal and weather-related features, enabling better allocation of bikes and supporting sustainable transportation efforts.

## Feature Description
The dataset contains 1600 entries with both categorical and numerical features. Categorical features include `hour_of_day`, `day_of_week`, `month`, `holiday`, `weekday`, and `summertime`, while numerical features include `temp`, `dew`, `humidity`, `precip`, `snow`, `snow_depth`, `windspeed`, `cloudcover`, and `visibility`. The label to be predicted is `increase_stock`, a categorical variable indicating whether additional bike stock is needed.

## Exploratory Data Analysis
Initial analysis revealed a class imbalance, with low-demand instances outnumbering high-demand ones approximately four to one. Demand varied notably by hour, peaking between 17:00 and 18:00 and dropping after midnight. Weekends showed higher proportions of high demand compared to weekdays. Seasonal patterns were also evident, with demand increasing in spring and autumn and dropping in winter. Weather conditions had significant effects: rainy days reduced high demand from 19.38% to 5.15%, and snowy days had no high-demand occurrences. Holidays had minimal influence. The data was split into 80% training, 10% validation, and 10% testing sets, using a random seed of 42 to ensure reproducibility.

## Model Development
Multiple classification models were implemented and compared. These included a Naive Classifier, Logistic Regression, Linear and Quadratic Discriminant Analysis, k-Nearest Neighbors (k=11), Decision Trees, Random Forests, Bagging, AdaBoost, and XGBoost. Each model was tuned using grid search with cross-validation. Logistic Regression with L1 penalty and QDA with PCA preprocessing performed well, while XGBoost, after regularization, balanced performance with a training accuracy of 0.929, validation of 0.900, and test accuracy of 0.875. Simpler models like k-NN and Decision Trees captured basic non-linear patterns but were outperformed by ensemble methods.

## Model Comparison
The performance of all models was benchmarked on the same test set. The Naive Classifier, though achieving a high accuracy of 0.865, had a poor macro F1 score of 0.45. Logistic Regression, LDA, and QDA each achieved balanced performance with macro F1 scores around 0.75. KNN performed decently with a macro F1 of 0.72. Among tree-based methods, Bagging slightly outperformed Random Forest and Decision Tree on the test set, while XGBoost and AdaBoost delivered strong, generalizable performance. Overall, ensemble boosting methods emerged as the most effective models for capturing complex patterns in the data.
