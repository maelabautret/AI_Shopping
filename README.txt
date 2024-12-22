README

Overview

This project implements a K-Nearest Neighbors (KNN) classifier to predict whether a user browsing an online shopping website will complete a purchase. The classifier uses data about user activity (e.g., number of pages visited, duration on certain sections, weekend browsing) to make predictions.

The script processes input data from a CSV file, splits it into training and testing sets (70%/30%), trains the KNN model, and evaluates its performance using sensitivity and specificity metrics. Additionally, the script uses cross-validation to assess the overall robustness of the model.

Features

CSV Data Loading: Reads the input file and extracts evidence (features) and labels (target values).

KNN Classifier: Uses a distance-weighted K-Nearest Neighbors model to classify user behavior.

Threshold Adjustment: Tests various classification thresholds (e.g., 0.3, 0.4, 0.5) to balance sensitivity and specificity.

Automatic k Selection: Evaluates multiple values of k (from 1 to 10) to find the best-performing configuration.

Cross-Validation: Evaluates the model across 5 folds to ensure its generalizability and avoid overfitting.

Performance Metrics:

Sensitivity (True Positive Rate): Measures the proportion of actual purchasers correctly identified.

Specificity (True Negative Rate): Measures the proportion of non-purchasers correctly identified.

Best Model

After testing multiple configurations, the best model was found with:

k = 3

Threshold = 0.30

Justification for k = 3

The project goal is to "perform reasonably on both metrics"—sensitivity and specificity. Here’s why k=3 is the best choice:

Balanced Sensitivity and Specificity:

Sensitivity: The model detects a reasonable proportion of actual purchasers (~54%).

Specificity: It maintains strong accuracy in identifying non-purchasers (~84%).

Cross-Validation Support:

k=3 shows consistent performance during cross-validation, indicating it generalizes well to unseen data.

Why Not Smaller or Larger k?:

Smaller k (e.g., k=1): High sensitivity but overfits, leading to lower specificity.

Larger k (e.g., k=5 or more): Sacrifices sensitivity significantly for slightly higher specificity.

Thus, k=3 strikes the right balance.

Usage

Place the CSV data file (e.g., shopping.csv) in the same directory as the script.

Run the script with:

python shopping.py shopping.csv

The script will output:

The best k and threshold.

Sensitivity and specificity metrics for the best model.

Cross-validation accuracy for each k value tested.