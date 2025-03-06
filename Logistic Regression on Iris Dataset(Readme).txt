Logistic Regression on Iris Dataset

Overview

This project implements a Logistic Regression model using scikit-learn to classify the Iris dataset into two classes (Setosa and Versicolor). The model is trained on a subset of the dataset and evaluated for accuracy.

Dataset

The Iris dataset consists of three classes of flowers. However, for this project, we are performing binary classification by selecting only the first two classes (Setosa and Versicolor).

Installation

Ensure you have Python installed along with the required libraries:

pip install numpy pandas scikit-learn

Code Implementation

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd  # Fixed the import statement

# Load the iris dataset
data = load_iris()

# Taking only two classes (for binary classification)
X = data.data[:100]  # Selecting the first 100 samples (class 0 and class 1)
y = data.target[:100]  # Corresponding target labels (0 and 1)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

Interpreting the Results

High Accuracy (>90%) → The model is performing well.

Low Accuracy (<70%) → Possible issues include:

Data imbalance (one class dominates the dataset)

Features not well separated

Overfitting or underfitting

Contributing

Feel free to submit issues or pull requests if you have suggestions for improvement.

License

This project is open-source and available under the MIT License.