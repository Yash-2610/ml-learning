# SVM Drug Classification — Normal vs PCA Version

This project applies Support Vector Machine (SVM) classification on the `drug200.csv` dataset to predict the correct drug type for a patient.  
The code builds **two models**:

1. SVM on normal preprocessed data  
2. SVM on PCA-reduced data  

Both versions are compared to understand how PCA affects SVM performance.

---

## 1. SVM Without PCA (Normal Version)

### Steps in the code:
- Load the dataset and inspect basic information.
- Separate the target variable `Drug` from the features.
- Convert categorical columns into numerical form using `get_dummies` with `drop_first=True`.
- Convert all feature values to float.
- Split the data into training and testing sets using an 80-20 split.
- Define a standard `SVC()` model.
- Use `GridSearchCV` to tune:
  - `C` values  
  - kernel type (`linear` or `rbf`)  
  - `gamma` values  
- Fit the model on training data.
- Predict on the test set.
- Print:
  - Best hyperparameters  
  - Accuracy without PCA  

This part gives the baseline performance of SVM on the raw encoded features.


# SVM Classification With PCA

This project applies Principal Component Analysis (PCA) before training an SVM classifier on the `drug200.csv` dataset. PCA is used to reduce the number of features while keeping most of the important information.

---

## Overview

- The dataset is preprocessed using one-hot encoding.
- All features are standardized because PCA and SVM both require scaled data.
- PCA is applied with `n_components=0.90`, meaning it keeps enough components to preserve 90% of the total variance.
- This reduces the feature space while removing noise and redundant information.
- An SVM model is then trained on the PCA-transformed features.
- Finally, the model is tested on PCA data, and accuracy is printed.

---

## What PCA Does Here

- Compresses high-dimensional data into fewer, more meaningful components.
- Removes correlated and less important features.
- Speeds up training and can improve generalization.
- Provides a cleaner, lower-dimensional input for SVM.

---

## 1. Scaling the Data
PCA requires all features to be on the same scale, so the data is standardized first:

`x_scaled = scalar.fit_transform(x)`

## 2. Applying PCA (Keep 90% Variance)

PCA reduces the number of features while keeping most of the information:
```
pca = PCA(n_components=0.90)
x_pca = pca.fit_transform(x_scaled)
```
This keeps components that preserve 90% of the dataset’s variance.



Model ia then train on this pca trained data 

## 3 Evaluating the PCA Model

Accuracy is calculated using the PCA-transformed test set:
```
y_pred_pca = pca_model.predict(x_pca_test)
accuracy_score(y_pred_pca, y_pca_test)
```