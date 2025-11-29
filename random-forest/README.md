Random Forest Projects

This folder contains two small projects where I tried Random Forest models on both regression and classification tasks. The goal was to understand how Random Forests work, how they handle mixed features and how to evaluate them properly.

1. Calories Burned Prediction (Regression)

This project predicts how many calories a person burns during physical activity.
The dataset includes values such as age, height, weight, duration of exercise, heart rate and body temperature. Gender was also encoded using one hot encoding.

Steps I followed

Loaded the calories.csv dataset

Dropped unnecessary columns like User_ID

Converted categorical features using one hot encoding

Split the data into training and testing sets

Trained a RandomForestRegressor with 30 trees

Evaluated the model using R² score and Mean Squared Error

I also added a small interactive part where a user can manually enter their details and the model predicts the calories burned.
This helped me understand how to prepare a single user input and match it with the training columns using reindexing.

2. Iris Flower Classification (Classification)

This project uses the classic Iris dataset to classify flowers into three species based on measurements of the sepals and petals.

Steps I followed

Loaded the dataset from Iris.csv and removed the Id column

Explored the unique species and basic structure

Split the dataset into training and testing sets

Trained a RandomForestClassifier with 30 trees

Evaluated the model using accuracy

What I learned

Working on these two tasks showed me how Random Forests can work for both regression and classification without much change in structure.
I learned how the number of trees affects performance, how to properly encode inputs for predictions and how regression metrics differ from classification metrics.
These projects are basic, but they gave me a clear understanding of how Random Forest models behave in practice.