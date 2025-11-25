Decision Tree Classifier Projects

This folder contains two projects where I used Decision Tree classifiers with GridSearchCV to understand how tree based models work on different kinds of data.

1. Drug Classification

In this project I worked with a medical drug dataset.
The goal was to predict which drug a patient should be given based on features like age, sex, blood pressure, cholesterol and sodium to potassium ratio.

Steps I followed

Loaded the dataset from drug200.csv

Explored the data and checked unique values for key columns

Converted categorical features into numerical form using one hot encoding

Split the data into training and testing sets

Built a DecisionTreeClassifier with a limited max_depth to avoid overfitting

Used GridSearchCV to tune hyperparameters like criterion and splitter

Evaluated the best model using accuracy

Plotted the final decision tree to understand the splits and decisions

This project helped me see how tree depth and splitting strategy affect the model and how GridSearchCV can automatically choose better parameters.

2. Road Accident Prediction

In this project I used an accident dataset to predict whether an accident occurs under certain conditions.
The dataset included features such as weather, road type, time of day, vehicle type, road condition, road light condition and accident severity.

Steps I followed

Loaded the dataset from accident_dataset.csv and removed missing values

Checked the number of unique values for categorical columns

Applied one hot encoding to multiple categorical features

Inspected the target distribution to understand the class balance

Split the dataset into training and testing sets

Built a DecisionTreeClassifier and tuned it with GridSearchCV over criterion, max_depth and splitter

Evaluated the best model using accuracy and printed a classification report

Visualized the final decision tree with a larger figure size for better readability

This project made me more comfortable with handling multiple categorical features, using GridSearchCV with more parameters and interpreting a larger decision tree on a more complex dataset.

What I learned

From these two projects I learned how decision trees
handle both numerical and categorical data,
how hyperparameters like depth and splitting strategy
change the complexity of the model
and how visualization helps in understanding
what the model is actually doing.