import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv("loan_data.csv")

# Apply get_dummies() to encode categorical columns into numerical form
df = pd.get_dummies(df, columns=[
                                    'person_gender', 
                                    'person_education', 
                                    'person_home_ownership', 
                                    'loan_intent', 
                                    'previous_loan_defaults_on_file'], 
    drop_first=True)

# Check the dataframe to ensure all object columns are encoded
print(df.info())  # This will show the datatypes after encoding

# Define X and y
x = df.iloc[:, :-1]  # All columns except the last column (loan_status)
y = df.iloc[:, -1]   # Last column (loan_status)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(x_train, y_train)

# Predict using the trained model
y_pred = model.predict(x_test)

# Print accuracy, confusion matrix, and classification report
print(f"Accuracy is: {accuracy_score(y_pred, y_test)}")
print(f"Confusion Matrix is: {confusion_matrix(y_pred, y_test)}")
print(f"Classification Report is: {classification_report(y_pred, y_test)}")

# Plot Confusion Matrix
cm = confusion_matrix(y_pred, y_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
