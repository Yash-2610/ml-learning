import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt

df=pd.read_csv("Student_Marks.csv")
print(df.head())
print(df.info())
x=df.iloc[ : , :2]
y=df.iloc[ : , -1]

X_train, X_test, y_train, y_test= train_test_split(x,y ,test_size=0.2)
model=LinearRegression()
model.fit(X_train, y_train)
y_predict=model.predict(X_test)

mse=mean_squared_error(y_test, y_predict)
r2=r2_score(y_test , y_predict)
print(f"MSE is: {mse}")
print(f"R square is: {r2_score(y_test, y_predict)}")

import matplotlib.pyplot as plt

plt.scatter(y_test, y_predict, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.title('Actual vs Predicted Marks')
plt.grid(True)
plt.show()


