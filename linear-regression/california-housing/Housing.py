import pandas as pd
from sklearn.datasets  import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

housing=fetch_california_housing()
x=housing.data
y=housing.target
df=pd.DataFrame(x,columns=housing.feature_names)
df['Target']=y
print(df.head())
print(df.info())
print(df.shape)

X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(X_train, y_train)
y_predict=model.predict(X_test)

mse=mean_squared_error(y_test, y_predict)
r2=r2_score(y_test, y_predict)

print("MSE is: ",mse)
print("R_Square is: ",r2)


