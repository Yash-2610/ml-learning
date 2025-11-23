import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix



df=pd.read_csv("weather_forecast_data.csv")
df['Rain']=df['Rain'].map({'no rain':0, 'rain': 1})
print(df.head())
print(df.info())
x=df.iloc[ : , :-1]
y=df.iloc[ : , -1]

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print(f"Accuracy is: {accuracy_score(y_pred, y_test)}")
print(f"Confusion Matrix is: \n {confusion_matrix(y_pred, y_test)}")
print(f"Classificitaion Report is: \n {classification_report(y_pred, y_test)}")



