import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv("Iris.csv")
df=df.drop(columns=['Id'])

print(df.head())
print(df.info())
print(df['Species'].unique())

y=df['Species']
x=df.iloc[ : , : -1]
print(x.head())
print(y.head())

x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2)
model=RandomForestClassifier(n_estimators=30)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print(f"Accuracy is: {accuracy_score(y_pred,y_test)}")

user_input={}
nums=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

for num in nums:
    user_input[num]=float(input(f"Enter {num}: "))
user_df=pd.DataFrame([user_input])
print(user_df)
user_pred=model.predict(user_df)
print(user_pred[0])
