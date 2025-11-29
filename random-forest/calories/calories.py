import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
 
df=pd.read_csv("calories.csv")
print(df.info())

y=df['Calories']
x=pd.get_dummies(df.drop(columns=['Calories','User_ID']),drop_first=True).astype(int)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=RandomForestRegressor(n_estimators=30)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(f"R² Score: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
user_input={}

gender=input("Enter gender(male/female): ")
user_input[gender]=gender

nums=['Age','Height','Weight','Duration','Heart_Rate','Body_Temp']

for num in nums:
    user_input[num]=float(input(f"Enter {num}: "))

user_df=pd.DataFrame([user_input])
print(user_df)

user_encoded=pd.get_dummies(user_df,drop_first=True).astype(int)
user_encoded=user_encoded.reindex(columns=x.columns,fill_value=0)

user_pred=model.predict(user_encoded)
print(user_pred[0])






