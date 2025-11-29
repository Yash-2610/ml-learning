import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("insurance.csv")
df=df.iloc[ : -10]

y=df['charges']
x=pd.get_dummies(df.drop(columns=['charges']),drop_first=True).astype(float)

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
model=XGBRegressor(n_estimators=300, max_depth=6)
parameters={
                'n_estimators':[40,60,80],
                'learning_rate':[0.1,0.15],
                'max_depth':[2,3,4],
                'gamma':[2,2.22,],
                'subsample':[0.7, 0.8],
                'colsample_bytree':[0.7, 0.8]
            }

grid=GridSearchCV(model,param_grid=parameters,cv=4)
grid.fit(x_train,y_train)
y_pred=grid.predict(x_test)
print(f"Best parameters: {grid.best_params_}")
print(f"MSE is: {mean_squared_error(y_test,y_pred)}")
print(f"R2: {r2_score(y_test, y_pred)}")

user_data={}
nums=['age','bmi','children']
objs=['sex','smoker','region']

for num in nums:
    user_data[num]=float(input(f"Enter {num}:"))
for obj in objs:
      user_data[obj]=input(f"Enter {obj}:")

user_df=pd.DataFrame([user_data])
user_encoded=pd.get_dummies(user_df,drop_first=True).astype(float)
user_encoded=user_encoded.reindex(columns=x.columns,fill_value=0)

user_pred = grid.best_estimator_.predict(user_encoded)
print(user_pred[0])

