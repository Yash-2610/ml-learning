import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

df=pd.read_csv("accident_dataset.csv")
df=df.dropna()
print(df.head(3))
print(df.tail(3))
unique_count=df[['Weather', 'Road_Type', 'Time_of_Day', 'Vehicle_Type', 'Road_Condition', 'Accident_Severity']].nunique()
print(unique_count)


df=pd.get_dummies(df,columns=[
                            'Weather',
                            'Road_Type',
                            'Time_of_Day',
                            'Vehicle_Type',
                            'Road_Condition',
                            'Accident_Severity',
                            'Road_Light_Condition'
                            ], drop_first=True).astype(int)
print(df.info())
print(df['Accident'].value_counts())

x=df.iloc[ : , : -1]
y=df.iloc[ : , -1]
print(x.head(2))
print(y.head(2))

x_train,x_test, y_train, y_test= train_test_split(x,y,test_size=0.2)
treemodel=DecisionTreeClassifier()
parameter= {
    
            'criterion':['gini','entropy'],
            'max_depth':[None,1,2,3],
            'splitter':['best','random']
}

grid_search=GridSearchCV(treemodel,param_grid=parameter,cv=5)
grid_search.fit(x_train,y_train)
y_pred=grid_search.predict(x_test)
print("Best parameter: ",grid_search.best_params_)

print("Accuracy:",accuracy_score(y_test,y_pred))

plt.figure(figsize=(35, 25))
tree.plot_tree(grid_search.best_estimator_, 
          filled=True,
          feature_names=x_train.columns,
          class_names=grid_search.best_estimator_.classes_.astype(str), 
          rounded=True)
plt.title("Best Decision Tree from GridSearchCV")
plt.show()






