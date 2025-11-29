import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

df=pd.read_csv("drug200.csv")
print(df.head(3))
print(df.tail(3))

print(df['Cholesterol'].nunique())
print(df['BP'].nunique())
print(df['Drug'].nunique())

print(df.info())

y = df['Drug'] 
x = pd.get_dummies(df.drop(columns=['Drug']), drop_first=True).astype(int)
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2)
treemodel=DecisionTreeClassifier(max_depth=3)
parameter= {
                'criterion':['gini','entropy'],
                'splitter':['best','random'],
            
        }

cv=GridSearchCV(treemodel,param_grid=parameter,cv=3)
cv.fit(x_train,y_train)
y_pred=cv.predict(x_test)
print('Best parameter: ',cv.best_params_)

print(f"Accuracy is: {accuracy_score(y_test,y_pred)}")

plt.figure(figsize=(5, 5))
tree.plot_tree(cv.best_estimator_,  
          filled=True,
          feature_names=x_train.columns,
          class_names=cv.best_estimator_.classes_.astype(str), 
          rounded=True)
plt.title("Best Decision Tree from GridSearchCV")
plt.show()