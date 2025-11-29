import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df=pd.read_csv('drug200.csv')
print(df.head())
print(df.info())

y=df['Drug']
x=pd.get_dummies(df.drop(columns=['Drug']),drop_first=True).astype(float)
print(x.head(1))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=SVC()
model.fit(x_train,y_train)
parameters={
        'C':[10,40,80],
        'kernel': ['rbf','linear'],
        'gamma': [0.001,0.0005]

        }

grid_search=GridSearchCV(model,param_grid=parameters,cv=5)
grid_search.fit(x_train,y_train)
y_pred=grid_search.predict(x_test)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Without PCA: {accuracy_score(y_pred,y_test)}")


pca_model=SVC()
scalar=StandardScaler()
x_scaled=scalar.fit_transform(x)

pca=PCA(n_components=0.90)
x_pca=pca.fit_transform(x_scaled)
x_pca_train,x_test_pca,y_pca_train,y_test_pca=train_test_split(x_pca,y,test_size=0.2)

pca_model.fit(x_pca_train,y_pca_train)
y_pred_pca=pca_model.predict(x_test_pca)
print(f"With PCA: Accuracy: {accuracy_score(y_pred,y_test)}")
