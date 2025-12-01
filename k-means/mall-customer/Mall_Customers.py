import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df=pd.read_csv("Mall_Customers.csv")

print(df.head())
print(df.info())
df=pd.get_dummies(df.drop(columns=['CustomerID']),drop_first=True).astype(float)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss=[]
for k in range(1,11):
    model=KMeans(n_clusters=k,max_iter=100,n_init=5,init='k-means++',random_state=42)
    model.fit(df)
    wcss.append(model.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("Value of K")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

kmeans=KMeans(n_clusters=5,max_iter=100,n_init=5,init='k-means++',random_state=42)
df['Cluster']=kmeans.fit_predict(df)

for cluster_id in range(5):

    plt.scatter(
                df[df['Cluster']==cluster_id] ['Annual Income (k$)'],   #say cluster_id is 0 for 0th cluster then inmer df will store all boolean values like 
                                                        #[true, false,true,false], then outter df will only take true value and from that too only "REgion" value will be plotted
                df[df['Cluster']==cluster_id] ['Spending Score (1-100)'],
                label=f'Cluster {cluster_id}'
    )


plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title('Annual Income (k$) VS Spending Score (1-100)')
plt.legend()
plt.show()
