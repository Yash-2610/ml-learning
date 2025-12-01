import pandas as pd
df=pd.read_csv("wine-clustering.csv")
print(df.head())
print(df.info())

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss=[]
for k in range(1,11):
    model=KMeans(n_clusters=k,max_iter=100,n_init=5,init='k-means++')
    model.fit(df)
    wcss.append(model.inertia_)

plt.plot(range(1,11),wcss)
plt.grid(True)
plt.show()

kmeans=KMeans(n_clusters=4,max_iter=100,n_init=5,init="k-means++")
df['Cluster']=kmeans.fit_predict(df)

for cluster_id in range(kmeans.n_clusters):
    plt.scatter(
        df[df['Cluster']==cluster_id] ['Alcohol'],
        df[df['Cluster']==cluster_id] ['Malic_Acid'],
        label=f'Cluster{cluster_id}'
    )

plt.title("Alcohol VS Malic Acid")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.legend()
plt.show()
    