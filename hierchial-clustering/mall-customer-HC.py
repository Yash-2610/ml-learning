import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt

df=pd.read_csv("Mall_Customers.csv")
df=df.drop(columns=['CustomerID'])
print(df.info())
print(df.head())
num_clms=df.select_dtypes(include=['int64'])
categorical_clms=df.select_dtypes(include=['object'])

scalar=StandardScaler()
num_clms=scalar.fit_transform(num_clms)
num_clms=pd.DataFrame(num_clms)

categorical_clms_encoded=pd.get_dummies(categorical_clms).astype(float)
df=pd.concat([num_clms,categorical_clms_encoded],axis=1)
df.columns = df.columns.astype(str)

print(df)



linked=linkage(df,method='ward')
dendrogram(linked, orientation='top',distance_sort='descending',show_leaf_counts=True)
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()

model=AgglomerativeClustering(n_clusters=4,linkage='ward')
df['Cluster']=model.fit_predict(df)

for cluster_num in df['Cluster'].unique():
    cluster_data=df[df['Cluster']==cluster_num]
    plt.scatter(cluster_data['1'], cluster_data['2'] ,label=f'Cluster:{cluster_num}')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()