import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Caricamento dei file necessari
datasets_dir = "other/Archivio Datasets/03_Lesson"
dataset_name = "Mall_Customers.csv"

df = pd.read_csv(os.path.join(datasets_dir, dataset_name))

# How many NAs are there in each column?
print("\nNA present in each column:\n")
print(df.isna().sum())
print("\n")
# No NAs present here

# Are there some (possibly faulty) extreme values?
for col in df.columns:
  unique_values = df[col].sort_values().unique()
  amount = min(len(unique_values), len(unique_values)//10) #TODO: fix this for situations like "Genre" with very few unique values
  #print(f"amount = {amount}")
  print(f"Column {col} has lowest unique values: {unique_values[:amount]}")
  print(f"Column {col} has highest unique values: {unique_values[-amount:]}")
# No weird/extreme values spotted


# Scaling data to improve distance-based clustering
feat_for_clustering = ['Annual Income (k$)', 'Spending Score (1-100)'] 
feat_scaled = ['SCALED Annual Income (k$)', 'SCALED Spending Score (1-100)'] 
scaler = StandardScaler()
df[feat_scaled] = scaler.fit_transform(df[feat_for_clustering])


####################################
# Model: KMeans
####################################

# Analyze which n_cluster makes more sense with an elbow-curve approach
ks = range(1, 11)
inertias = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(df[feat_scaled])
    inertias.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(list(ks), inertias, marker="o")
plt.xticks(list(ks))
plt.xlabel("k (number of clusters)")
plt.ylabel("Inertia (sum of squared distances)")
plt.title("Elbow curve")
plt.tight_layout()
plt.show()

# Elbow curve shows that 5 might be the most reasonable number of clusters for this population.
n_clusters = 5
print(f"\n{n_clusters} seems like the most reasonable number of clusters according to the elbow-graph.")
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
df["label"] = kmeans.fit_predict(df[feat_scaled])

# Plot of the clusters
X = df[feat_for_clustering].to_numpy()  # original units for nicer axes
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=df["label"], s=25)
plt.scatter(centers[:,0], centers[:,1], s=200, marker="X", edgecolor="k")
plt.xlabel("Annual Income (k$)"); plt.ylabel("Spending Score (1–100)")
plt.tight_layout(); 
plt.show()

# Distance of each point from the centroid of its cluster
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
df["dist_from_centroids"] = np.linalg.norm(df[feat_for_clustering] - centers[df["label"]], axis=1)

mean_dist_per_cluster = df.groupby("label")["dist_from_centroids"].mean()
print("\nThe mean distance of points from their respective cluster's centroid is the following: \n")
print(mean_dist_per_cluster)