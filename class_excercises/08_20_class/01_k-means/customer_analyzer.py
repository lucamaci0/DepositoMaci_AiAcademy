import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, "plots")
BASE_COLORS = {
    0:"#1f77b4", 1:"#ff7f0e", 2:"#2ca02c", 3:"#d62728", 4:"#9467bd",
    5:"#8c564b", 6:"#e377c2", 7:"#7f7f7f", 8:"#bcbd22", 9:"#17becf"
}

ignore_plots = False


####################################
# Data Preprocessing
####################################

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
df[feat_scaled] = df[feat_scaled].to_numpy()

####################################
# Model: KMeans
####################################

# Analyze which n_cluster makes more sense with an elbow-curve approach
ks = range(2, 11)
inertias = []
silhouettes = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(df[feat_scaled])
    # elbow with inertia:
    inertias.append(km.inertia_)
    # elbow with silhouette:
    silhouettes.append(silhouette_score(df[feat_scaled], km.fit_predict(df[feat_scaled])))

if not ignore_plots:
  ks_sil = ks[1:] if len(silhouettes) == len(ks) - 1 else ks
  fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
  # Right plot: elbow with inertia
  axes[0].plot(ks, inertias, marker="o")
  axes[0].set_xticks(list(ks))
  axes[0].set_xlabel("Number of clusters")
  axes[0].set_ylabel("Inertia (sum of squared distances)")
  axes[0].set_title("Inertia vs n_clusters")
  axes[0].grid(True, alpha=0.3)
  # Right plot: elbow with silhouettes
  axes[1].plot(ks_sil, silhouettes, marker="o")
  axes[1].set_xticks(list(ks_sil))
  axes[1].set_xlabel("Number of clusters")
  axes[1].set_ylabel("Silhouette score")
  axes[1].set_title("Silhouette vs n_clusters")
  axes[1].grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(os.path.join(PLOTS_DIR, "elbow_curve.png"), dpi=300, bbox_inches="tight")


# Elbow curves shows that 5 might be the most reasonable number of clusters for this population.
n_clusters = 5
print(f"\n{n_clusters} seems like the most reasonable number of clusters according to the elbow-graph.")
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
df["km_label"] = kmeans.fit_predict(df[feat_scaled])

# Plot of the clusters
labels = df["km_label"].to_numpy()
X = df[feat_for_clustering].to_numpy()  # original units for nicer axes
centers = scaler.inverse_transform(kmeans.cluster_centers_)

cluster_colors = {int(c): BASE_COLORS[int(c) % 10] for c in np.unique(labels)}
plt.figure(figsize=(6,6))
point_colors = [cluster_colors[int(c)] for c in labels]
plt.scatter(X[:,0], X[:,1], c=point_colors, s=25)
center_colors = [cluster_colors[i] for i in range(n_clusters)] 
plt.scatter(centers[:,0], centers[:,1], c=center_colors, s=200, marker="X", edgecolor="k")
plt.xlabel("Annual Income (k$)"); plt.ylabel("Spending Score (1–100)")
plt.tight_layout(); 
plt.savefig(os.path.join(PLOTS_DIR, "k-means_clustering.png"), dpi=300, bbox_inches="tight")

# Distance of each point from the centroid of its cluster
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
df["dist_from_centroids"] = np.linalg.norm(df[feat_for_clustering] - centers[df["km_label"]], axis=1)

mean_dist_per_cluster = df.groupby("km_label")["dist_from_centroids"].mean()
print("\nThe mean distance of points from their respective cluster's centroid is the following: \n")
print(mean_dist_per_cluster)




from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Silhouette on the same scaled space used by KMeans
X_scaled = df[feat_scaled].to_numpy()
labels   = df["km_label"].to_numpy()

sil_avg = float(silhouette_score(X_scaled, labels))
sample_sil = np.array(silhouette_samples(X_scaled, labels))

# Sort by (cluster label, then descending silhouette) for a clean stacked look
order = np.lexsort((-sample_sil, labels))
sorted_scores  = sample_sil[order]
sorted_labels  = labels[order]
bar_colors = [cluster_colors[int(c)] for c in sorted_labels]

plt.figure(figsize=(12, 4))
plt.bar(range(len(sorted_scores)), sorted_scores,
        color=bar_colors, edgecolor="black", linewidth=0.3)
plt.axhline(sil_avg, color="red", linestyle="--", linewidth=2,
            label=f"Mean silhouette = {sil_avg:.2f}")
plt.title("Silhouette score per point (colored by KMeans cluster)")
plt.xlabel("Points sorted by cluster")
plt.ylabel("Silhouette score")
plt.xticks([])
plt.ylim(-1, 1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "silhouette_points.png"), dpi=300, bbox_inches="tight")