# Unsupervised Learning  Model - Part I — [2026-07-06]

## TL;DR

* **K-Means** partitions data into **K spherical clusters** by minimizing the distance between points and their assigned centroid.
* **DBSCAN** groups points based on **density** and identifies **outliers (noise)** automatically.
* **HDBSCAN** is an extension of DBSCAN that can find **clusters of varying densities** and usually requires less parameter tuning.
* **Choose K-Means** for fast clustering on roughly spherical, similarly sized clusters.
* **Choose DBSCAN/HDBSCAN** when clusters have irregular shapes or when outlier detection is important.

## Key Concepts

### K-Means

* Centroid-based clustering.
* Requires specifying **K** beforehand.
* Sensitive to initialization and outliers.
* Works well for convex/spherical clusters.
* Uses iterative optimization:
  1. Assign points to nearest centroid.
  2. Update centroids.
  3. Repeat until convergence.

### DBSCAN

* Density-based clustering.
* No need to specify number of clusters.
* Parameters:
  * **eps (ε):** neighborhood radius.
  * **min_samples (MinPts):** minimum neighbors required to form a dense region.
* Labels points as:
  * Core point
  * Border point
  * Noise (outlier)
* Can discover arbitrarily shaped clusters.

### HDBSCAN

* Hierarchical version of DBSCAN.
* Handles clusters with **different densities**.
* Builds a hierarchy of density-connected clusters and selects the most stable ones.
* Main parameter:

  * **min_cluster_size**
* Usually performs better than DBSCAN when densities vary.

## Math / Formula

### Euclidean Distance (used by K-Means)
$$
d(x,y)=\sqrt{\sum_{j=1}^{n}(x_j-y_j)^2}
$$

### DBSCAN Density Condition

A point is a **Core Point** if:

$$
|N_{\varepsilon}(p)| \ge \text{MinPts}
$$

where:
* $N_{\varepsilon}(p)$ = Neighborhood of point $p$ within radius $\varepsilon$
* $\varepsilon$ = Radius (epsilon)

## Code
```python

# Generate 5,000 synthetic 2D data points around four predefined cluster centers with a standard deviation of 0.9, returning features (X) and true cluster labels (y).
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
# Plotting the clusters
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):
  my_members = (k_means_labels == k)
  cluster_center = k_means_cluster_centers[k]
  ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)
  ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# DBSCAN and HDBSCAN model training
coords_scaled["Latitude"] = 2*coords_scaled["Latitude"]
dbscan = DBSCAN(eps=1, min_samples=3, metric='euclidean').fit(coords_scaled)
df['Cluster'] = dbscan.fit_predict(coords_scaled)  # Assign the cluster labels

hdb = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
df['Cluster'] = hdb.fit_predict(coords_scaled)
```

## Interview-Style Questions

### 1. You have customer data with unknown number of groups, irregular cluster shapes, and many outliers. Which algorithm would you choose among K-Means, DBSCAN, and HDBSCAN, and why?

* K-Means is unsuitable because it assumes spherical clusters and requires specifying **K**.
* DBSCAN can find arbitrary shapes and identify noise but struggles when cluster densities differ.
* HDBSCAN is the best choice because it automatically determines clusters, handles varying densities, and detects outliers.

### 2. Compare K-Means, DBSCAN, and HDBSCAN in terms of assumptions, parameters, strengths, and limitations. When would each fail?

* **K-Means:** Fails on non-spherical clusters, varying cluster sizes, and outliers.
* **DBSCAN:** Fails when a single `eps` cannot capture clusters of different densities.
* **HDBSCAN:** More robust to varying densities but is computationally heavier and can be slower on very large datasets.
