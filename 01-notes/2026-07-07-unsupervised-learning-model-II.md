# Unsupervised Learning models Part II — 2026-07-07

## TL;DR
- **PCA (Principal Component Analysis)** is a **linear** dimensionality reduction technique that projects data onto directions of maximum variance.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)** is a **non-linear** visualization algorithm that preserves **local neighborhoods** extremely well.
- **UMAP (Uniform Manifold Approximation and Projection)** is a **non-linear** manifold learning algorithm that preserves both **local and some global structure**, while being faster and more scalable than t-SNE.
- **PCA → Feature extraction**
- **t-SNE → Visualization**
- **UMAP → Visualization + Feature embedding**

## Key concepts

### PCA
- Linear projection
- Maximizes variance
- Produces orthogonal principal components
- Sensitive to feature scaling
- Deterministic (same input → same output)
- Works well for high-dimensional data and reduces noise
- Cannot capture non-linear relationships

### t-SNE
- Non-linear manifold learning
- Preserves local neighborhood similarity
- Converts distances into probabilities
- Mainly used for 2D/3D visualization

### UMAP
- Non-linear manifold learning
- Builds nearest-neighbor graph
- Preserves local and partial global structure
- Faster than t-SNE
- Scales well to large datasets

## Code
```python
#  Initialize a 2-component PCA model and then fit and transform the feature space
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
components = pca.components_
evr = pca.explained_variance_ratio_

# t-SNE model initialization and training
X, labels_ = make_blobs(n_samples=500, centers=centers, n_features=3, cluster_std=cluster_std, random_state=42)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP model initialization and training
umap_model = UMAP.UMAP(n_components=2, random_state=42, min_dist=0.5, spread=1,n_jobs=1)
X_umap = umap_model.fit_transform(X_scaled)
```

## Interview-style question I could be asked
### 1. If PCA, t-SNE, and UMAP all reduce dimensions, how would you choose between them?
- Use **PCA** when you need fast feature reduction, interpretability, or preprocessing for machine learning models.
- Use **t-SNE** when the goal is purely visualization and discovering local clusters.
- Use **UMAP** when you want visualization that is faster than t-SNE, preserves more global structure, scales better, and can generate embeddings for new data.

### 2. Why do many practitioners apply PCA before t-SNE or UMAP?
Applying PCA first:
- Removes noise
- Reduces computational cost
- Compresses redundant features
- Improves stability of downstream non-linear embedding
- Commonly reduces data to 30–100 dimensions before running t-SNE or UMAP

## Links
https://www.coursera.org/learn/machine-learning-with-python/