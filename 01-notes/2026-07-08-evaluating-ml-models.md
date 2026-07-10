# Evaluatin Machine Learning model - 2026-07-08

## TL;DR

- **Classification:** Measures how well a model predicts categories.
- **Regression:** Measures how close predictions are to actual continuous values.
- **Unsupervised Learning:** Measures the quality of clustering without labels.

## Key Concepts

- **Higher is Better:** Accuracy, Precision, Recall, F1, ROC-AUC, R², Silhouette Score
- **Lower is Better:** MSE, MAE, RMSE, DBI, Inertia
- **Confusion Matrix:** Foundation for most classification metrics.

| Metric | Goal | Best Value |
|---------|------|------------|
| Accuracy | Overall correctness | High |
| Precision | Reduce False Positives | High |
| Recall | Reduce False Negatives | High |
| F1 Score | Balance Precision & Recall | High |
| ROC-AUC | Class separability | Close to 1 |
| MSE | Penalize large errors | Low |
| MAE | Average absolute error | Low |
| RMSE | Error in original units | Low |
| R² | Explained variance | Close to 1 |
| Silhouette Score | Cluster quality | High |
| DBI | Cluster separation | Low |
| Inertia | Cluster compactness | Low |

## Math/ Formulas

### Classification

**Accuracy =**
$
\frac{TP+TN}{TP+TN+FP+FN}
$

**Precision =**
$
\frac{TP}{TP+FP}
$

**Recall =**
$
\frac{TP}{TP+FN}
$

**F1 Score =**
$
2\times\frac{Precision\times Recall}{Precision+Recall}
$

**ROC-AUC**
- Area under the ROC curve.
- Measures model's ability to distinguish between classes.
- Range: **0.5 (random) → 1 (perfect)**.

**Confusion Matrix**

| Actual \ Predicted | Positive | Negative |
|-------------------|----------|----------|
| Positive | TP | FN |
| Negative | FP | TN |

---

## Regression

**MSE =**
$
\frac{1}{n}\sum(y-\hat y)^2$

**MAE =**
$
\frac{1}{n}\sum|y-\hat y|
$

**RMSE =**
$
\sqrt{MSE}
$

**R² Score =**
$
1-\frac{SS_{res}}{SS_{tot}}
$

* Range:
    - **1** → Perfect fit
    - **0** → Same as predicting mean
    - **<0** → Worse than baseline

---

### Unsupervised Learning

**Silhouette Score**
- Range: **-1 to 1**
- Higher = Better clusters

**Davies-Bouldin Index (DBI)**
- Lower = Better clustering

**Inertia**
- Sum of squared distances of points to cluster centroids.
- Lower = Tighter clusters (used in the Elbow Method).

## Code
```python
# Classification evaluation metrics
print(f"KNN Testing Accuracy: {accuracy_score(y_test, y_pred_knn):.3f}")
print(classification_report(y_test, y_pred_knn))
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='d', ax=axes[0],
            xticklabels=labels, yticklabels=labels)

# Evaluating Random Forest model
importances = rf_regressor.feature_importances_
indices = np.argsort(importances)[::-1]
features = data.feature_names
plt.bar(range(X.shape[1]), importances[indices],  align="center")

# Evaluation of K-means clustering
inertia_values = kmeans.inertia_
silhouette_scores = silhouette_score(X, y_kmeans)
davies_bouldin_indices = davies_bouldin_score(X, y_kmeans)
```

## Interview-style question I could be asked
### 1. When would you prefer Precision over Recall, and vice versa?
- **Precision:** When False Positives are costly (e.g., spam detection).
- **Recall:** When False Negatives are costly (e.g., disease detection).

### 2. Why can't Accuracy alone be trusted for an imbalanced dataset?
Because a model can achieve high accuracy by predicting only the majority class while completely missing the minority class. Metrics like **Precision, Recall, F1 Score, and ROC-AUC** provide a more reliable evaluation.