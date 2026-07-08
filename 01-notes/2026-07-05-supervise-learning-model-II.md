# Supervised Learning models - II — 2026-07-05

## SVM (Support Vector Machine)
**TL;DR:** Finds the widest possible gap between classes.

**Key Concepts**
	- Learns the optimal separating hyperplane
	- Maximizes margin between classes
	- Kernel trick handles non-linear data

## KNN (K-Nearest Neighbors)
**TL;DR:** Majority vote of the nearest neighbors.

**Key Concepts**
	- Instance-based (lazy) learning
	- Prediction based on K closest points
	- Sensitive to feature scaling and choice of K

## Bias
**TL;DR:** Model is too simple → Underfitting.

**Key Concepts**
	- Comes from overly simplistic assumptions
	- Misses important patterns
	- High training and test error

## Variance
**TL;DR:** Model memorizes data → Overfitting.

**Key Concepts**
	- Sensitive to small changes in training data
	- Learns noise instead of patterns
	- Low training error, high test error

## Ensemble Models
**TL;DR:** Combine multiple models for better predictions.

**Key Concepts**
	- Improves robustness and accuracy
	- Reduces bias, variance, or both
	- Includes Bagging and Boosting methods

## Random Forest
**TL;DR:** Many randomized decision trees vote together.

**Key Concepts**
	- Ensemble of decision trees
	- Uses bagging and random feature selection
	- Reduces overfitting
	- Good default model for tabular data

## XGBoost
**TL;DR:** Sequential trees learn from previous mistakes.

**Key Concepts**
	- Gradient Boosting algorithm
	- Each tree corrects previous errors
	- Regularized for better generalization
	- Excellent for structured/tabular datasets

## Math / formula (if applicable)
	- Decision boundary: wᵀx + b = 0 (for SVM)
	- Margin = 2 / ||w|| (for SVM)
	- Euclidean Distance = √Σ(xᵢ - yᵢ)² (for KNN)
	- Bias-Variance Tradeoff = Bias² + Variance + Irreducible Error

## Code
```python
# Decision Tree Classifier training
from sklearn.tree import DecisionTreeClassifier, plot_tree
drugtree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
drugtree.fit(X_train, y_train)
tree_pred = drugtree.predict(X_test)
# Plotting the trained Decision Tree model
plot_tree(drugtree)
# ROC-AUC score
y_pred_dt = drugtree.predict_proba(X_test)[:,1]
roc_dt = roc_auc_score(y_test, y_pred_dt)

# performing L1 normalization on the input features
X = normalize(X, norm='l1', copy=False)
# Decision Tree Regression training
dt_reg = DecisionTreeRegressor(criterion='squared_error', max_depth=4, random_state=42)

# SVM model training
svm = LinearSVC(class_weight='balanced', random_state=42, loss='hinge', fir_intercept=False)
svm.fit(X_train, y_train)
# ROC-AUC score
y_pred_svm= svm.decision_function(X_test)
roc_svm = roc_auc_score(y_test, y_pred_svm)

# KNN training
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_model = knn_classifier.fit(X_train,y_train)
yhat = knn_model.predict(X_test)
print("Test set Accuracy: ", accuracy_score(y_test, yhat))

# Random Forest and XGBoost modle training
n_estimators=100
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)

```

## Interview-style question I could be asked

### 1. What is the difference between Random Forest and XGBoost?

| Random Forest | XGBoost |
|---------------|----------|
| Uses **Bagging** | Uses **Boosting** |
| Trees are built **independently** | Trees are built **sequentially** |
| Reduces **variance** | Reduces **bias and variance** |
| Faster to train | Slower but usually more accurate |
| Easier to tune | More hyperparameters to tune |
| Great default choice, less tuning. | Best for high-performance tabular datasets when accuracy matters. |

### 2. Why is KNN not performing well even for the optimal value of K?

- Possible reasons:
	- Features are not properly scaled.
	- Irrelevant or noisy features.
	- Too many features (curse of dimensionality).
	- Overlapping classes.
	- Wrong distance metric.

### 3. Why does the model deteriorate as K increases?
	- As K increases, KNN considers more distant neighbors. This smooths the decision boundary too much, increasing bias and causing underfitting. Eventually, the model starts predicting based on the overall majority class rather than the local neighborhood, reducing accuracy.

## Links
- https://www.coursera.org/learn/machine-learning-with-python/