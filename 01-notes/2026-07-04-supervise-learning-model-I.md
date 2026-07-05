# Supervised Learning models - I — 2026-07-04

## Classification
**TL;DR:** Predict a category.

**Key Concepts**
	- Predicts discrete labels/classes
	- Examples: Spam/Not Spam, Fraud/Not Fraud
	- Common metrics: Accuracy, Precision, Recall, F1, ROC-AUC

## Regression
**TL;DR:** Predict a number.

**Key Concepts**
	- Predicts continuous values
	- Examples: House Price, Temperature
	- Common metrics: MAE, MSE, RMSE, R²

## Decision Tree
**TL;DR:** Series of if-else questions → prediction.

**Key Concepts**
	- Tree-based model using decision rules
	- Splits data to reduce impurity/error
	- Easy to interpret
	- Can overfit if too deep

## Regression Tree
**TL;DR:** Decision tree for predicting numbers.

**Key Concepts**
	- Used for regression tasks
	- Splits data to minimize variance/MSE
	- Outputs continuous values

## Math / formula (if applicable)
	- Entropy = -Σ pᵢ log₂(pᵢ)
	- Gini = 1 - Σ pᵢ²

## Code
```python
# To return a list of float64 columns in the dataframe
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()

# Applying One Hot Encoder to the categorical columns to convert them into numerical columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

# Performing label encoding on target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

# One vs All classification
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)
y_pred_ova = model_ova.predict(X_test)

# One Vs One classifier
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)
y_pred_ovo = model_ovo.predict(X_test)

```

## Interview-style question I could be asked

### 1. What is the difference between Classification and Regression?

	- **Classification** predicts **categorical outputs** (e.g., Spam/Not Spam, Yes/No).
	- **Regression** predicts **continuous numerical values** (e.g., House Price, Temperature).

	**Examples**
		- Classification → Email is Spam or Not Spam.
		- Regression → Predict the price of a house.

## Links
- https://www.coursera.org/learn/machine-learning-with-python/