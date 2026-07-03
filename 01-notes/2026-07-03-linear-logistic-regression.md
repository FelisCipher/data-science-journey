# Linear and Logistic Regression — 2026-07-03

## TL;DR
| Model                          | Purpose                   | Input Features                                     | Output                          | Key Idea                                                                                                                                         | When to Use                                            |
| ------------------------------ | ------------------------- | -------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| **Simple Linear Regression**   | Predict continuous values | **1 feature**                                      | Continuous                      | Fits a **straight line** between one feature and the target.                                                                                     | One feature has a linear relationship with the target. |
| **Multiple Linear Regression** | Predict continuous values | **2 or more features**                             | Continuous                      | Uses multiple independent variables to predict one continuous target. Assumes a **linear relationship**.                                         | Multiple features influence the target.                |
| **Polynomial Regression**      | Predict continuous values | 1 or more features (converted to polynomial terms) | Continuous                      | Captures **curved (non-linear) relationships** by adding powers like x², x³, etc. Still uses Linear Regression internally.                       | Data has a curved trend.                               |
| **Non-Linear Regression**      | Predict continuous values | Any number of features                             | Continuous                      | Fits data using a **non-linear equation** (e.g., exponential, logarithmic, sigmoid). Parameters are estimated iteratively.                       | Relationship cannot be approximated by a polynomial.   |
| **Logistic Regression**        | Classification            | 1 or more features                                 | Class (0/1 or multiple classes) | Predicts **probability** using the **sigmoid function**, then converts it into a class label. Despite its name, it's a classification algorithm. | Binary or multiclass classification.                   |


## Key concepts
 - **Simple Linear Regression** - Predicts a continuous value using one independent variable.
 - **Multiple Linear Regression** - Predicts a continuous value using multiple independent variables.
 - **Polynomial Regression** - Uses polynomial features (x², x³...) to model curved relationships while remaining linear in the coefficients.
 - **Non-Linear Regression** - Uses a non-linear mathematical function when the relationship cannot be modeled well by a straight line or polynomial.
 - **Logistic Regression** - Predicts the probability of belonging to a class, then classifies based on a threshold (usually 0.5).
 
## Math / formula (if applicable)
 - Simple linear Regression: y = b<sub>0</sub> + b<sub>1</sub>x
 - Multiple linear Regression: y = b<sub>0</sub> + b<sub>1</sub>x<sub>1</sub> + b<sub>2</sub>x<sub>2</sub>...
 - Polynomial Regression: y = b<sub>0</sub> + b<sub>1</sub>x<sub>1</sub> + b<sub>2</sub>x<sub>2</sub><sup>2</sup>
 - Non-Linear Regression: y = b<sub>0</sub>e<sup>b<sub>1</sub>x</sup> 
 or y = b<sub>0</sub>ln(x + b<sub>1</sub>)
 - Logistic Regression: σ(z)= 1/(1+e<sup>−z</sup>)
 

## Code
```python

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Model training and predicting using Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print ('Coefficients: ', regressor.coef_[0]) 
print ('Intercept: ',regressor.intercept_)

# preprocessing the input features by standardizing them
from sklearn import preprocessing
std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)

# Model training and predicting using Logistic Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train,y_train)
yhat = LR.predict(X_test) # predicts the class for the datapoints
yhat_prob = LR.predict_proba(X_test) #gives the actual probabilityvalues for the datapoints
log_loss(y_test, yhat_prob) # measure prediction error based on predicted probabilities
```

## Interview-style question I could be asked
1. How do you decide which regression model to use (Linear, Multiple Linear, Polynomial, Non-Linear, or Logistic Regression)?
	a) Simple Linear Regression → One feature, continuous output, linear relationship.
	b) Multiple Linear Regression → Multiple features, continuous output, linear relationship.
	c) Polynomial Regression → Continuous output with a curved relationship (uses polynomial features).
	d) Non-Linear Regression → Complex relationships that cannot be modeled using polynomial features (e.g., exponential, logarithmic).
	e) Logistic Regression → Classification problems (binary or multiclass), predicts probabilities.

2. Why can't we use Linear Regression for classification and use Logistic Regression instead?
	- Linear Regression predicts continuous values and can produce outputs less than 0 or greater than 1, so it is unsuitable for classification.
	- Logistic Regression applies the sigmoid function to produce probabilities between 0 and 1, which can then be converted into class labels using a threshold (e.g., 0.5).

## Links
- https://www.coursera.org/learn/machine-learning-with-python/