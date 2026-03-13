<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Insurance Charges Prediction</title>
<style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 20px auto; line-height: 1.6; color: #333; background: #f9f9f9; padding: 0 20px; }
    h1, h2, h3 { color: #1a73e8; }
    pre { background: #eee; padding: 10px; border-radius: 6px; overflow-x: auto; }
    .highlight { color: #d9534f; font-weight: bold; }
</style>
</head>
<body>

<h1>Insurance Charges Prediction</h1>

<h2>1. Overview</h2>
<p>Predict medical insurance charges using demographic and health data with Machine Learning.</p>

<h2>2. Dataset</h2>
<p>Contains 1337 records with features:</p>
<ul>
    <li>age, sex (0=female,1=male), bmi, children, smoker (0=no,1=yes), region (one-hot encoded), charges (target)</li>
</ul>

<h2>3. Preprocessing</h2>
<ul>
    <li>Removed duplicates</li>
    <li>Label encoding for <code>sex</code> and <code>smoker</code></li>
    <li>One-hot encoding for <code>region</code></li>
    <li>Log-transform target <code>charges</code> to reduce skewness</li>
    <li>Split: 80% training, 20% testing</li>
    <li>Standard scale numerical features (<code>age</code>, <code>bmi</code>, <code>children</code>)</li>
</ul>

<h2>4. Models</h2>

<h3>Polynomial Linear Regression</h3>
<p>Captures nonlinear relationships in the data.</p>
<pre><code>
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)

y_pred_poly_final = np.expm1(poly_model.predict(X_test_poly))
</code></pre>
<p><strong>Performance:</strong> R² = <span class="highlight">88.56%</span>, MAE = <span class="highlight">$0.19K</span></p>

<h3>Optimized KNN Regression</h3>
<p>Hyperparameter tuned using GridSearchCV.</p>
<pre><code>
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# Pipeline: Scaling, Feature Selection, KNN
# GridSearchCV to find best parameters
</code></pre>
<p><strong>Best Parameters:</strong> n_neighbors=30, weights='uniform', p=2, select__k=4<br>
<strong>Performance:</strong> R² = <span class="highlight">85.13%</span>, RMSE = <span class="highlight">0.0371</span></p>

<h2>5. Results Comparison</h2>
<ul>
    <li>Polynomial regression performs slightly better than KNN.</li>
    <li>KNN is competitive with tuned hyperparameters.</li>
</ul>

<h2>6. Conclusion</h2>
<p>Both models successfully predict insurance charges. Preprocessing and feature engineering improve model accuracy. Polynomial regression provides the best R².</p>

</body>
</html>
