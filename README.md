<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Healthcare-Insurance-Expense</title>
<style>
    body { font-family: Arial, sans-serif; max-width: 1000px; margin: 20px auto; line-height: 1.6; color: #333; background: #f9f9f9; padding: 0 20px; }
    h1, h2, h3 { color: #1a73e8; }
    code { background: #eee; padding: 2px 6px; border-radius: 4px; }
    pre { background: #eee; padding: 10px; border-radius: 6px; overflow-x: auto; }
    table { border-collapse: collapse; width: 100%; margin: 15px 0; }
    th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
    .highlight { color: #d9534f; font-weight: bold; }
</style>
</head>
<body>

<h1>Insurance Charges Prediction Project</h1>

<h2>1. Project Overview</h2>
<p>This project predicts medical insurance charges using Machine Learning. It leverages demographic and health features to model insurance costs accurately.</p>

<h2>2. Dataset</h2>
<p>The dataset contains 1337 records after removing duplicates and includes these features:</p>
<ul>
    <li><strong>age</strong> - Age of the individual</li>
    <li><strong>sex</strong> - Gender (0=female, 1=male)</li>
    <li><strong>bmi</strong> - Body Mass Index</li>
    <li><strong>children</strong> - Number of children covered</li>
    <li><strong>smoker</strong> - Smoking status (0=no, 1=yes)</li>
    <li><strong>region</strong> - Geographical region (One-Hot Encoded)</li>
    <li><strong>charges</strong> - Insurance charges (target)</li>
</ul>

<h2>3. Data Preprocessing</h2>
<ol>
    <li>Check for nulls and duplicates. Dropped 1 duplicate row.</li>
    <li>Label encode <code>sex</code> and <code>smoker</code>.</li>
    <li>One-Hot encode <code>region</code> to avoid false ordinal relationships.</li>
    <li>Log-transform <code>charges</code> to reduce skewness.</li>
    <li>Split dataset: 80% training, 20% testing.</li>
    <li>Standard scale numerical features (<code>age</code>, <code>bmi</code>, <code>children</code>).</li>
</ol>

<h2>4. Exploratory Data Analysis</h2>
<ul>
    <li>Boxplots for <code>age</code>, <code>bmi</code>, <code>charges</code> to detect outliers.</li>
    <li>Histogram for <code>charges</code> distribution and skewness check.</li>
    <li>Scatter plot of <code>bmi</code> vs <code>charges</code>, colored by smoker status.</li>
</ul>

<h2>5. Modeling</h2>

<h3>5.1 Polynomial Linear Regression</h3>
<pre><code>
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)

y_pred_poly_log = poly_model.predict(X_test_poly)
y_pred_poly_final = np.expm1(y_pred_poly_log)
</code></pre>
<p><strong>Results:</strong> R² = <span class="highlight">88.56%</span>, MAE = <span class="highlight">$0.19K</span></p>

<h3>5.2 K-Nearest Neighbors (KNN) with GridSearchCV</h3>
<pre><code>
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# Pipeline: Scaling, Feature Selection, KNN
# Grid Search to optimize parameters
</code></pre>
<p><strong>Best Parameters:</strong> n_neighbors=30, weights='uniform', p=2, select__k=4<br>
<strong>Performance:</strong> R² = <span class="highlight">85.13%</span>, RMSE = <span class="highlight">0.0371</span></p>

<h2>6. Results Comparison</h2>
<p>Both models provide accurate predictions:</p>
<ul>
    <li>Polynomial regression captures nonlinear relationships slightly better (higher R² and lower MAE).</li>
    <li>KNN is competitive but slightly less accurate; depends on neighbors and feature scaling.</li>
</ul>

<h2>7. Visualizations</h2>
<ul>
    <li>Actual vs Predicted charges scatter plots for Polynomial and KNN.</li>
    <li>Residual analysis for Polynomial model shows errors distributed around zero.</li>
</ul>

<h2>8. Tools & Libraries</h2>
<ul>
    <li>Pandas, NumPy</li>
    <li>Matplotlib, Seaborn</li>
    <li>Scikit-learn: LinearRegression, KNeighborsRegressor, PolynomialFeatures, GridSearchCV, preprocessing tools</li>
</ul>

<h2>9. Conclusion</h2>
<p>
The project successfully implements multiple regression techniques to predict insurance charges. Preprocessing, log transformation, feature scaling, and hyperparameter tuning significantly improved model performance.
</p>

</body>
</html>
