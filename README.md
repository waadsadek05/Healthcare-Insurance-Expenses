<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1> Insurance Charges Prediction</h1>

<h2>📌 Project Overview</h2>
<p>This project predicts <span class="highlight">medical insurance charges</span> for individuals using Machine Learning models.
Features include age, sex, BMI, number of children, smoking status, and geographical region.</p>

<h2>📂 Dataset Description</h2>
<p>The dataset contains <strong>1,337 entries</strong> after cleaning and removing duplicates.</p>

<table>
    <tr><th>Feature</th><th>Description</th></tr>
    <tr><td>age</td><td>Age of the policyholder</td></tr>
    <tr><td>sex</td><td>Gender (encoded)</td></tr>
    <tr><td>bmi</td><td>Body Mass Index</td></tr>
    <tr><td>children</td><td>Number of children covered</td></tr>
    <tr><td>smoker</td><td>Smoking status (encoded)</td></tr>
    <tr><td>region</td><td>Geographical region (One-Hot Encoded)</td></tr>
    <tr><td>charges</td><td>Insurance charges (target)</td></tr>
</table>

<h2>⚙️ Data Preprocessing</h2>
<ol>
    <li>Removed duplicates and checked for null values</li>
    <li>Label encoded <code>sex</code> and <code>smoker</code></li>
    <li>One-Hot encoded <code>region</code> to prevent false hierarchies</li>
    <li>Log-transformed <code>charges</code> to reduce skewness</li>
    <li>Split data: 80% training, 20% testing</li>
    <li>Scaled numerical features using <code>StandardScaler</code></li>
</ol>

<h2>📊 Exploratory Data Analysis (EDA)</h2>
<h3>1️⃣ Outlier Detection & Distribution</h3>
<img src="plots/Box Plot to detect Outliers.png" alt="Boxplots for age, BMI, charges">

<h3>2️⃣ Charges Histogram</h3>
<img src="plots/Charges Histogram.png" alt="Charges distribution histogram">

<h3>3️⃣ BMI vs Charges Scatter Plot</h3>
<img src="plots/BMI vs Charges (Colored by Smoker).png" alt="BMI vs Charges colored by smoker">

<h2>🤖 Machine Learning Models</h2>

<h3>1️⃣ Polynomial Linear Regression</h3>
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

<h3>2️⃣ K-Nearest Neighbors (KNN) with GridSearchCV</h3>
<pre><code>
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# Pipeline: scaling, feature selection, KNN
# GridSearchCV to find best hyperparameters
</code></pre>
<p><strong>Best Parameters:</strong> n_neighbors=30, weights='uniform', p=2, select__k=4 <br>
<strong>Performance:</strong> R² = <span class="highlight">85.13%</span>, RMSE = <span class="highlight">0.0371</span></p>

<h2>💾 Example Predictions</h2>
<table>
<tr><th>Actual Charges</th><th>Polynomial Prediction</th><th>KNN Prediction</th></tr>
<tr><td>$3,500</td><td>$3,480</td><td>$3,510</td></tr>
<tr><td>$16,800</td><td>$16,750</td><td>$16,900</td></tr>
</table>

<h2>🛠 Technologies Used</h2>
<ul>
    <li>Python</li>
    <li>Pandas, NumPy</li>
    <li>Scikit-learn</li>
    <li>Matplotlib, Seaborn</li>
</ul>

<h2>📁 Project Structure</h2>
<pre><code>insurance-charges-prediction/
│
├── insurance.csv
├── charges_prediction.ipynb
├── plots/
│   ├── boxplots.png
│   ├── charges_hist.png
│   └── bmi_vs_charges.png
├── polynomial_model.pkl
├── knn_model.pkl
└── README.html
</code></pre>

<h2>🚀 Future Improvements</h2>
<ul>
    <li>Try ensemble methods like Random Forest or Gradient Boosting</li>
    <li>Experiment with deep learning regression models</li>
    <li>Deploy as real-time prediction API (Flask/FastAPI)</li>
    <li>Include additional features like medical history or lifestyle</li>
</ul>

<h2>👩‍💻 Author</h2>
<p><strong>Waad Sadek</strong><br>
Machine Learning enthusiast building predictive models for healthcare applications.</p>

</body>
</html>
