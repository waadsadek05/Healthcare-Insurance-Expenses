# Healthcare-Insurance-Expenses
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title># Healthcare-Insurance-Expenses Insurance Charges Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1000px;
            margin: 20px auto;
            padding: 0 20px;
            color: #333;
            background-color: #f9f9f9;
        }
        h1, h2, h3 {
            color: #1a73e8;
        }
        code {
            background-color: #eee;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.95em;
        }
        pre {
            background-color: #eee;
            padding: 10px;
            border-radius: 6px;
            overflow-x: auto;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }
        table, th, td {
            border: 1px solid #ccc;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        .highlight {
            color: #d9534f;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Insurance Charges Prediction Project</h1>

    <h2>Project Overview</h2>
    <p>
        This project focuses on predicting medical insurance charges for individuals using Machine Learning models. 
        The dataset contains demographic and health-related features including age, sex, BMI, number of children, smoking status, and region.
    </p>

    <h2>Dataset</h2>
    <p>Sample data columns:</p>
    <ul>
        <li><strong>age</strong> - Age of the policyholder</li>
        <li><strong>sex</strong> - Gender (encoded)</li>
        <li><strong>bmi</strong> - Body Mass Index</li>
        <li><strong>children</strong> - Number of children covered</li>
        <li><strong>smoker</strong> - Smoking status (encoded)</li>
        <li><strong>region</strong> - Geographical region (One-Hot Encoded)</li>
        <li><strong>charges</strong> - Insurance charges (target)</li>
    </ul>

    <h2>Data Preprocessing</h2>
    <ol>
        <li>Check for nulls and duplicates.</li>
        <li>Label encode categorical features <code>sex</code> and <code>smoker</code>.</li>
        <li>One-Hot encode the <code>region</code> feature to avoid false hierarchies.</li>
        <li>Log-transform the <code>charges</code> target to reduce skewness.</li>
        <li>Split the dataset into training (80%) and testing (20%) sets.</li>
        <li>Scale numerical features (<code>age</code>, <code>bmi</code>, <code>children</code>) using <code>StandardScaler</code>.</li>
    </ol>

    <h2>Exploratory Data Analysis</h2>
    <p>Key plots included:</p>
    <ul>
        <li>Box plots for <code>age</code>, <code>bmi</code>, and <code>charges</code> to detect outliers.</li>
        <li>Histogram of <code>charges</code> to check skewness.</li>
        <li>Scatter plot of <code>bmi</code> vs <code>charges</code> colored by smoker status.</li>
    </ul>

    <h2>Models Implemented</h2>
    <h3>Polynomial Linear Regression</h3>
    <pre><code>
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)
y_pred_poly_log = poly_model.predict(X_test_poly)
y_pred_poly_final = np.expm1(y_pred_poly_log)
    </code></pre>
    <p>
        <strong>Performance:</strong> R² = <span class="highlight">88.56%</span>, MAE = <span class="highlight">$0.19K</span>
    </p>

    <h3>K-Nearest Neighbors (KNN) with GridSearchCV</h3>
    <pre><code>
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# Pipeline with scaling, feature selection, and KNN
# Grid search to optimize parameters
</code></pre>
    <p>
        <strong>Best Parameters:</strong> n_neighbors=30, weights='uniform', p=2, select__k=4 <br>
        <strong>Performance:</strong> R² = <span class="highlight">85.13%</span>, RMSE = <span class="highlight">0.0371</span>
    </p>

    <h2>Results Comparison</h2>
    <p>
        Both models show strong predictive performance. Polynomial regression slightly outperforms KNN in terms of R² and MAE.
    </p>
    <ul>
        <li>Polynomial Regression captures nonlinear relationships.</li>
        <li>KNN performs well but is sensitive to the number of neighbors and scaling.</li>
    </ul>

    <h2>Visualization</h2>
    <p>
        Scatter plots were generated to compare actual charges vs predicted charges for both models and to analyze residuals for the polynomial model.
    </p>

    <h2>Tools & Libraries</h2>
    <ul>
        <li>Pandas, NumPy</li>
        <li>Matplotlib, Seaborn</li>
        <li>Scikit-learn (LinearRegression, KNeighborsRegressor, GridSearchCV, PolynomialFeatures, preprocessing tools)</li>
    </ul>

    <h2>Conclusion</h2>
    <p>
        The project successfully builds and compares multiple regression models to predict insurance charges. Feature preprocessing, scaling, encoding, and log-transformation significantly improve model performance.
    </p>
</body>
</html>
