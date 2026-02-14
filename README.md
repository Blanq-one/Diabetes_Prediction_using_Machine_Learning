# Comprehensive Diabetes Risk Assessment System

This project implements a diagnostic system to predict the likelihood of diabetes in patients based on medical diagnostic measurements. It utilizes several machine learning models and follows a rigorous pipeline including exploratory data analysis, preprocessing, feature engineering, and hyperparameter optimization.

## Project Contributors
Developed in collaboration with a project teammate.

## Dataset Overview

The system uses the Pima Indians Diabetes Dataset.

### Key Features
* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

### Target Variable
`Outcome`: Indicates the presence (1) or absence (0) of diabetes.

## Implementation Pipeline

### 1. Exploratory Data Analysis (EDA)
Data distributions and class balances were analyzed through various visualizations:
* Histograms and distribution plots for numerical features.
* Correlation matrix heatmaps to identify feature relationships.
* Class imbalance assessment for target distribution.

### 2. Data Preprocessing and Cleaning
* Missing value imputation using class-based medians to maintain data integrity.
* Outlier management using Interquartile Range (IQR) and Local Outlier Factor (LOF) techniques.

### 3. Advanced Feature Engineering
New analytical features were derived to improve model sensitivity:
* Categorical BMI classification (e.g., Underweight, Normal, Obesity).
* Insulin and Glucose level categorization (Normal vs. Abnormal/High).
* Dimensionality management using one-hot encoding for categorical variables.

### 4. Feature Scaling
Numerical features were transformed using `RobustScaler` to minimize the influence of remaining outliers and improve convergence in gradient-based models.

### 5. Model Training and Selection
Base models were evaluated using 10-fold cross-validation to ensure reliable performance estimates:
* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Support Vector Machine
* Gradient Boosting
* LightGBM

### 6. Hyperparameter Optimization
GridSearchCV was employed to tune the most promising models:
* Random Forest
* Gradient Boosting
* LightGBM
Visual analysis of feature importance was conducted for model interpretability.

## Performance Evaluation

| Model | Hyperparameter Tuning | Cross-Validation Accuracy |
| :--- | :--- | :--- |
| Random Forest | Optimized | ~87.5% |
| LightGBM | Optimized | ~89.2% |
| Gradient Boosting | Optimized | ~89.5% |

## Installation and Requirements

To set up the development environment, install the following dependencies:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn lightgbm statsmodels missingno
```
