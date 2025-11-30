# Employee-Attrition-Prediction-Using-Machine-Learning

Employee Attrition Prediction Using Machine Learning

This project applies multiple supervised machine learning models to predict employee attrition using a large HR dataset. The study compares four algorithms—Logistic Regression, Support Vector Machine (SVM), Gaussian Naive Bayes, and Gradient Boosting—and evaluates their performance before and after class balancing and hyperparameter tuning.

This work was completed as part of the Applied Machine Learning (AML) module for the MSc in Data Science & Business Analytics at Asia Pacific University.

# ⭐ Project Overview

Employee attrition is a critical challenge affecting organizational productivity, costs, and workforce stability. Predicting which employees are likely to leave enables HR teams to make informed decisions, improve working conditions, and retain talent.

# This project explores:

Data cleaning and preprocessing

Exploratory Data Analysis (EDA)

Feature engineering

SMOTE for class balancing

Model implementation

Hyperparameter tuning (GridSearchCV & Optuna)

Model evaluation using accuracy, precision, recall, and F1-score

# 📂 Dataset

Source: Kaggle Employee Attrition Dataset

Size: 74,498 records, 24 features

Target: Attrition (binary: Stayed / Left)

# Feature Types

Numerical: Age, Monthly Income, Years at Company, Distance from Home, etc.

Categorical: Gender, Job Role, Education Level, Work-Life Balance, Overtime, etc.

# 🔧 Technologies & Libraries

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

SMOTE (Imbalanced-learn)

Optuna

Google Colab

# 📊 Methodology
1. Data Preprocessing

Removed irrelevant columns (e.g., Employee ID)

One-hot encoding for categorical features

Label encoding for binary variables

MinMax scaling

Train-test split (80/20)

SMOTE for class balancing

# 2. Exploratory Data Analysis (EDA)

Checked missing values (none found)

Distribution plots for numerical and categorical variables

Attrition distribution visualization

Insights on job satisfaction, promotions, company tenure, etc.

# 3. Models Implemented

Logistic Regression

Gaussian Naive Bayes

Support Vector Machine

Gradient Boosting

Each model was tested in four stages:

Before class balancing

Before class balancing + tuning

After class balancing

After class balancing + tuning

| Model                  | Best Accuracy |
| ---------------------- | ------------- |
| **Gradient Boosting**  | **76.50%**    |
| Logistic Regression    | 75.75%        |
| Support Vector Machine | 75.64%        |
| Gaussian Naive Bayes   | 73.87%        |

# Key Findings

Gradient Boosting achieved the highest accuracy after tuning.

Class balancing improved LR and SVM but slightly reduced NB performance.

Hyperparameter tuning consistently improved all models.

Gradient Boosting and SVM showed strong predictive power for this dataset.

# 📈 Why Gradient Boosting Performed Best

Learns sequentially from errors

Handles complex relationships in large datasets

Reduces bias through boosting

Performs well after tuning with Optuna
