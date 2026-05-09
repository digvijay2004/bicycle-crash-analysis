# Crash Severity Prediction using Machine Learning

## Overview
This project predicts severe non-motorized crash occurrence involving pedestrians and cyclists using machine learning models.

The project was originally implemented in R and later migrated to Python.

## Dataset
Michigan non-motorized crash dataset (2009–2021)

## Machine Learning Models
- Logistic Regression
- LASSO Logistic Regression
- Random Forest
- XGBoost

## Features
- Data preprocessing
- Handling imbalanced data
- Train/test split
- ROC-AUC evaluation
- Feature importance analysis
- Visualization outputs

## Technologies Used
- Python
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib

## Outputs
- ROC Curves
- Feature Importance Graphs
- Classification Reports

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python crash_analysis_python.py
```

## Results
XGBoost achieved the best performance with an AUC score of approximately 0.70.

## Authors
- Digvijay Bhagat
- Sia Mwende