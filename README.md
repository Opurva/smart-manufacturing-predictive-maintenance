# SmartFactory AI: Predictive Maintenance Intelligence Platform

A company-inspired smart manufacturing predictive maintenance prototype that uses machine learning to estimate machine failure risk, machine health score, risk level, and maintenance recommendations.

---

## Project Overview

Manufacturing companies depend on continuous machine operation. Unexpected machine failures can cause production downtime, maintenance delays, quality issues, and operational losses.

This project demonstrates how machine learning can support predictive maintenance by analyzing machine operating parameters such as:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Engineered machine stress indicators

The system predicts machine failure risk and provides actionable maintenance insights through an interactive Streamlit dashboard.

## Live Demo
https://smart-manufacturing-predictive-maintenance.streamlit.app/
---

## Data Usage Note

This project is designed as a company-inspired predictive maintenance prototype.

Due to confidentiality and limited access to real company machine logs, representative industrial machine sensor records were used for model development and testing.

The same pipeline can be adapted to real company machines when live sensor data or historical maintenance logs become available.

---

## Key Features

- Machine failure prediction
- Failure probability estimation
- Machine health score generation
- Low / Medium / High risk classification
- Preventive maintenance recommendation
- Prediction history tracking
- Model performance analytics
- Explainable AI insights using feature importance and SHAP
- Professional Streamlit dashboard

---

## Machine Learning Workflow

1. Data collection and problem framing
2. Exploratory Data Analysis
3. Data cleaning and preprocessing
4. Feature engineering
5. Model preparation
6. Model training
7. Model evaluation
8. Explainable AI analysis
9. Streamlit dashboard development

---

## Models Trained

The following machine learning models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The best model was selected based on evaluation metrics such as Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

---

## Engineered Features

To make the system more industry-relevant, additional machine health indicators were created:

- Temperature Difference
- Mechanical Power
- Wear Rate
- Thermal Stress

These features help capture relationships between machine load, heat, tool degradation, and failure risk.

---

## Dashboard Pages

### Home
Project overview, platform capabilities, and industrial ML workflow.

### Predictive Maintenance Engine
Accepts machine operating inputs and returns:

- Failure probability
- Machine health score
- Risk level
- Maintenance recommendation
- Health gauge
- Prediction history

### Analytics
Displays model evaluation results, metric comparison, performance heatmap, and training time analysis.

### Explainable AI
Shows key machine signals influencing predictions and SHAP-based feature contribution analysis.

### About
Explains the project context, data usage note, features, and technology stack.

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- GitHub

---

## Project Structure

```text
smart-manufacturing-predictive-maintenance/

├── dashboard/
│   ├── app.py
│   └── assets/
│       └── styles.css
│
├── dataset/
│   ├── ai4i2020.csv
│   └── ai4i2020_processed.csv
│
├── models/
│   ├── preprocessing/
│   └── trained_models/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Preparation.ipynb
│   ├── 04_Model_Training.ipynb
│   ├── 05_Model_Evaluation.ipynb
│   └── 06_Model_Explainability.ipynb
│
├── reports/
├── requirements.txt
├── README.md
└── LICENSE