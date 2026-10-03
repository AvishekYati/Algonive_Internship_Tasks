# Algonive Employee Attrition Prediction System

An end-to-end machine learning and explainable AI system designed to estimate employee attrition risk using publicly available HR analytics data.

## Project Overview

Employee attrition creates substantial operational challenges for organizations because employee departures can affect productivity, recruitment costs, team continuity and workforce planning.

This project develops an end-to-end predictive analytics solution that:

* analyzes HR and workplace attributes
* engineers business-oriented features
* compares machine learning models
* tunes a Random Forest classifier
* handles class imbalance
* optimizes the classification threshold
* evaluates model performance on a held-out test set
* provides SHAP-based explainability
* performs demographic audit analysis
* provides an interactive Streamlit dashboard
* supports single-employee and batch prediction

## Business Objective

The objective is to create an analytical early-warning system that can help HR and workforce analytics teams identify patterns associated with employee attrition.

The model is intended as decision support and not as an automated employment decision system.

## Dataset

The project uses the IBM HR Analytics Employee Attrition dataset.

The dataset contains employee-level HR and workplace attributes including job information, compensation, satisfaction measures, experience and other workforce characteristics.

## Machine Learning Workflow

```text
Raw HR Data
    |
    v
Data Quality Checks
    |
    v
Exploratory Analysis
    |
    v
Feature Engineering
    |
    v
Train / Validation / Test Split
    |
    +-----------------------+
    |                       |
    v                       v
Logistic Regression     Random Forest
    |                       |
    +----------+------------+
               |
               v
       Validation Analysis
               |
               v
       Threshold Optimization
               |
               v
         Final Test Set
               |
        +------+------+
        |      |      |
        v      v      v
      XAI   Metrics  Audit
        |
        v
    Streamlit App
```

## Engineered Features

The project creates additional analytical features including:

* SatisfactionIndex
* IncomePerJobLevel
* TenureToExperienceRatio
* ManagerTenureRatio
* PromotionWaitRatio
* EarlyCareerFlag
* FrequentJobChangeFlag
* OverTimeFlag

## Models

### Logistic Regression

Used as the interpretable baseline classification model.

### Random Forest

Used as the production candidate because it can model nonlinear relationships and interactions among employee/workplace attributes.

Hyperparameter search is performed using cross-validation.

## Evaluation

The project evaluates:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion Matrix

The classification threshold is optimized on validation data instead of assuming a fixed 0.50 threshold.

## Explainable AI

Tree SHAP is used to explain:

* global feature importance
* individual prediction contributions

This helps users understand which model inputs contributed most strongly to a prediction.

## Responsible AI

Age, Gender and MaritalStatus are excluded from the production prediction model and retained separately for audit/analytics.

Model outputs should not be treated as evidence that an employee will leave or as an instruction to take employment action.

## Streamlit Application

The application contains three major modules:

### 1. Single Employee Risk

Enter employee/workplace information and receive an estimated attrition probability.

### 2. Batch Scoring

Upload an HR CSV and receive predicted probabilities and early-warning flags for multiple employees.

### 3. Workforce Analytics

Explore sample-level attrition patterns across job roles and overtime.

## Project Structure

```text
Algonive_Employee_Attrition_Prediction_System/
│
├── app.py
├── README.md
├── MODEL_CARD.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── models/
│   └── attrition_model.joblib
│
├── reports/
│   ├── classification_report.csv
│   ├── test_metrics.csv
│   ├── validation_model_comparison.csv
│   ├── feature_importance.csv
│   ├── fairness_audit.csv
│   ├── best_params.json
│   └── figures/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── precision_recall_curve.png
│       ├── feature_importance.png
│       └── shap_summary.png
│
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── train_model.py
    └── explain_model.py
```

## Installation

```bash
git clone <your-github-repository-url>

cd Algonive_Employee_Attrition_Prediction_System

python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run:

```bash
python src\train_model.py
```

## Generate SHAP Analysis

Run:

```bash
python src\explain_model.py
```

## Launch Application

Run:

```bash
streamlit run app.py
```

## Results

The repository contains automatically generated:

* model comparison results
* final test metrics
* classification report
* confusion matrix
* ROC curve
* precision-recall curve
* feature importance chart
* fairness audit
* SHAP summary

Actual values are generated by the training pipeline and should not be hardcoded into the documentation.

## Technologies

Python
Pandas
NumPy
Scikit-learn
Random Forest
Logistic Regression
SHAP
Matplotlib
Seaborn
Plotly
Streamlit
Joblib
Git/GitHub

## Future Improvements

Possible production extensions include:

* real organizational HR data with appropriate privacy controls
* MLflow experiment tracking
* automated CI/CD
* model drift monitoring
* calibration monitoring
* cloud deployment
* role-specific dashboards
* human feedback loops
* scheduled retraining

## Internship

Developed as an Algonive internship project.

GitHub repository:

`<insert repository URL>`

LinkedIn:

`<insert LinkedIn profile>`
