# Model Card — Employee Attrition Prediction System

## Purpose

This model is an internship/portfolio prototype designed to demonstrate
machine learning techniques for predicting employee attrition risk.

## Intended Use

The system is intended for:

- Machine learning education
- HR analytics experimentation
- Workforce trend analysis
- Early-warning research
- Demonstration of explainable AI

## Not Intended For

The model should not be used as an automated system to:

- terminate employees
- reject employees
- deny promotions
- penalize employees
- make compensation decisions
- make legally significant employment decisions

## Dataset

The project uses the publicly available IBM HR Analytics Employee Attrition
dataset.

## Target

Attrition:

- 0 = No
- 1 = Yes

## Modeling

The project compares:

- Logistic Regression
- Random Forest

The production candidate is a tuned Random Forest model.

## Feature Engineering

The project creates:

- SatisfactionIndex
- IncomePerJobLevel
- TenureToExperienceRatio
- ManagerTenureRatio
- PromotionWaitRatio
- EarlyCareerFlag
- FrequentJobChangeFlag
- OverTimeFlag

## Sensitive Attributes

Age, Gender and MaritalStatus are excluded from production model scoring.

They may be used for audit analysis.

## Explainability

Tree SHAP is used to provide model-level and individual prediction
explanations.

## Evaluation

The project reports:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion Matrix

## Limitations

The public dataset is limited in size and represents a historical/fictitious
sample rather than a live enterprise workforce.

Model performance on this dataset should not be interpreted as evidence that
the same performance will occur in a real organization.

The model identifies statistical patterns rather than proving causal reasons
for employee departure.

## Responsible AI

Human review is required before any real-world organizational action.

Model outputs should be treated as analytical signals rather than decisions.