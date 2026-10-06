🚀 Algonive Internship — Data Science & Machine Learning Projects

<p align="center">A collection of end-to-end Data Science, Machine Learning & Deep Learning projects developed during my Algonive Internship.

</p><p align="center">"Python" (https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
"Machine Learning" (https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge)
"Deep Learning" (https://img.shields.io/badge/Deep%20Learning-PyTorch-red?style=for-the-badge&logo=pytorch&logoColor=white)
"Streamlit" (https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
"GitHub" (https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)

</p>---

👨‍💻 About This Repository

This repository contains my completed Data Science and Machine Learning projects developed as part of my Algonive Internship.

The projects demonstrate practical implementation of complete Machine Learning workflows — from data preprocessing and feature engineering to model training, evaluation, explainability and interactive application development.

📌 Projects Included

#| Project| Domain| Core Technology
01| Employee Attrition Prediction System| HR Analytics| Scikit-Learn
02| Cryptocurrency Price Prediction System| Time-Series Forecasting| PyTorch + LSTM

---

📂 Repository Structure

Algonive_Internship_Tasks/
│
├── 📁 Employee_Attrition_Prediction_System/
│   ├── 📄 app.py
│   ├── 📄 README.md
│   ├── 📄 MODEL_CARD.md
│   ├── 📄 requirements.txt
│   ├── 📁 data/
│   ├── 📁 models/
│   ├── 📁 reports/
│   └── 📁 src/
│
├── 📁 Cryptocurrency_Price_Prediction_System/
│   ├── 📄 app.py
│   ├── 📄 README.md
│   ├── 📄 requirements.txt
│   ├── 📁 data/
│   ├── 📁 models/
│   ├── 📁 artifacts/
│   ├── 📁 reports/
│   └── 📁 src/
│
└── 📄 README.md

---

🧠 Project 01 — Employee Attrition Prediction System

«Predicting employee attrition risk using Machine Learning and Explainable AI.»

🎯 Project Objective

Employee turnover can have a significant impact on organizations through recruitment costs, productivity loss and workforce instability.

This project develops a Machine Learning system that estimates whether an employee is likely to leave the organization based on relevant HR and employee characteristics.

The system combines predictive modeling with explainability to make the predictions easier to understand.

---

🔍 What the System Does

          HR Employee Data
                 │
                 ▼
        ┌──────────────────┐
        │  Data Cleaning   │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │ Feature          │
        │ Engineering      │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │ Preprocessing &  │
        │ Encoding         │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │ Model Training   │
        └────────┬─────────┘
                 ▼
       ┌─────────┴─────────┐
       ▼                   ▼
 Logistic Regression   Random Forest
       │                   │
       └─────────┬─────────┘
                 ▼
       Hyperparameter Tuning
                 │
                 ▼
        Threshold Optimization
                 │
                 ▼
       Model Evaluation
                 │
                 ▼
      SHAP / Feature Analysis
                 │
                 ▼
       Streamlit Application

---

⚙️ Key Features

📊 Predictive Analytics

Predict employee attrition risk using trained classification models.

🧹 Data Preprocessing

Handles numerical and categorical employee attributes before model training.

⚖️ Class Imbalance Handling

Techniques are incorporated to address differences between attrition and non-attrition classes.

🌲 Random Forest Modeling

A Random Forest classifier is used to capture nonlinear relationships within the HR data.

📈 Model Optimization

The project includes:

- Hyperparameter tuning
- Cross-validation
- Decision threshold optimization

🔎 Explainable AI

Model behavior is analyzed using:

- Feature importance
- SHAP explainability
- Individual prediction analysis

🖥️ Interactive Dashboard

A Streamlit interface provides:

- Individual employee prediction
- Employee risk scoring
- Batch scoring
- Workforce analytics

---

🛠️ Technology Stack

Python
│
├── Pandas
├── NumPy
├── Scikit-Learn
│   ├── Logistic Regression
│   └── Random Forest
│
├── SHAP
├── Matplotlib
├── Seaborn
├── Plotly
│
└── Streamlit

---

📁 Important Files

File| Description
"app.py"| Interactive Streamlit application
"src/preprocessing.py"| Data preprocessing pipeline
"src/train_model.py"| Model training and evaluation
"src/explain_model.py"| Model explainability
"models/attrition_model.joblib"| Trained model
"MODEL_CARD.md"| Model documentation
"requirements.txt"| Project dependencies

---

📈 Project 02 — Cryptocurrency Price Prediction System

«Time-series cryptocurrency forecasting using an LSTM Deep Learning model.»

🎯 Project Objective

Cryptocurrency markets generate highly sequential and volatile data.

This project uses historical cryptocurrency market information and engineered technical indicators to build an LSTM-based time-series prediction system.

The model learns patterns from historical sequences and generates cryptocurrency price predictions.

---

🔄 Prediction Pipeline

             Historical Market Data
                       │
                       ▼
              Data Cleaning
                       │
                       ▼
             Feature Engineering
                       │
                       ▼
          Technical Indicators
                       │
                       ▼
           Leakage-Safe Scaling
                       │
                       ▼
           60-Day Time Sequences
                       │
                       ▼
        Chronological Data Split
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          Train     Validation   Test
             │         │         │
             └─────────┼─────────┘
                       ▼
                 LSTM Network
                       │
                       ▼
                    Training
                       │
                       ▼
                   Prediction
                       │
                       ▼
                Model Evaluation
                       │
                       ▼
             Streamlit Dashboard

---

📊 Market Features

The project works with historical market variables such as:

- Open
- High
- Low
- Close
- Volume

Additional engineered indicators include:

- SMA
- EMA
- RSI
- MACD
- ATR
- Bollinger Band Width
- Returns
- Volatility
- Momentum
- Volume-based indicators

---

🧠 LSTM Model

The cryptocurrency prediction system uses a Long Short-Term Memory (LSTM) neural network.

Why LSTM?

LSTM networks are designed to process sequential data and retain information from previous time steps.

For cryptocurrency forecasting, this allows the model to learn temporal relationships within historical market sequences.

Input Sequence

The model uses a:

60-day historical sequence

to predict the target cryptocurrency price.

---

🏗️ Model Training

The project uses:

- PyTorch
- LSTM
- Dropout
- Adam Optimizer
- Huber Loss
- Learning Rate Scheduling
- Early Stopping
- Model Checkpointing

Dataset Split

70% ───────── Training
15% ───────── Validation
15% ───────── Testing

The split is chronological to reduce the risk of future information leaking into the training process.

---

📊 Model Evaluation

The final test evaluation produced the following results:

Metric| Result
RMSE| $28,105.74
MAE| $22,635.30
MAPE| 22.57%
R² Score| -1.1736
Directional Accuracy| 51.51%

📌 Interpretation

The evaluation metrics are included to provide a transparent assessment of the model's predictive performance.

The Directional Accuracy of 51.51% indicates that the model correctly identified the direction of price movement slightly more than half of the time on the evaluated test set.

The relatively high error and negative R² also demonstrate the difficulty of accurately forecasting highly volatile cryptocurrency prices and are important limitations of the current model.

---

🔬 Additional Analysis

The project also includes analysis such as:

- Actual vs Predicted Prices
- Prediction Error Analysis
- Naive Baseline Comparison
- Directional Movement Analysis
- Model Performance Evaluation
- Prediction Uncertainty Analysis
- MC-Dropout-based uncertainty estimation

---

🖥️ Interactive Applications

Both projects include Streamlit-based interfaces.

Employee Attrition

Employee Information
        ↓
Prediction Engine
        ↓
Attrition Probability
        ↓
Risk Classification
        ↓
Explainability

Cryptocurrency Prediction

Market Data
     ↓
Feature Engineering
     ↓
LSTM Model
     ↓
Price Prediction
     ↓
Performance Analysis

---

🧰 Complete Technology Stack

Category| Technologies
Language| Python
Data Processing| Pandas, NumPy
Machine Learning| Scikit-Learn
Deep Learning| PyTorch
Time-Series Modeling| LSTM
Explainable AI| SHAP
Visualization| Matplotlib, Seaborn, Plotly
Web Application| Streamlit
Development| Jupyter Notebook, VS Code
Version Control| Git, GitHub

---

🚀 Installation & Usage

1️⃣ Clone the Repository

git clone <YOUR_GITHUB_REPOSITORY_URL>

cd Algonive_Internship_Tasks

---

2️⃣ Run Employee Attrition Project

cd Employee_Attrition_Prediction_System

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

---

3️⃣ Run Cryptocurrency Prediction Project

cd Cryptocurrency_Price_Prediction_System

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

---

🎓 Skills Demonstrated

Through these projects, I developed practical experience in:

- Data Science
- Machine Learning
- Deep Learning
- Data Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Classification
- Time-Series Forecasting
- LSTM Networks
- Model Evaluation
- Hyperparameter Optimization
- Explainable AI
- SHAP
- Imbalanced Dataset Handling
- Technical Indicator Engineering
- Streamlit Development
- Git & GitHub
- End-to-End ML Project Development

---

💡 Key Learning Outcomes

Employee Attrition Project

«Learned how to transform HR data into a complete predictive analytics system while considering model performance, interpretability and practical usability.»

Cryptocurrency Project

«Learned how to build a time-series forecasting pipeline using sequential data, technical indicators and an LSTM deep learning model.»

---

🏆 Internship Deliverables

Deliverable| Status
Task 1 — Employee Attrition Prediction| ✅ Completed
Task 2 — Cryptocurrency Price Prediction| ✅ Completed
Source Code| ✅
Documentation| ✅
Model Training| ✅
Model Evaluation| ✅
Interactive Applications| ✅
GitHub Repository| ✅

---

👨‍💻 About Me

Avishek Yati

🎓 MCA — Data Science
🏫 Lovely Professional University

Interested in:

Data Science • Machine Learning • Artificial Intelligence • Python • Deep Learning • Data Analytics

---

🤝 Acknowledgement

I would like to thank Algonive for providing the opportunity to work on practical Data Science and Machine Learning projects as part of the internship.

The internship provided valuable hands-on experience in applying Machine Learning concepts to practical problem statements.

---

⚠️ Disclaimer

The cryptocurrency prediction system is developed for educational and demonstration purposes only.

Cryptocurrency markets are highly volatile and unpredictable. The model's predictions should not be considered financial, trading or investment advice.

---

<p align="center">⭐ If you find this repository useful, consider giving it a star!

Built with Python • Machine Learning • Deep Learning • Curiosity

</p>
