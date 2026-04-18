# Telco Customer Churn Analysis & Prediction
## Project Overview
#### Customer churn is a major problem in the telecom industry. This project analyzes customer data to identify the key factors that lead to churn and builds a machine learning model to predict whether a customer is likely to leave.

The goal is to move beyond raw data and provide actionable insights that can help businesses improve customer retention.

📊 The "Accuracy Trap" & Performance
In this project, I intentionally moved away from a "high accuracy" model (73%) that missed half of the churners, in favor of a Tuned Random Forest that prioritizes Recall.

Key Finding: By adjusting the classification threshold to 0.40, I increased the model's sensitivity. While this lowered the overall accuracy, it successfully captured 88% of churners, which is significantly more valuable for a business looking to prevent customer loss.

Model Insights (What Drives Churn?)
Using the model's coefficients, I identified the primary drivers for customer behavior:

Top Churn Drivers: High MonthlyCharges and TotalCharges.

Top Retention Drivers: Two-year Contracts and long Tenure.

Irrelevant Features: Age was found to have near-zero impact on the prediction.

🛠️ Tech Stack
Python 3.10+

Scikit-Learn: Logistic Regression & Random Forest Classifier.

Pandas/NumPy: Data manipulation and feature engineering.

Joblib: Model and Scaler serialization.

FastAPI (Planned): For deploying the model as a web service.

📂 Project Structure
telco_churn_model.pkl: The trained "brain" of the Random Forest.

telco_churn_scaler.pkl: The saved StandardScaler state to ensure consistent data preprocessing in production.

main.ipynb: Full exploratory data analysis and model tuning process.

## How to use

import joblib
import pandas as pd

#### 1. Load the model and scaler
model = joblib.load('telco_churn_model.pkl')
scaler = joblib.load('telco_churn_scaler.pkl')

#### 2. Prepare raw data (ensure column order matches training)
features = ['tenure', 'MonthlyCharges', 'TotalCharges', ...] 
new_data = pd.DataFrame([[12, 70.5, 846.0, ...]], columns=features)

### 3. Scale and Predict with the custom 0.40 threshold
scaled_data = scaler.transform(new_data)
probs = model.predict_proba(scaled_data)[:, 1]
prediction = (probs > 0.40).astype(int)

print(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")

📈 Future Improvements ("If needed")
[ ] Deploy the model as a FastAPI endpoint.

[ ] Build a Streamlit Dashboard for real-time customer "What-If" analysis.

[ ] Experiment with XGBoost to see if we can maintain the 88% Recall while boosting Accuracy back above 70%.
