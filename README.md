# 👨‍💼 Employee Attrition Prediction

An AI-powered Machine Learning web application that predicts whether an employee is likely to **leave** or **stay** in a company based on employee-related factors.

## 🚀 Project Overview

Employee attrition is a major challenge for organizations. This project uses Machine Learning to analyze employee information and predict the possibility of employee attrition.

The application provides an interactive Streamlit interface where users can enter employee details and instantly get an attrition prediction along with prediction confidence.

## ✨ Features

- 👤 Employee information input
- 💼 Job and career-related features
- 💰 Monthly income analysis
- 😊 Job satisfaction and work-life balance
- ⏰ Overtime information
- 🏠 Remote work details
- 📈 Promotion and job-level information
- 🏢 Company environment factors
- 🔮 Employee attrition prediction
- 🎯 Prediction confidence
- 📊 Interactive Streamlit dashboard
- 📋 View entered employee information

## 🧠 Machine Learning

The application uses a trained Machine Learning model saved as:

`categorical_nb_model.pkl`

The model is loaded using `joblib` and used directly for making predictions.

### Prediction Flow

```text
Employee Information
        ↓
Data Preprocessing
        ↓
Trained ML Model
        ↓
Prediction
        ↓
Stay / Leave
        ↓
Prediction Confidence
