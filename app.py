import streamlit as st
import pandas as pd
import joblib


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)


# -----------------------------------
# Load Model
# -----------------------------------

model = joblib.load("model/categorical_nb_model.pkl")


# -----------------------------------
# Title
# -----------------------------------

st.title("👨‍💼 Employee Attrition Prediction")

st.write(
    "Enter employee details below to predict whether the employee "
    "is likely to leave the company or stay."
)


# -----------------------------------
# Input Form
# -----------------------------------

with st.form("employee_form"):

    st.subheader("📋 Employee Information")

    col1, col2, col3 = st.columns(3)

    # -------------------------------
    # Numerical Features
    # -------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        years_at_company = st.number_input(
            "Years at Company",
            min_value=0,
            max_value=100,
            value=5
        )

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=0,
            value=5000
        )

        number_of_promotions = st.number_input(
            "Number of Promotions",
            min_value=0,
            max_value=20,
            value=1
        )

    with col2:

        distance_from_home = st.number_input(
            "Distance from Home",
            min_value=0,
            value=10
        )

        number_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=20,
            value=2
        )

        company_tenure = st.number_input(
            "Company Tenure",
            min_value=0,
            max_value=100,
            value=5
        )


    # --------------------------------
    # Get categorical values
    # --------------------------------

    transformer = model.named_steps["columntransformer"]

    encoder = None

    for name, transformer_obj, columns in transformer.transformers_:

        if name == "cat":
            encoder = transformer_obj
            break

    categorical_values = encoder.categories_


    # --------------------------------
    # Categorical Features
    # --------------------------------

    with col3:

        gender = st.selectbox(
            "Gender",
            categorical_values[0]
        )

        job_role = st.selectbox(
            "Job Role",
            categorical_values[1]
        )

        work_life_balance = st.selectbox(
            "Work-Life Balance",
            categorical_values[2]
        )

        job_satisfaction = st.selectbox(
            "Job Satisfaction",
            categorical_values[3]
        )


    col4, col5, col6 = st.columns(3)

    with col4:

        performance_rating = st.selectbox(
            "Performance Rating",
            categorical_values[4]
        )

        overtime = st.selectbox(
            "Overtime",
            categorical_values[5]
        )

        education_level = st.selectbox(
            "Education Level",
            categorical_values[6]
        )

        marital_status = st.selectbox(
            "Marital Status",
            categorical_values[7]
        )

        job_level = st.selectbox(
            "Job Level",
            categorical_values[8]
        )


    with col5:

        company_size = st.selectbox(
            "Company Size",
            categorical_values[9]
        )

        remote_work = st.selectbox(
            "Remote Work",
            categorical_values[10]
        )

        leadership_opportunities = st.selectbox(
            "Leadership Opportunities",
            categorical_values[11]
        )

        innovation_opportunities = st.selectbox(
            "Innovation Opportunities",
            categorical_values[12]
        )

        company_reputation = st.selectbox(
            "Company Reputation",
            categorical_values[13]
        )


    with col6:

        employee_recognition = st.selectbox(
            "Employee Recognition",
            categorical_values[14]
        )


    # --------------------------------
    # Predict Button
    # --------------------------------

    submit = st.form_submit_button(
        "🔮 Predict Attrition"
    )


# -----------------------------------
# Prediction
# -----------------------------------

if submit:

    input_data = pd.DataFrame({

        "Age": [age],
        "Gender": [gender],
        "Years at Company": [years_at_company],
        "Job Role": [job_role],
        "Monthly Income": [monthly_income],
        "Work-Life Balance": [work_life_balance],
        "Job Satisfaction": [job_satisfaction],
        "Performance Rating": [performance_rating],
        "Number of Promotions": [number_of_promotions],
        "Overtime": [overtime],
        "Distance from Home": [distance_from_home],
        "Education Level": [education_level],
        "Marital Status": [marital_status],
        "Number of Dependents": [number_of_dependents],
        "Job Level": [job_level],
        "Company Size": [company_size],
        "Company Tenure": [company_tenure],
        "Remote Work": [remote_work],
        "Leadership Opportunities": [leadership_opportunities],
        "Innovation Opportunities": [innovation_opportunities],
        "Company Reputation": [company_reputation],
        "Employee Recognition": [employee_recognition]

    })


    # --------------------------------
    # Prediction
    # --------------------------------

    prediction = model.predict(input_data)


    # --------------------------------
    # Probability
    # --------------------------------

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_data)

        confidence = max(probability[0]) * 100


    # --------------------------------
    # Result
    # --------------------------------

    st.subheader("📊 Prediction Result")

    result = prediction[0]

    if str(result).lower() in ["1", "left", "yes"]:

        st.error("⚠️ Employee is likely to LEAVE the company.")

    else:

        st.success("✅ Employee is likely to STAY in the company.")


    if hasattr(model, "predict_proba"):

        st.info(
            f"Prediction Confidence: **{confidence:.2f}%**"
        )


    # --------------------------------
    # Show Input Data
    # --------------------------------

    with st.expander("🔍 View Entered Employee Data"):

        st.dataframe(input_data)