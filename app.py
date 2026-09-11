import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- Main App ---------- */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 50%,
            #f8fafc 100%
        );
    }

    /* ---------- Header ---------- */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1e1b4b;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 25px;
    }

    /* ---------- Section Cards ---------- */

    .section-card {
        background: white;
        padding: 20px 24px;
        border-radius: 18px;
        margin-bottom: 20px;
        box-shadow: 0 5px 20px rgba(30, 41, 59, 0.08);
        border: 1px solid #e2e8f0;
    }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #312e81;
        margin-bottom: 10px;
    }

    /* ---------- Prediction Cards ---------- */

    .prediction-card {
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }

    .stay-card {
        background: linear-gradient(
            135deg,
            #ecfdf5,
            #d1fae5
        );
        border: 2px solid #10b981;
    }

    .leave-card {
        background: linear-gradient(
            135deg,
            #fff1f2,
            #ffe4e6
        );
        border: 2px solid #ef4444;
    }

    .prediction-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .confidence-text {
        font-size: 18px;
        color: #475569;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #1e1b4b 0%,
            #312e81 50%,
            #4338ca 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* ---------- Button ---------- */

    div.stButton > button,
    div.stFormSubmitButton > button {
        width: 100%;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 700;
        padding: 12px;
        border: none;
    }

    /* ---------- Metrics ---------- */

    div[data-testid="stMetric"] {
        background: white;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }

    /* ---------- Divider ---------- */

    hr {
        border: none;
        height: 1px;
        background: #cbd5e1;
        margin: 25px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("model/categorical_nb_model.pkl")


model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <h1 style="text-align:center;">👨‍💼</h1>
        <h2 style="text-align:center;">Attrition AI</h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 📌 About")

    st.write(
        """
        This AI-powered application predicts whether
        an employee is likely to **leave** or **stay**
        in the company.
        """
    )

    st.markdown("---")

    st.markdown("### 🧠 Prediction Flow")

    st.write(
        """
        👤 Employee Data  
        ↓  
        ⚙️ ML Model  
        ↓  
        📊 Prediction  
        ↓  
        🎯 Attrition Risk
        """
    )

    st.markdown("---")

    st.markdown("### 🎯 Risk Guide")

    st.write("🟢 **Stay** — Employee likely to remain")

    st.write("🔴 **Leave** — Employee may leave")

    st.markdown("---")

    st.caption("Built with Python • Streamlit • Machine Learning")


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">👨‍💼 Employee Attrition Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered employee retention risk assessment dashboard'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# GET CATEGORICAL VALUES FROM MODEL
# =========================================================

transformer = model.named_steps["columntransformer"]

encoder = None

for name, transformer_obj, columns in transformer.transformers_:

    if name == "cat":
        encoder = transformer_obj
        break


categorical_values = encoder.categories_


# =========================================================
# EMPLOYEE FORM
# =========================================================

with st.form("employee_form"):

    # =====================================================
    # PERSONAL & CAREER INFORMATION
    # =====================================================

    st.markdown(
        """
        <div class="section-card">
        <div class="section-title">
        👤 Personal & Career Information
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.slider(
            "🎂 Age",
            min_value=18,
            max_value=70,
            value=30
        )

        gender = st.radio(
            "⚧ Gender",
            categorical_values[0],
            horizontal=True
        )

        years_at_company = st.slider(
            "🏢 Years at Company",
            min_value=0,
            max_value=40,
            value=5
        )

    with col2:

        job_role = st.selectbox(
            "💼 Job Role",
            categorical_values[1]
        )

        job_level = st.selectbox(
            "📈 Job Level",
            categorical_values[8]
        )

        education_level = st.selectbox(
            "🎓 Education Level",
            categorical_values[6]
        )

    with col3:

        marital_status = st.selectbox(
            "💍 Marital Status",
            categorical_values[7]
        )

        number_of_dependents = st.slider(
            "👨‍👩‍👧 Number of Dependents",
            min_value=0,
            max_value=10,
            value=2
        )

        number_of_promotions = st.slider(
            "🚀 Number of Promotions",
            min_value=0,
            max_value=10,
            value=1
        )


    st.markdown("---")


    # =====================================================
    # JOB & COMPENSATION
    # =====================================================

    st.markdown(
        """
        <div class="section-card">
        <div class="section-title">
        💼 Job & Compensation
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        monthly_income = st.slider(
            "💰 Monthly Income",
            min_value=0,
            max_value=50000,
            value=5000,
            step=500
        )

        performance_rating = st.radio(
            "⭐ Performance Rating",
            categorical_values[4],
            horizontal=True
        )

    with col2:

        job_satisfaction = st.radio(
            "😊 Job Satisfaction",
            categorical_values[3],
            horizontal=True
        )

        work_life_balance = st.radio(
            "⚖️ Work-Life Balance",
            categorical_values[2],
            horizontal=True
        )

    with col3:

        overtime = st.radio(
            "⏰ Overtime",
            categorical_values[5],
            horizontal=True
        )

        distance_from_home = st.slider(
            "📍 Distance from Home",
            min_value=0,
            max_value=100,
            value=10
        )


    st.markdown("---")


    # =====================================================
    # WORK ENVIRONMENT
    # =====================================================

    st.markdown(
        """
        <div class="section-card">
        <div class="section-title">
        🏢 Work Environment
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        company_size = st.radio(
            "🏭 Company Size",
            categorical_values[9],
            horizontal=True
        )

        company_tenure = st.slider(
            "📅 Company Tenure",
            min_value=0,
            max_value=50,
            value=5
        )

    with col2:

        remote_work = st.radio(
            "🏠 Remote Work",
            categorical_values[10],
            horizontal=True
        )

        leadership_opportunities = st.radio(
            "👑 Leadership Opportunities",
            categorical_values[11],
            horizontal=True
        )

    with col3:

        innovation_opportunities = st.radio(
            "💡 Innovation Opportunities",
            categorical_values[12],
            horizontal=True
        )

        company_reputation = st.radio(
            "🏆 Company Reputation",
            categorical_values[13],
            horizontal=True
        )


    # =====================================================
    # RECOGNITION
    # =====================================================

    st.markdown("---")

    employee_recognition = st.selectbox(
        "🏅 Employee Recognition",
        categorical_values[14]
    )


    # =====================================================
    # SUBMIT
    # =====================================================

    st.markdown("---")

    submit = st.form_submit_button(
        "🔮  PREDICT EMPLOYEE ATTRITION"
    )


# =========================================================
# PREDICTION
# =========================================================

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


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = model.predict(input_data)

    result = prediction[0]


    # =====================================================
    # PROBABILITY
    # =====================================================

    confidence = None

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_data)

        confidence = max(probability[0]) * 100


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if str(result).lower() in ["1", "left", "yes"]:

        st.markdown(
            f"""
            <div class="prediction-card leave-card">
                <div class="prediction-title">
                    🔴 Employee is likely to LEAVE
                </div>
                <div class="confidence-text">
                    The model predicts a higher probability of employee attrition.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if confidence is not None:

            st.metric(
                "🎯 Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                int(confidence)
            )

        st.warning(
            "⚠️ Consider reviewing employee satisfaction, workload, "
            "career growth and workplace conditions."
        )


    else:

        st.markdown(
            f"""
            <div class="prediction-card stay-card">
                <div class="prediction-title">
                    🟢 Employee is likely to STAY
                </div>
                <div class="confidence-text">
                    The model predicts a higher probability of employee retention.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if confidence is not None:

            st.metric(
                "🎯 Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                int(confidence)
            )

            if confidence >= 80:
                st.balloons()

        st.success(
            "✅ The employee currently shows a lower predicted attrition risk."
        )


    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    st.markdown("---")

    with st.expander("🔍 View Entered Employee Data"):

        st.dataframe(
            input_data,
            use_container_width=True
        )