import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📡",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-bottom: 15px;
}

.risk-high {
    padding: 20px;
    border-radius: 12px;
    background-color: #ffe5e5;
    border: 1px solid #ff4b4b;
    text-align: center;
}

.risk-low {
    padding: 20px;
    border-radius: 12px;
    background-color: #e6f7ed;
    border: 1px solid #21c16b;
    text-align: center;
}

.probability {
    font-size: 40px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("customer_churn_xgb_pipeline.pkl")


model = load_model()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">📡 Telecom Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict customer churn risk using an optimized XGBoost machine learning model.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


with col2:

    tenure_months = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

    contract_type = st.selectbox(
        "Contract Type",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Credit card",
            "Bank transfer",
            "Mailed check"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


with col3:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "Fiber optic",
            "DSL",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


# =========================================================
# USAGE & BILLING
# =========================================================

st.subheader("💳 Usage & Billing")

col4, col5, col6 = st.columns(3)

with col4:

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


with col5:

    avg_monthly_gb_usage = st.number_input(
        "Average Monthly GB Usage",
        min_value=0.0,
        value=100.0
    )

    num_support_calls = st.number_input(
        "Number of Support Calls",
        min_value=0,
        max_value=20,
        value=1
    )


with col6:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=800.0
    )


# =========================================================
# FEATURE ENGINEERING
# =========================================================

number_of_services = sum([
    phone_service == "Yes",
    multiple_lines == "Yes",
    internet_service != "No",
    online_security == "Yes",
    tech_support == "Yes",
    streaming_tv == "Yes",
    streaming_movies == "Yes"
])


# =========================================================
# PREDICTION
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
)


if predict_button:

    input_data = pd.DataFrame({

        "Age": [age],
        "Senior_Citizen": [senior_citizen],
        "Tenure_Months": [tenure_months],
        "Avg_Monthly_GB_Usage": [avg_monthly_gb_usage],
        "Num_Support_Calls": [num_support_calls],
        "Monthly_Charges": [monthly_charges],
        "Total_Charges": [total_charges],
        "Number_of_Services": [number_of_services],

        "Gender": [gender],
        "Partner": [partner],
        "Dependents": [dependents],
        "Contract_Type": [contract_type],
        "Payment_Method": [payment_method],
        "Paperless_Billing": [paperless_billing],
        "Phone_Service": [phone_service],
        "Multiple_Lines": [multiple_lines],
        "Internet_Service": [internet_service],
        "Online_Security": [online_security],
        "Tech_Support": [tech_support],
        "Streaming_TV": [streaming_tv],
        "Streaming_Movies": [streaming_movies]
    })


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    probability = model.predict_proba(input_data)[0][1]

    threshold = 0.30

    prediction = probability >= threshold


    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    st.subheader("📊 Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)


    # Probability

    with result_col1:

        st.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )


    # Risk

    with result_col2:

        if probability >= 0.60:

            risk = "High Risk"

        elif probability >= 0.30:

            risk = "Medium Risk"

        else:

            risk = "Low Risk"

        st.metric(
            "Risk Level",
            risk
        )


    # Prediction

    with result_col3:

        if prediction:

            st.metric(
                "Prediction",
                "Likely to Churn"
            )

        else:

            st.metric(
                "Prediction",
                "Likely to Stay"
            )


    # =====================================================
    # RISK MESSAGE
    # =====================================================

    if probability >= 0.60:

        st.markdown(
            f"""
            <div class="risk-high">

            <h2>🚨 High Churn Risk</h2>

            <p>
            This customer has a high probability of leaving.
            Immediate retention action is recommended.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning(
            "💡 Recommendation: Contact the customer quickly and consider "
            "a personalized retention offer."
        )


    elif probability >= 0.30:

        st.info(
            "⚠️ Medium Churn Risk: The customer may be at risk. "
            "Consider proactive engagement."
        )


    else:

        st.markdown(
            f"""
            <div class="risk-low">

            <h2>✅ Low Churn Risk</h2>

            <p>
            The customer is currently less likely to churn.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # PROBABILITY BAR
    # =====================================================

    st.subheader("Churn Probability")

    st.progress(float(probability))

    st.caption(
        f"Model threshold: {threshold:.0%}"
    )


    # =====================================================
    # CUSTOMER SUMMARY
    # =====================================================

    st.subheader("📋 Customer Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.write(f"**Age:** {age}")
        st.write(f"**Tenure:** {tenure_months} months")
        st.write(f"**Contract:** {contract_type}")
        st.write(f"**Internet:** {internet_service}")
        st.write(f"**Payment:** {payment_method}")

    with summary_col2:

        st.write(f"**Monthly Charges:** ${monthly_charges:.2f}")
        st.write(f"**Total Charges:** ${total_charges:.2f}")
        st.write(f"**Support Calls:** {num_support_calls}")
        st.write(f"**Number of Services:** {number_of_services}")