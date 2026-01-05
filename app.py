import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="CreditGuard - Credit Risk Analytics",
    layout="wide"
)

# =====================================================
# Load Data
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("credit_risk_dataset.csv")

df = load_data()

TARGET = "loan_status"

# =====================================================
# Preprocessing
# =====================================================
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols = [c for c in categorical_cols if c != TARGET]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(TARGET, axis=1)
y = df_encoded[TARGET]

# =====================================================
# Train Model
# =====================================================
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, auc

model, auc = train_model(X, y)

# =====================================================
# Expected Loss Function
# =====================================================
def calculate_expected_loss(pd, lgd, ead):
    return pd * lgd * ead

# =====================================================
# Tabs
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Individual Credit Scoring",
    "📊 Portfolio Analytics",
    "⚠️ Stress Testing",
    "🆕 New Customer Evaluation"
])

# =====================================================
# TAB 4 — NEW CUSTOMER EVALUATION
# =====================================================
with tab4:
    st.header("New Customer Credit Evaluation")

    st.markdown(
        "Enter customer information below. "
        "Click **Evaluate Credit Risk** to calculate "
        "**Probability of Default (PD)**, **Expected Loss (EL)** "
        "and the final credit decision."
    )

    # -------------------------
    # User Input Section
    # -------------------------
    with st.form("new_customer_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            person_age = st.number_input(
                "Age", 18, 100, 35, key="age_tab4"
            )
            person_income = st.number_input(
                "Annual Income", 0, 1_000_000, 50_000, key="income_tab4"
            )
            person_emp_length = st.number_input(
                "Employment Length (Years)", 0, 50, 5, key="emp_tab4"
            )

        with col2:
            loan_amnt = st.number_input(
                "Loan Amount", 1_000, 1_000_000, 100_000, key="loan_amt_tab4"
            )
            loan_int_rate = st.number_input(
                "Interest Rate (%)", 0.0, 40.0, 12.5, key="rate_tab4"
            )
            loan_percent_income = loan_amnt / max(person_income, 1)

        with col3:
            cb_person_cred_hist_length = st.number_input(
                "Credit History Length (Years)", 0, 50, 10, key="hist_tab4"
            )

        person_home_ownership = st.selectbox(
            "Home Ownership",
            ["RENT", "OWN", "MORTGAGE", "OTHER"],
            key="home_tab4"
        )

        loan_intent = st.selectbox(
            "Loan Purpose",
            [
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "PERSONAL",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION"
            ],
            key="intent_tab4"
        )

        loan_grade = st.selectbox(
            "Loan Grade",
            ["A", "B", "C", "D", "E", "F", "G"],
            key="grade_tab4"
        )

        cb_person_default_on_file = st.selectbox(
            "Previous Default on File?",
            ["Y", "N"],
            key="default_tab4"
        )

        lgd = st.slider(
            "Loss Given Default (LGD)",
            min_value=0.1,
            max_value=1.0,
            value=0.45,
            key="lgd_tab4"
        )

        ead = st.number_input(
            "Exposure at Default (EAD)",
            value=loan_amnt,
            key="ead_tab4"
        )

        submitted = st.form_submit_button("Evaluate Credit Risk")

    # -------------------------
    # Prediction (ONLY AFTER BUTTON)
    # -------------------------
    if submitted:
        user_input = {
            "person_age": person_age,
            "person_income": person_income,
            "person_emp_length": person_emp_length,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "person_home_ownership": person_home_ownership,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "cb_person_default_on_file": cb_person_default_on_file
        }

        user_df = pd.DataFrame([user_input])
        user_df = pd.get_dummies(user_df)
        user_df = user_df.reindex(columns=X.columns, fill_value=0)

        pd_score = model.predict_proba(user_df)[0, 1]
        expected_loss = calculate_expected_loss(pd_score, lgd, ead)

        st.divider()

        col1, col2, col3 = st.columns(3)
        col1.metric("Probability of Default (PD)", f"{pd_score:.2%}")
        col2.metric("Expected Loss", f"{expected_loss:,.0f}")
        col3.metric("Model AUC", f"{auc:.3f}")

        if pd_score < 0.05:
            st.success("✅ CREDIT APPROVED")
        elif pd_score < 0.15:
            st.warning("⚠️ MANUAL REVIEW REQUIRED")
        else:
            st.error("❌ CREDIT REJECTED")
