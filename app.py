import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import plotly.express as px

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
categorical_cols = df.select_dtypes(include="object").columns.tolist()
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
        X, y, test_size=0.25, stratify=y, random_state=42
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
# Helper Functions
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
# TAB 1 — INDIVIDUAL CREDIT SCORING
# =====================================================
with tab1:
    st.header("Individual Credit Scoring")

    customer_id = st.slider(
        "Select Customer Index",
        0,
        len(df) - 1,
        0,
        key="cust_slider_tab1"
    )

    customer = df.iloc[[customer_id]]
    customer_encoded = pd.get_dummies(customer)
    customer_encoded = customer_encoded.reindex(columns=X.columns, fill_value=0)

    pd_score = model.predict_proba(customer_encoded)[0, 1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Probability of Default (PD)", f"{pd_score:.2%}")
    col2.metric("Loan Amount", f"{customer['loan_amnt'].values[0]:,.0f}")
    col3.metric("Loan Grade", customer["loan_grade"].values[0])

    if pd_score < 0.05:
        st.success("✅ APPROVED")
    elif pd_score < 0.15:
        st.warning("⚠️ REVIEW REQUIRED")
    else:
        st.error("❌ REJECTED")

# =====================================================
# TAB 2 — PORTFOLIO ANALYTICS
# =====================================================
with tab2:
    st.header("Portfolio Analytics")

    df["PD"] = model.predict_proba(X)[:, 1]
    df["Expected_Loss"] = df["PD"] * 0.45 * df["loan_amnt"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Expected Loss", f"{df['Expected_Loss'].sum():,.0f}")
    col2.metric("Average PD", f"{df['PD'].mean():.2%}")
    col3.metric("Default Rate", f"{df[TARGET].mean():.2%}")

    fig = px.histogram(
        df,
        x="PD",
        nbins=40,
        title="PD Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 3 — STRESS TESTING
# =====================================================
with tab3:
    st.header("Stress Testing")

    stress_factor = st.slider(
        "Macroeconomic Stress Factor",
        1.0,
        2.0,
        1.2,
        key="stress_slider_tab3"
    )

    stressed_pd = np.clip(df["PD"] * stress_factor, 0, 1)
    stressed_el = stressed_pd * 0.45 * df["loan_amnt"]

    col1, col2 = st.columns(2)
    col1.metric("Baseline EL", f"{df['Expected_Loss'].sum():,.0f}")
    col2.metric("Stressed EL", f"{stressed_el.sum():,.0f}")

    fig = px.histogram(
        stressed_pd,
        nbins=40,
        title="Stressed PD Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 4 — NEW CUSTOMER EVALUATION
# =====================================================
with tab4:
    st.header("New Customer Credit Evaluation")

    with st.form("new_customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", 18, 100, 35, key="age4")
            income = st.number_input("Annual Income", 0, 1_000_000, 50_000, key="inc4")
            emp = st.number_input("Employment Length", 0, 50, 5, key="emp4")

        with col2:
            loan = st.number_input("Loan Amount", 1_000, 1_000_000, 100_000, key="loan4")
            rate = st.number_input("Interest Rate (%)", 0.0, 40.0, 12.5, key="rate4")
            lpi = loan / max(income, 1)

        with col3:
            hist = st.number_input("Credit History Length", 0, 50, 10, key="hist4")

        home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], key="home4")
        intent = st.selectbox(
            "Loan Purpose",
            ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
            key="intent4"
        )
        grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"], key="grade4")
        default = st.selectbox("Previous Default?", ["Y", "N"], key="def4")

        lgd = st.slider("LGD", 0.1, 1.0, 0.45, key="lgd4")
        ead = st.number_input("EAD", value=loan, key="ead4")

        submit = st.form_submit_button("Evaluate Credit Risk")

    if submit:
        user = {
            "person_age": age,
            "person_income": income,
            "person_emp_length": emp,
            "loan_amnt": loan,
            "loan_int_rate": rate,
            "loan_percent_income": lpi,
            "cb_person_cred_hist_length": hist,
            "person_home_ownership": home,
            "loan_intent": intent,
            "loan_grade": grade,
            "cb_person_default_on_file": default
        }

        user_df = pd.DataFrame([user])
        user_df = pd.get_dummies(user_df)
        user_df = user_df.reindex(columns=X.columns, fill_value=0)

        pd_score = model.predict_proba(user_df)[0, 1]
        el = calculate_expected_loss(pd_score, lgd, ead)

        col1, col2 = st.columns(2)
        col1.metric("Probability of Default", f"{pd_score:.2%}")
        col2.metric("Expected Loss", f"{el:,.0f}")

        if pd_score < 0.05:
            st.success("✅ APPROVED")
        elif pd_score < 0.15:
            st.warning("⚠️ REVIEW REQUIRED")
        else:
            st.error("❌ REJECTED")
