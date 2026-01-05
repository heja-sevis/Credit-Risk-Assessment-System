# CreditGuard — Professional Credit Risk & Portfolio Analytics

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import lightgbm as lgb
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditGuard | Credit Risk Management",
    layout="wide"
)

st.title("💳 CreditGuard: Credit Risk & Portfolio Analytics Platform")

# ------------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("credit_risk_dataset.csv")

df = load_data()

TARGET = "loan_status"

categorical_cols = [
    c for c in df.select_dtypes(include="object").columns
    if c != TARGET
]

# ------------------------------------------------------------------------------
# MODEL TRAINING (CACHED)
# ------------------------------------------------------------------------------
@st.cache_resource
def train_pd_model(df):

    df_encoded = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    X = df_encoded.drop(TARGET, axis=1)
    y = df_encoded[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
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

    auc = roc_auc_score(
        y_test,
        model.predict_proba(X_test)[:, 1]
    )

    return model, auc, X


model, auc, X = train_pd_model(df)

# ------------------------------------------------------------------------------
# UTILITY
# ------------------------------------------------------------------------------
def expected_loss(pd, lgd, ead):
    return pd * lgd * ead

# ------------------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Individual Credit Scoring",
    "📊 Portfolio Analytics",
    "⚠️ Stress Testing",
    "🆕 New Customer Evaluation"
])

# ==============================================================================
# TAB 1 — INDIVIDUAL CREDIT SCORING
# ==============================================================================
with tab1:
    st.subheader("Individual Credit Risk Assessment")

    customer_id = st.selectbox("Select Customer ID", X.index)

    customer = X.loc[[customer_id]]
    pd_score = model.predict_proba(customer)[0, 1]

    lgd = st.slider("Loss Given Default (LGD)", 0.1, 1.0, 0.45)
    ead = st.number_input("Exposure at Default (EAD)", value=100_000)

    el = expected_loss(pd_score, lgd, ead)

    col1, col2, col3 = st.columns(3)
    col1.metric("Probability of Default", f"{pd_score:.2%}")
    col2.metric("Expected Loss", f"{el:,.0f}")
    col3.metric("Model AUC", f"{auc:.3f}")

    # Decision
    if pd_score < 0.05:
        st.success("✅ CREDIT APPROVED")
    elif pd_score < 0.15:
        st.warning("⚠️ MANUAL REVIEW REQUIRED")
    else:
        st.error("❌ CREDIT REJECTED")

    # SHAP EXPLANATION
    st.subheader("Why did this customer receive this score?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(customer)

    shap_df = pd.DataFrame({
        "Feature": customer.columns,
        "SHAP Value": shap_values.values[0]
    })

    shap_df["Abs"] = shap_df["SHAP Value"].abs()
    shap_df = shap_df.sort_values("Abs", ascending=False).head(10)

    shap_df["Impact"] = np.where(
        shap_df["SHAP Value"] > 0,
        "Increases PD",
        "Decreases PD"
    )

    fig = px.bar(
        shap_df.sort_values("SHAP Value"),
        x="SHAP Value",
        y="Feature",
        orientation="h",
        color="Impact",
        title="Top Factors Affecting PD"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 2 — PORTFOLIO ANALYTICS
# ==============================================================================
with tab2:
    st.subheader("Portfolio Risk Overview")

    df_portfolio = X.copy()
    df_portfolio["PD"] = model.predict_proba(X)[:, 1]
    df_portfolio["LGD"] = 0.45
    df_portfolio["EAD"] = df["loan_amnt"].values

    df_portfolio["Expected_Loss"] = expected_loss(
        df_portfolio["PD"],
        df_portfolio["LGD"],
        df_portfolio["EAD"]
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Expected Loss", f"{df_portfolio['Expected_Loss'].sum():,.0f}")
    col2.metric("Average PD", f"{df_portfolio['PD'].mean():.2%}")
    col3.metric("Observed Default Rate", f"{df[TARGET].mean():.2%}")

    # Risk Buckets
    df_portfolio["Risk Segment"] = pd.cut(
        df_portfolio["PD"],
        bins=[0, 0.05, 0.15, 1],
        labels=["Low", "Medium", "High"]
    )

    # Risk Matrix
    fig = px.scatter(
        df_portfolio,
        x="PD",
        y="EAD",
        size="Expected_Loss",
        color="Risk Segment",
        title="Risk Matrix (PD × EAD)",
        labels={"PD": "Probability of Default", "EAD": "Exposure at Default"}
    )

    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 3 — STRESS TESTING (REALISTIC)
# ==============================================================================
with tab3:
    st.subheader("Macroeconomic Stress Testing")

    inflation = st.slider("Inflation Shock (%)", 0, 30, 10)

    df_stress = X.copy()

    # Income shock
    df_stress["person_income"] *= (1 - inflation / 100)

    # Recalculate ratios
    df_stress["loan_percent_income"] = (
        df_stress["loan_amnt"] /
        df_stress["person_income"].clip(lower=1)
    )

    pd_stressed = model.predict_proba(df_stress)[:, 1]

    df_portfolio["PD_Stressed"] = pd_stressed
    df_portfolio["EL_Stressed"] = expected_loss(
        pd_stressed,
        df_portfolio["LGD"],
        df_portfolio["EAD"]
    )

    col1, col2 = st.columns(2)
    col1.metric(
        "Base Expected Loss",
        f"{df_portfolio['Expected_Loss'].sum():,.0f}"
    )
    col2.metric(
        "Stressed Expected Loss",
        f"{df_portfolio['EL_Stressed'].sum():,.0f}"
    )

    fig = px.bar(
        x=["Base", "Stressed"],
        y=[
            df_portfolio["Expected_Loss"].sum(),
            df_portfolio["EL_Stressed"].sum()
        ],
        title="Expected Loss: Base vs Stress Scenario"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 4 — NEW CUSTOMER
# ==============================================================================
with tab4:
    st.subheader("New Customer Credit Evaluation")

    with st.form("new_customer"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", 18, 100, 35)
            income = st.number_input("Annual Income", 0, 1_000_000, 50_000)
            emp = st.number_input("Employment Length", 0, 40, 5)

        with col2:
            loan = st.number_input("Loan Amount", 1_000, 1_000_000, 100_000)
            rate = st.number_input("Interest Rate (%)", 0.0, 40.0, 12.5)
            lpi = loan / max(income, 1)

        with col3:
            hist = st.number_input("Credit History Length", 0, 50, 10)

        home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        intent = st.selectbox(
            "Loan Purpose",
            ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )
        grade = st.selectbox("Loan Grade", list("ABCDEFG"))
        default = st.selectbox("Previous Default?", ["Y", "N"])

        lgd = st.slider("LGD", 0.1, 1.0, 0.45)
        submit = st.form_submit_button("Evaluate")

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
        el = expected_loss(pd_score, lgd, loan)

        st.metric("Probability of Default", f"{pd_score:.2%}")
        st.metric("Expected Loss", f"{el:,.0f}")

        if pd_score < 0.05:
            st.success("✅ APPROVED")
        elif pd_score < 0.15:
            st.warning("⚠️ REVIEW REQUIRED")
        else:
            st.error("❌ REJECTED")
