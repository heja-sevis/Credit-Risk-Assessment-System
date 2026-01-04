# =====================================================
# CreditGuard: End-to-End Credit Risk & Portfolio Analytics
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import plotly.express as px

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Model Explainability
import shap

# -----------------------------------------------------
# Streamlit Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="CreditGuard | Credit Risk Management",
    layout="wide"
)

st.title("💳 CreditGuard: End-to-End Credit Risk & Portfolio Analytics")
st.markdown(
    """
    This application provides **individual credit scoring**, 
    **portfolio-level risk monitoring**, and **stress testing** 
    using a Probability of Default (PD) model.
    """
)

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("credit_risk_dataset.csv")

df = load_data()

# -----------------------------------------------------
# Target & Feature Configuration
# -----------------------------------------------------
TARGET = "loan_status"  # 1 = Default, 0 = Non-default

# Automatically detect categorical features
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols = [c for c in categorical_cols if c != TARGET]

# One-Hot Encoding
df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

X = df_encoded.drop(TARGET, axis=1)
y = df_encoded[TARGET]

# -----------------------------------------------------
# Train Probability of Default (PD) Model
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
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

pd_test = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pd_test)

# -----------------------------------------------------
# Expected Loss Function
# -----------------------------------------------------
def calculate_expected_loss(pd, lgd, ead):
    """
    Expected Loss = PD × LGD × EAD
    """
    return pd * lgd * ead

# -----------------------------------------------------
# Tabs
# -----------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "👤 Individual Credit Scoring",
    "📊 Portfolio Analytics",
    "⚠️ Stress Testing"
])

# =====================================================
# TAB 1 — INDIVIDUAL CREDIT SCORING (UNDERWRITING)
# =====================================================
with tab1:
    st.subheader("Individual Credit Risk Assessment")

    customer_id = st.selectbox(
        "Select Customer ID",
        X.index
    )

    customer_data = X.loc[[customer_id]]
    pd_score = model.predict_proba(customer_data)[0, 1]

    lgd = st.slider(
        "Loss Given Default (LGD)",
        min_value=0.1,
        max_value=1.0,
        value=0.45
    )

    ead = st.number_input(
        "Exposure at Default (EAD)",
        value=100_000
    )

    expected_loss = calculate_expected_loss(
        pd_score,
        lgd,
        ead
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Probability of Default (PD)", f"{pd_score:.2%}")
    col2.metric("Expected Loss", f"{expected_loss:,.0f}")
    col3.metric("Model AUC", f"{auc:.3f}")

    # Credit Decision Logic
    if pd_score < 0.05:
        st.success("✅ CREDIT APPROVED")
    elif pd_score < 0.15:
        st.warning("⚠️ MANUAL REVIEW REQUIRED")
    else:
        st.error("❌ CREDIT REJECTED")

    # SHAP Explainability
    st.subheader("Why did this customer receive this score?")
    
    explainer = shap.TreeExplainer(model)
shap_values = explainer(customer_data)

fig, ax = plt.subplots()
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        feature_names=customer_data.columns,
        data=customer_data.values[0]
    ),
    show=False
)

st.pyplot(fig)

# =====================================================
# TAB 2 — PORTFOLIO ANALYTICS
# =====================================================
with tab2:
    st.subheader("Portfolio Risk Overview")

    df_portfolio = X.copy()
    df_portfolio["PD"] = model.predict_proba(X)[:, 1]
    df_portfolio["LGD"] = 0.45
    df_portfolio["EAD"] = np.random.randint(
        50_000,
        300_000,
        len(df_portfolio)
    )

    df_portfolio["Expected_Loss"] = calculate_expected_loss(
        df_portfolio["PD"],
        df_portfolio["LGD"],
        df_portfolio["EAD"]
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total Expected Loss",
        f"{df_portfolio['Expected_Loss'].sum():,.0f}"
    )
    col2.metric(
        "Average PD",
        f"{df_portfolio['PD'].mean():.2%}"
    )
    col3.metric(
        "Default Rate",
        f"{y.mean():.2%}"
    )

    # Risk Segmentation
    df_portfolio["Risk Segment"] = pd.cut(
        df_portfolio["PD"],
        bins=[0, 0.05, 0.15, 1],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    fig = px.histogram(
        df_portfolio,
        x="PD",
        color="Risk Segment",
        title="Portfolio Risk Segmentation (PD Distribution)"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 3 — STRESS TESTING
# =====================================================
with tab3:
    st.subheader("Macroeconomic Stress Testing")

    inflation_shock = st.slider(
        "Inflation Increase (%)",
        0,
        20,
        5
    )

    unemployment_shock = st.slider(
        "Unemployment Increase (%)",
        0,
        10,
        3
    )

    stress_multiplier = 1 + inflation_shock / 100 + unemployment_shock / 100

    df_stress = df_portfolio.copy()
    df_stress["PD_Stressed"] = np.clip(
        df_stress["PD"] * stress_multiplier,
        0,
        1
    )

    df_stress["Expected_Loss_Stressed"] = calculate_expected_loss(
        df_stress["PD_Stressed"],
        df_stress["LGD"],
        df_stress["EAD"]
    )

    col1, col2 = st.columns(2)
    col1.metric(
        "Base Scenario Expected Loss",
        f"{df_portfolio['Expected_Loss'].sum():,.0f}"
    )
    col2.metric(
        "Stressed Scenario Expected Loss",
        f"{df_stress['Expected_Loss_Stressed'].sum():,.0f}"
    )

    fig = px.bar(
        x=["Base Scenario", "Stressed Scenario"],
        y=[
            df_portfolio["Expected_Loss"].sum(),
            df_stress["Expected_Loss_Stressed"].sum()
        ],
        title="Expected Loss Comparison: Base vs Stress Scenario"
    )

    st.plotly_chart(fig, use_container_width=True)
