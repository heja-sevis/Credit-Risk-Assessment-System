# =========================
# CreditGuard Streamlit App
# =========================

import streamlit as st
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Explainability
import shap

# -------------------------
# Streamlit Config
# -------------------------
st.set_page_config(page_title="CreditGuard | Risk Management", layout="wide")
st.title("💳 CreditGuard: Uçtan Uca Kredi Risk ve Portföy Analitiği")

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("/mnt/data/credit_risk_dataset.csv")
    return df

df = load_data()

# -------------------------
# Feature Engineering
# -------------------------
TARGET = "default"

categorical_cols = ["sector", "region"]
numerical_cols = [c for c in df.columns if c not in categorical_cols + [TARGET]]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop(TARGET, axis=1)
y = df[TARGET]

# -------------------------
# Train PD Model
# -------------------------
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
pd_test = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, pd_test)

# -------------------------
# Expected Loss Functions
# -------------------------
def calculate_el(pd, lgd, ead):
    return pd * lgd * ead

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs([
    "🧍 Bireysel Skorlama (Underwriting)",
    "📊 Portföy Analizi",
    "⚠️ Stres Testi"
])

# ======================================================
# TAB 1 – INDIVIDUAL CREDIT SCORING
# ======================================================
with tab1:
    st.subheader("Bireysel Müşteri Kredi Riski")

    customer_id = st.selectbox("Müşteri Seç", X.index)
    customer = X.loc[[customer_id]]

    pd_score = model.predict_proba(customer)[0, 1]

    lgd = st.slider("LGD (Teminat Sonrası Kayıp Oranı)", 0.1, 1.0, 0.45)
    ead = st.number_input("EAD (Kalan Anapara)", value=100000)

    el = calculate_el(pd_score, lgd, ead)

    col1, col2, col3 = st.columns(3)
    col1.metric("PD", f"{pd_score:.2%}")
    col2.metric("Expected Loss", f"{el:,.0f} ₺")
    col3.metric("Model AUC", f"{auc:.3f}")

    if pd_score < 0.05:
        st.success("✅ KREDİ ONAY")
    elif pd_score < 0.15:
        st.warning("⚠️ İNCELEME GEREKİYOR")
    else:
        st.error("❌ KREDİ RED")

    # SHAP Explainability
    st.subheader("📌 Neden Bu Skoru Aldı?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(customer)

    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[1][0],
            base_values=explainer.expected_value[1],
            feature_names=customer.columns,
            data=customer.values[0]
        ),
        show=False
    )
    st.pyplot(fig)

# ======================================================
# TAB 2 – PORTFOLIO ANALYSIS
# ======================================================
with tab2:
    st.subheader("Portföy Genel Görünümü")

    df_portfolio = X.copy()
    df_portfolio["PD"] = model.predict_proba(X)[:, 1]
    df_portfolio["LGD"] = 0.45
    df_portfolio["EAD"] = np.random.randint(50_000, 300_000, len(df_portfolio))
    df_portfolio["EL"] = calculate_el(
        df_portfolio["PD"],
        df_portfolio["LGD"],
        df_portfolio["EAD"]
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Expected Loss", f"{df_portfolio['EL'].sum():,.0f} ₺")
    col2.metric("Ortalama PD", f"{df_portfolio['PD'].mean():.2%}")
    col3.metric("Batık Oranı", f"{y.mean():.2%}")

    # Risk Segmentation
    df_portfolio["Risk Segment"] = pd.cut(
        df_portfolio["PD"],
        bins=[0, 0.05, 0.15, 1],
        labels=["Düşük", "Orta", "Yüksek"]
    )

    fig = px.histogram(
        df_portfolio,
        x="PD",
        color="Risk Segment",
        title="Risk Segmentasyonu (PD Dağılımı)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 3 – STRESS TESTING
# ======================================================
with tab3:
    st.subheader("Makroekonomik Stres Testi")

    inflation_shock = st.slider("Enflasyon Şoku (%)", 0, 20, 5)
    unemployment_shock = st.slider("İşsizlik Artışı (%)", 0, 10, 3)

    stress_multiplier = 1 + inflation_shock / 100 + unemployment_shock / 100

    df_stress = df_portfolio.copy()
    df_stress["PD_Stressed"] = np.clip(
        df_stress["PD"] * stress_multiplier,
        0,
        1
    )
    df_stress["EL_Stressed"] = calculate_el(
        df_stress["PD_Stressed"],
        df_stress["LGD"],
        df_stress["EAD"]
    )

    col1, col2 = st.columns(2)
    col1.metric("Baz EL", f"{df_portfolio['EL'].sum():,.0f} ₺")
    col2.metric("Stresli EL", f"{df_stress['EL_Stressed'].sum():,.0f} ₺")

    fig = px.line(
        pd.DataFrame({
            "Normal": df_portfolio["EL"],
            "Stres": df_stress["EL_Stressed"]
        }).mean(),
        title="Stres Senaryosu EL Karşılaştırması"
    )
    st.plotly_chart(fig, use_container_width=True)
