import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model_engine import run_complete_pipeline # Arka plandaki motor

st.set_page_config(page_title="CreditGuard Risk Analyzer", layout="wide")

# --- CSS ile Bankacılık Teması ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_stdio=True)

st.title("🏦 CreditGuard: Uçtan Uca Risk Yönetimi & Reject Inference")

# Veri ve Model Yükleme (Cache kullanarak hızı artırıyoruz)
@st.cache_resource
def initialize_engine():
    return run_complete_pipeline('credit_risk_dataset.csv')

# Motoru çalıştır ve modelleri/verileri al
base_model, inferred_model, acc_df, rej_df, metrics = initialize_engine()

# --- SIDEBAR: Navigasyon ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
    menu = st.radio("Menü", ["Portföy Özeti", "Reject Inference Analizi", "Bireysel Kredi Skorlama", "Stres Testi"])

# --- TAB 1: Portföy Özeti ---
if menu == "Portföy Özeti":
    st.subheader("📊 Mevcut Portföy Sağlık Durumu")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam Portföy Büyüklüğü", f"${acc_df['loan_amnt'].sum():,.0f}")
    m2.metric("Ortalama PD (Risk)", f"%{acc_df['loan_status'].mean()*100:.2f}")
    m3.metric("Onay Oranı (Approval Rate)", f"%{(len(acc_df)/(len(acc_df)+len(rej_df)))*100:.1f}")
    m4.metric("Kötü Kredi Sayısı", f"{acc_df['loan_status'].sum()}")

    col_left, col_right = st.columns(2)
    with col_left:
        fig_grade = px.pie(acc_df, names='loan_grade', title="Kredi Derecelerine (Grade) Göre Dağılım", hole=0.4)
        st.plotly_chart(fig_grade)
    with col_right:
        fig_scatter = px.scatter(acc_df, x="person_income", y="loan_amnt", color="loan_status", 
                                 title="Gelir vs Kredi Tutarı (Default Dağılımı)", opacity=0.5)
        st.plotly_chart(fig_scatter)

# --- TAB 2: Reject Inference Analizi ---
elif menu == "Reject Inference Analizi":
    st.subheader("🧠 Reject Inference & Seçim Yanlılığı (Selection Bias)")
    
    st.info("""
    **Metodoloji:** Banka geçmişte bazı müşterileri reddetti. Bu müşterilerin 'etiketi' (ödeme durumu) yok. 
    Aşağıda, reddedilen bu kitleye modelimizle 'Pseudo-label' atayarak modelin tüm evreni tanımasını sağladık.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Model A (Sadece Onaylılar)**")
        st.write(f"Gini Katsayısı: {metrics['base_gini']:.3f}")
        st.progress(metrics['base_gini'])
        
    with col2:
        st.write("**Model B (Reject Inference Sonrası)**")
        st.write(f"Gini Katsayısı: {metrics['inferred_gini']:.3f}")
        st.progress(metrics['inferred_gini'])

    # Seçim Yanlılığı Görseli
    st.write("### Onaylanan vs. Reddedilen Kitle Farkı")
    rej_df['source'] = 'Reddedilenler'
    acc_df['source'] = 'Onaylananlar'
    combined_comp = pd.concat([acc_df, rej_df])
    
    fig_comp = px.box(combined_comp, x='source', y='person_income', color='source', title="Gelir Dağılımı Farkı (Yanlılık Kanıtı)")
    st.plotly_chart(fig_comp, use_container_width=True)

# --- TAB 3: Bireysel Skorlama ---
elif menu == "Bireysel Kredi Skorlama":
    st.subheader("🎯 Akıllı Kredi Karar Destek Sistemi")
    
    with st.expander("Müşteri Bilgilerini Giriniz", expanded=True):
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Yaş", 18, 90, 30)
        income = c2.number_input("Yıllık Gelir ($)", 0, 500000, 50000)
        emp_length = c3.slider("İş Tecrübesi (Yıl)", 0, 40, 5)
        
        intent = c1.selectbox("Kredi Amacı", acc_df['loan_intent'].unique())
        grade = c2.selectbox("Kredi Derecesi (Grade)", sorted(acc_df['loan_grade'].unique()))
        loan_amount = c3.number_input("Talep Edilen Tutar ($)", 0, 100000, 10000)

    if st.button("Risk Analizi Yap"):
        # Model Prediction (Dummy logic for now, will connect to model_engine)
        pd_score = 0.15 # Örnek PD skoru
        
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Temerrüt Olasılığı (PD)", f"%{pd_score*100:.1f}")
            if pd_score < 0.20:
                st.success("KARAR: ONAYLANABİLİR")
            else:
                st.error("KARAR: RED / YÜKSEK RİSK")
        
        with res_col2:
            # Gauge Chart for Risk
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pd_score * 100,
                title = {'text': "Risk Seviyesi"},
                gauge = {'axis': {'range': [0, 100]},
                         'bar': {'color': "darkblue"},
                         'steps' : [
                             {'range': [0, 20], 'color': "green"},
                             {'range': [20, 50], 'color': "yellow"},
                             {'range': [50, 100], 'color': "red"}]}))
            st.plotly_chart(fig_gauge, use_container_width=True)
