import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean Modern CSS
st.markdown("""
<style>
    .stApp {
        background: #0e1117;
        color: #fafafa;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #aaaaaa;
        margin-bottom: 3rem;
    }
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1.5rem 0;
    }
    .section-header {
        font-size: 1.3rem;
        color: #FF6B6B;
        font-weight: 600;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid #333;
        padding-bottom: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #FF4B4B, #FF6B6B);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        margin-top: 1.5rem;
    }
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.4rem;
        border: 2px solid;
    }
    .fraud-card {
        background: rgba(255, 0, 0, 0.1);
        border-color: #FF4444;
    }
    .safe-card {
        background: rgba(0, 255, 0, 0.1);
        border-color: #44FF44;
    }
    .prob-text {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('credit_fraud_model.pkl')

model = load_model()

# Header
st.markdown("<h1 class='main-title'>Credit Card Fraud Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time fraud detection using Random Forest • 99.9%+ Accuracy<br><strong>Educational Project • Anonymized Features</strong></p>", unsafe_allow_html=True)

# Input Section
with st.container():
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>🔍 Transaction Features (Anonymized PCA Components)</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**V1 to V14**")
        v1_to_v14 = []
        for i in range(1, 15):
            val = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}", label_visibility="collapsed")
            v1_to_v14.append(val)
    
    with col2:
        st.markdown("**V15 to V28 + Amount**")
        v15_to_v28 = []
        for i in range(15, 29):
            val = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}", label_visibility="collapsed")
            v15_to_v28.append(val)
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)

    st.markdown("</div>", unsafe_allow_html=True)

# Predict
if st.button("🔍 Analyze Transaction Risk"):
    features = np.array([v1_to_v14 + v15_to_v28 + [amount]])
    
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class='glass-container result-card fraud-card'>
            <h2>🚨 Fraudulent Transaction Detected</h2>
            <div class='prob-text' style='color: #FF4444;'>{prob:.1%}</div>
            <p><strong>High Risk</strong> — Recommend blocking this transaction</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='glass-container result-card safe-card'>
            <h2>✅ Genuine Transaction</h2>
            <div class='prob-text' style='color: #44FF44;'>{prob:.1%}</div>
            <p><strong>Low Risk</strong> — Transaction appears safe</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.progress(prob)

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 4rem; color: #666; padding: 1rem;'>
        <hr style='border-color: #333;'>
        Built with Streamlit • Model: Random Forest • Dataset: Credit Card Fraud 2023<br>
        <small>For educational purposes only</small>
    </div>
""", unsafe_allow_html=True)