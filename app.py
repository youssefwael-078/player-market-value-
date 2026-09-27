import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import base64
import os

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="FIFA Player Price Predictor",
    page_icon="⚽",
    layout="centered"
)

BASE_DIR = os.path.dirname(__file__)

# ===============================
# Load Models, Scaler & Metrics
# ===============================
@st.cache_resource(show_spinner=False)
def load_assets():
    rf_model = joblib.load(os.path.join(BASE_DIR, "player_price_rf_model.pkl"))
    nn_model = joblib.load(os.path.join(BASE_DIR, "player_price_nn_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

    metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    return rf_model, nn_model, scaler, metrics

try:
    rf_model, nn_model, scaler, metrics = load_assets()
except FileNotFoundError as e:
    st.error(
        "Model files not found. Make sure the following files are next to this app: "
        "player_price_rf_model.pkl, player_price_nn_model.pkl, scaler.pkl, "
        "model_metrics.json (run train_model.py first to generate them)."
    )
    st.stop()

MODELS = {
    "Random Forest": rf_model,
    "Neural Network": nn_model,
}

# ===============================
# Background & Custom CSS
# ===============================
def apply_custom_styles(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .stApp::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(10, 15, 30, 0.55);
            z-index: -1;
        }}

        [data-testid="stSidebar"] {{
            background: rgba(15, 23, 42, 0.85) !important;
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }}

        div.stButton > button {{
            width: 100%;
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 12px 24px;
            border-radius: 12px;
            border: none;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
            transition: all 0.3s ease;
        }}

        div.stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6);
            background: linear-gradient(135deg, #34d399, #10b981);
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

image_path = os.path.join(BASE_DIR, "images", "stadium.jpg")
if os.path.exists(image_path):
    apply_custom_styles(image_path)

# ===============================
# Sidebar: Model Choice
# ===============================
st.sidebar.markdown("<h2 style='color: #4ade80;'>🤖 Choose Model</h2>", unsafe_allow_html=True)

model_name = st.sidebar.selectbox(
    "Prediction algorithm",
    list(MODELS.keys())
)
selected_model = MODELS[model_name]

# Show the selected model's performance (regression metrics, not classification accuracy)
if model_name in metrics:
    r2 = metrics[model_name]["r2"]
    mae = metrics[model_name]["mae_eur"]
    badge_html = (
        '<div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; '
        'border-radius: 10px; padding: 12px; margin-top: 8px; text-align:center;">'
        '<div style="color:#4ade80; font-weight:700; font-size:13px;">R² SCORE</div>'
        f'<div style="color:white; font-size:26px; font-weight:900;">{r2*100:.1f}%</div>'
        f'<div style="color:#94a3b8; font-size:12px; margin-top:4px;">MAE: €{mae:,.0f}</div>'
        '</div>'
    )
    st.sidebar.markdown(badge_html, unsafe_allow_html=True)
else:
    st.sidebar.info("No performance data found for this model.")

st.sidebar.markdown("---")

# ===============================
# Sidebar: Player Attributes
# ===============================
st.sidebar.markdown("<h2 style='color: #4ade80;'>⚽ Player Attributes</h2>", unsafe_allow_html=True)

age = st.sidebar.slider("🎂 Age", 16, 45, 22)
potential = st.sidebar.slider("🚀 Potential Rating", 40, 99, 80)

st.sidebar.markdown("---")
st.sidebar.markdown("<h4 style='color: #cbd5e1;'>Technical Stats</h4>", unsafe_allow_html=True)

pace = st.sidebar.slider("⚡ Pace", 1, 99, 70)
shooting = st.sidebar.slider("🎯 Shooting", 1, 99, 70)
passing = st.sidebar.slider("🎯 Passing", 1, 99, 70)
dribbling = st.sidebar.slider("🌀 Dribbling", 1, 99, 70)
defending = st.sidebar.slider("🛡 Defending", 1, 99, 50)
physic = st.sidebar.slider("💪 Physic", 1, 99, 70)

# ===============================
# Header Section
# ===============================
st.markdown(
    """
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #ffffff; font-size: 2.8rem; font-weight: 800; text-shadow: 0 4px 12px rgba(0,0,0,0.6);'>
            ⚽ FIFA Player Price Predictor
        </h1>
        <p style='color:#cbd5e1;'>Pick a model from the sidebar and compare the prediction and accuracy</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# Prediction & Card Display
# ===============================
if st.button(" Predict Market Value"):
    input_data = pd.DataFrame({
        'age': [age],
        'potential': [potential],
        'pace': [pace],
        'shooting': [shooting],
        'passing': [passing],
        'dribbling': [dribbling],
        'defending': [defending],
        'physic': [physic]
    })

    # Random Forest was trained on unscaled data.
    # Neural Network was trained on scaled data - must use the same scaler.
    if model_name == "Neural Network":
        model_input = scaler.transform(input_data)
    else:
        model_input = input_data

    log_price = selected_model.predict(model_input)[0]
    price = np.expm1(log_price)

    # Price Formatting
    if price >= 1_000_000:
        price_str = f"{price/1_000_000:.2f}M €"
    elif price >= 1_000:
        price_str = f"{price/1_000:.2f}K €"
    else:
        price_str = f"{price:.2f} €"

    acc_line = ""
    if model_name in metrics:
        r2_value = metrics[model_name]["r2"] * 100
        mae_value = metrics[model_name]["mae_eur"]
        acc_line = (
            f'<div style="margin-top:14px; font-size:12px; color:#94a3b8;">'
            f'R\u00b2 Score: <b style="color:#4ade80;">{r2_value:.1f}%</b>'
            f' &nbsp;|&nbsp; MAE: <b style="color:#4ade80;">€{mae_value:,.0f}</b>'
            f'</div>'
        )

    card_html = (
        '<div style="margin-top:30px; background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), '
        'rgba(15, 23, 42, 0.95)); border: 2px solid #f59e0b; padding: 28px; border-radius: 24px; '
        'color: white; text-align: center; box-shadow: 0 12px 35px rgba(245, 158, 11, 0.25); '
        'max-width: 440px; margin-left: auto; margin-right: auto; backdrop-filter: blur(10px);">'
        '<div style="font-size:14px; text-transform:uppercase; letter-spacing:2px; color:#f59e0b; font-weight:700;">'
        'Estimated Value</div>'
        f'<div style="font-size:42px; font-weight:900; margin:12px 0; color:#ffffff; '
        f'text-shadow:0 2px 10px rgba(255,255,255,0.2);">💰 {price_str}</div>'
        '<div style="display:flex; justify-content:space-around; margin-top:20px; padding-top:15px; '
        'border-top:1px solid rgba(255,255,255,0.1); font-size:13px; color:#cbd5e1;">'
        f'<div>AGE: <b style="color:white;">{age}</b></div>'
        f'<div>POT: <b style="color:#4ade80;">{potential}</b></div>'
        '</div>'
        f'{acc_line}'
        '</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)
