import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load model and features
model = joblib.load('lung_model (5).joblib')
features = joblib.load('features (4).joblib')

# Set page config
st.set_page_config(page_title="Lung Cancer Predictor", layout="centered")

# Set background image with custom CSS
def set_bg():
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url('blue lung image.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
            font-weight: bold;
        }}
        h1, h2, h3, label, .stButton > button {{
            color: white;
            font-size: 22px;
        }}
        </style>
    """, unsafe_allow_html=True)

set_bg()

st.title("💙 Lung Cancer Prediction App")

# Define input function
def get_input():
    input_data = []
    gender = st.selectbox("Gender", ["Male", "Female"])
    input_data.append(1 if gender == "Male" else 0)

    yes_no_fields = [
        "Do you have yellow fingers?",
        "Do you feel fatigue?",
        "Do you have allergies?",
        "Do you experience wheezing?",
        "Do you have shortness of breath?",
        "Do you experience swallowing difficulty?",
        "Do you have chest pain?"
    ]

    for field in yes_no_fields:
        value = st.radio(field, ["Yes", "No"], horizontal=True)
        input_data.append(1 if value == "Yes" else 0)

    return np.array([input_data])

# Prediction and result display
def predict_and_display(input_data):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    confidence = round(np.max(proba) * 100, 2)

    if prediction == 1:
        st.error("🩺 **Prediction: Positive Lung Cancer**")
        st.warning(f"📊 **Confidence: {confidence}%**")
        st.markdown("### ⚠️ **Health Tip**")
        st.markdown("- See a doctor immediately.")
        st.markdown("- Avoid smoking, eat healthy, exercise regularly.")
        st.markdown("- Increase intake of fruits & vegetables like **broccoli, garlic, spinach**.")
    else:
        st.success("✅ **Prediction: Negative Lung Cancer**")
        st.info(f"📊 **Confidence: {confidence}%**")
        st.markdown("### 💡 **Stay Healthy Tips**")
        st.markdown("- Keep a smoke-free environment.")
        st.markdown("- Exercise regularly and drink plenty of water.")
        st.markdown("- Annual checkups are recommended.")

# Buttons
input_data = None
if st.button("🔍 Predict"):
    input_data = get_input()
    predict_and_display(input_data)

if st.button("🧹 Clear"):
    st.experimental_rerun()

if st.button("❌ Exit"):
    st.stop()
