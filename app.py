import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load model and features
model = joblib.load('lung_model.joblib')
features = joblib.load('features.joblib')

# Background image
page_bg = f"""
<style>
.stApp {{
    background-image: url("blue lung image.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
    font-weight: bold;
}}
h1, h2, h3, .stButton > button {{
    color: white;
    font-weight: bold;
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# App title
st.markdown("<h1 style='text-align: center;'>🫁 Lung Cancer Detection & Classification</h1>", unsafe_allow_html=True)
st.markdown("### Fill out the details below:")

# Helper function to convert Yes/No
def convert_input(val):
    return 1 if val == "Yes" else 0

# Gender selection
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 30)

# Binary fields
input_data = []
binary_fields = [
    "Smoking", "Yellow Fingers", "Anxiety", "Peer Pressure",
    "Chronic Disease", "Fatigue", "Allergy", "Wheezing",
    "Alcohol Consumption", "Coughing", "Shortness of Breath",
    "Swallowing Difficulty", "Chest Pain"
]

for field in binary_fields:
    val = st.selectbox(f"{field}?", ["Yes", "No"], key=field)
    input_data.append(convert_input(val))

# Convert gender
gender_val = 1 if gender == "Male" else 0
input_data.insert(0, gender_val)
input_data.insert(1, age)

# Buttons
col1, col2, col3 = st.columns(3)
with col1:
    predict_btn = st.button("🔍 Predict")
with col2:
    clear_btn = st.button("🧹 Clear")
with col3:
    exit_btn = st.button("❌ Exit")

# Predict button logic
if predict_btn:
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    confidence = model.predict_proba(input_array).max() * 100

    if prediction == "Positive":
        st.markdown(f"""
            <h2 style='color: red;'>🔴 Positive Lung Cancer ({confidence:.2f}% confidence)</h2>
            <p>⚠️ Please consult a healthcare professional immediately.</p>
            <p>💡 Tip: Avoid smoking, maintain a healthy diet, regular check-ups are crucial.</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <h2 style='color: green;'>🟢 Negative Lung Cancer ({confidence:.2f}% confidence)</h2>
            <p>✅ Stay healthy! No signs of lung cancer detected.</p>
            <p>🥗 Health Tip: Eat fruits, veggies, stay active, and avoid smoking.</p>
        """, unsafe_allow_html=True)

# Clear button logic
if clear_btn:
    st.experimental_rerun()

# Exit button logic
if exit_btn:
    st.markdown("<h3>👋 Thank you for using the app!</h3>", unsafe_allow_html=True)
    st.stop()
