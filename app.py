import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load model and features
model = joblib.load("lung_model (3).joblib")
features = joblib.load("features (3).joblib")

# Set background image
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = image.read()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded.encode('base64').decode()}");
            background-size: cover;
            color: white;
            font-weight: bold;
            font-size: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("blue lung image.jpg")

st.title("💨 Lung Cancer Detection & Classification")
st.markdown("### Please enter the following details:")

def get_yes_no(prompt):
    return st.radio(f"{prompt}", ["Yes", "No"]) == "Yes"

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 35)

smoking = get_yes_no("Do you smoke?")
yellow_fingers = get_yes_no("Do you have yellow fingers?")
anxiety = get_yes_no("Do you feel anxiety?")
peer_pressure = get_yes_no("Do you have peer pressure?")
chronic_disease = get_yes_no("Do you have any chronic disease?")
fatigue = get_yes_no("Do you feel fatigue?")
allergy = get_yes_no("Do you have allergies?")
wheezing = get_yes_no("Do you have wheezing?")
alcohol_consuming = get_yes_no("Do you consume alcohol?")
coughing = get_yes_no("Do you cough often?")
shortness_of_breath = get_yes_no("Do you feel shortness of breath?")
swallowing_difficulty = get_yes_no("Do you face difficulty swallowing?")
chest_pain = get_yes_no("Do you feel chest pain?")

# Convert inputs
input_data = [
    1 if gender == "Male" else 0,
    age,
    int(smoking),
    int(yellow_fingers),
    int(anxiety),
    int(peer_pressure),
    int(chronic_disease),
    int(fatigue),
    int(allergy),
    int(wheezing),
    int(alcohol_consuming),
    int(coughing),
    int(shortness_of_breath),
    int(swallowing_difficulty),
    int(chest_pain)
]

# Buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 Predict"):
        input_array = np.array([input_data])
        prediction = model.predict(input_array)[0]
        confidence = model.predict_proba(input_array).max() * 100

        if prediction == 1:
            st.markdown(f"## 🔴 **Positive Lung Cancer ({confidence:.2f}% confidence)**")
            st.markdown("🚨 Please consult a medical professional immediately.")
            st.markdown("🥦 **Health Tip**: Avoid smoking, get regular checkups, and maintain a healthy diet.")
        else:
            st.markdown(f"## 🟢 **Negative Lung Cancer ({confidence:.2f}% confidence)**")
            st.markdown("✅ Stay healthy! No signs of lung cancer detected.")
            st.markdown("🥗 **Health Tip**: Eat fruits, veggies, stay active, and avoid smoking.")

with col2:
    if st.button("🔄 Clear"):
        st.experimental_rerun()

with col3:
    if st.button("❌ Exit"):
        st.stop()
