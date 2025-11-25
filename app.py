import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ================================
# 1. Cargar modelo
# ================================
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_heartdisease_pipeline.pkl")

modelo = cargar_modelo()

st.set_page_config(page_title="Predicción de Enfermedad Cardíaca", layout="centered")

st.title("❤️ Predicción de Enfermedad Cardíaca")
st.write("Ingrese los datos del paciente para obtener la predicción.")

# ================================
# 2. Inputs del usuario
# ================================

age = st.number_input("Edad", min_value=1, max_value=120, value=40)
blood_pressure = st.number_input("Presión arterial", min_value=40, max_value=250, value=120)
cholesterol = st.number_input("Nivel de colesterol", min_value=50, max_value=400, value=200)
triglycerides = st.number_input("Triglicéridos", min_value=20, max_value=600, value=150)
glucose = st.number_input("Glucosa en ayunas", min_value=50, max_value=400, value=100)
bmi = st.number_input("Índice BMI", min_value=10.0, max_value=60.0, value=25.0)
sleep = st.number_input("Horas de sueño", min_value=1, max_value=16, value=7)

# Campos categóricos
smoking = st.selectbox("Fuma", ["No", "Sí"])
alcohol = st.selectbox("Consume alcohol", ["No", "Sí"])
diabetes = st.selectbox("Tiene diabetes", ["No", "Sí"])
family_history = st.selectbox("Antecedentes familiares", ["No", "Sí"])

gender = st.selectbox("Género", ["Female", "Male"])
exercise = st.selectbox("Ejercicio", ["Low", "High"])
stress = st.selectbox("Estrés", ["Low", "High"])

# ================================
# 3. Convertir a DataFrame
# ================================
entrada = pd.DataFrame([{
    "Age": age,
    "Blood Pressure": blood_pressure,
    "Cholesterol Level": cholesterol,
    "Triglyceride Level": triglycerides,
    "Fasting Blood Sugar": glucose,
    "BMI": bmi,
    "Sleep Hours": sleep,
    "Smoking": smoking,
    "Alcohol Consumption": alcohol,
    "Diabetes": diabetes,
    "Family Heart Disease": family_history,
    "Gender_Female": 1 if gender == "Female" else 0,
    "Gender_Male": 1 if gender == "Male" else 0,
    "Exercise_Low": 1 if exercise == "Low" else 0,
    "Exercise_High": 1 if exercise == "High" else 0,
    "Stress_Low": 1 if stress == "Low" else 0,
    "Stress_High": 1 if stress == "High" else 0
}])

# ================================
# 4. Botón de predicción
# ================================
if st.button("Predecir"):
    pred = modelo.predict(entrada)[0]
    prob = modelo.predict_proba(entrada)[0][1]

    if pred == 1:
        st.error(f"🔴 **Alto riesgo de enfermedad cardíaca** (probabilidad: {prob:.2f})")
    else:
        st.success(f"🟢 **Bajo riesgo de enfermedad cardíaca** (probabilidad: {prob:.2f})")

