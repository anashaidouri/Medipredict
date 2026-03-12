import pandas as pd
import numpy as np

FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

def default_user_input():
    return {
        "Pregnancies": 0,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 35
    }

def validate_inputs(user_input: dict):
    errors = []

    if not (0 <= user_input["Pregnancies"] <= 20):
        errors.append("Le nombre de grossesses doit être compris entre 0 et 20.")

    if not (40 <= user_input["Glucose"] <= 250):
        errors.append("Le taux de glucose doit être dans une plage biologiquement plausible (40-250).")

    if not (30 <= user_input["BloodPressure"] <= 150):
        errors.append("La pression artérielle doit être dans une plage plausible (30-150).")

    if not (0 <= user_input["SkinThickness"] <= 100):
        errors.append("L'épaisseur du pli cutané doit être comprise entre 0 et 100.")

    if not (0 <= user_input["Insulin"] <= 900):
        errors.append("L'insuline doit être comprise entre 0 et 900.")

    if not (10 <= user_input["BMI"] <= 70):
        errors.append("L'IMC doit être dans une plage plausible (10-70).")

    if not (0.0 <= user_input["DiabetesPedigreeFunction"] <= 3.0):
        errors.append("Le score d'antécédents familiaux doit être compris entre 0 et 3.")

    if not (18 <= user_input["Age"] <= 100):
        errors.append("L'âge doit être compris entre 18 et 100 ans.")

    return errors

def prepare_input_data(user_input: dict):
    df = pd.DataFrame([user_input])
    return df[FEATURE_ORDER]

def predict_risk(model, scaler, X_user: pd.DataFrame):
    X_scaled = scaler.transform(X_user)
    probability = model.predict_proba(X_scaled)[0][1]
    return probability

def risk_label(probability: float):
    if probability < 0.30:
        return "Faible"
    elif probability < 0.60:
        return "Modéré"
    return "Élevé"