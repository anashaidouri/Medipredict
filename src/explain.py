import numpy as np
import pandas as pd
import shap

def get_shap_values(model, scaler, X_user: pd.DataFrame):
    X_scaled = scaler.transform(X_user)

    # Pour la régression logistique ou arbre compatible
    explainer = shap.Explainer(model, X_scaled)
    shap_exp = explainer(X_scaled)

    shap_values = shap_exp.values[0]
    feature_names = list(X_user.columns)
    feature_values = X_user.iloc[0].values

    return shap_values, feature_names, feature_values

def generate_natural_explanation(shap_df: pd.DataFrame, risk_category: str):
    top_features = shap_df.head(3)

    positive_drivers = top_features[top_features["Impact_SHAP"] > 0]["Variable"].tolist()
    negative_drivers = top_features[top_features["Impact_SHAP"] < 0]["Variable"].tolist()

    parts = [f"Votre profil a été classé dans la catégorie de risque **{risk_category.lower()}**."]

    if positive_drivers:
        parts.append(
            "Les éléments qui ont le plus augmenté ce niveau de risque sont : "
            + ", ".join(positive_drivers) + "."
        )

    if negative_drivers:
        parts.append(
            "À l'inverse, certains éléments ont réduit le niveau de risque estimé, notamment : "
            + ", ".join(negative_drivers) + "."
        )

    parts.append(
        "Cette estimation reste indicative et ne remplace pas une évaluation par un professionnel de santé."
    )

    return " ".join(parts)