import shap
import pandas as pd
import numpy as np

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

def get_shap_values(model, scaler, X_user):

    X_scaled = scaler.transform(X_user)

    explainer = shap.LinearExplainer(model, X_scaled)

    shap_values = explainer.shap_values(X_scaled)

    feature_names = X_user.columns
    feature_values = X_user.iloc[0].values

    shap_df = pd.DataFrame({
        "Variable": feature_names,
        "Valeur": feature_values,
        "Impact_SHAP": shap_values[0]
    })

    shap_df = shap_df.sort_values("Impact_SHAP", key=np.abs, ascending=False)

    return shap_df, explainer, shap_values


def generate_model_decision_explanation(shap_df, probability, risk_category):

    text = ""

    text += (
        f"Le modèle estime une probabilité de diabète de **{probability:.2f}**, "
        f"ce qui correspond à un niveau de risque **{risk_category.lower()}**.\n\n"
    )

    text += (
        "Cette estimation est obtenue en combinant l'influence de plusieurs "
        "variables médicales présentes dans le modèle.\n\n"
    )

    top_features = shap_df.head(4)

    text += "Les variables qui ont le plus influencé la décision du modèle sont :\n\n"

    for _, row in top_features.iterrows():

        var = row["Variable"]
        val = row["Valeur"]
        impact = row["Impact_SHAP"]

        if impact > 0:
            direction = "augmente"
        else:
            direction = "réduit"

        text += f"- **{var} ({val})** {direction} l'estimation du risque.\n"

    text += (
        "\nLe modèle combine ensuite l'effet de toutes les variables pour produire "
        "une estimation globale du risque."
    )

    return text