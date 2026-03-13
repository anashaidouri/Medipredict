import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from src.predict import (
    prepare_input_data,
    predict_risk,
    risk_label,
    validate_inputs,
    default_user_input,
)
from src.explain import get_shap_values, generate_natural_explanation
from src.visualize import (
    plot_risk_gauge,
    plot_feature_distributions,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_roc_curve,
)

# =========================
# Configuration générale
# =========================
st.set_page_config(
    page_title="MediPredict",
    page_icon="🩺",
    layout="wide"
)

# Style accessibilité minimal
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-size: 16px;
        }
        .main-title {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .section-title {
            font-size: 24px;
            font-weight: 600;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .legal-box {
            padding: 1rem;
            border-radius: 10px;
            border: 2px solid #999;
            background-color: #f7f7f7;
            color: #111;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# Chargement ressources
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("model/medipredict_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv("data/diabetes.csv")
model1 = joblib.load("model/medipredict_model.pkl")
model, scaler = load_model()
df = load_data()

# IMPORTANT :
# Remplace ces valeurs par TES résultats réels du notebook
MODEL_METRICS = {
    "accuracy": 0.72,
    "precision": 0.62,
    "recall": 0.54,
    "f1": 0.57,
    "auc": 0.79
}


y_true_demo = None
y_pred_demo = None
y_proba_demo = None

# =========================
# Session state
# =========================
if "consent_given" not in st.session_state:
    st.session_state.consent_given = False

if "user_input" not in st.session_state:
    st.session_state.user_input = default_user_input()

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "risk_probability" not in st.session_state:
    st.session_state.risk_probability = None

if "risk_category" not in st.session_state:
    st.session_state.risk_category = None

if "processed_input" not in st.session_state:
    st.session_state.processed_input = None

# =========================
# Sidebar navigation
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller à :",
    [
        "Accueil",
        "Mon profil de risque",
        "Comprendre ma prédiction",
        "Explorer les données",
    ]
)

# =========================
# Page 1 — Accueil
# =========================
if page == "Accueil":
    st.markdown('<div class="main-title">🩺 MediPredict</div>', unsafe_allow_html=True)
    st.write(
        """
        MediPredict est une application de sensibilisation qui permet d'estimer
        un niveau de risque de diabète de type 2 à partir d'indicateurs de santé anonymes.
        """
    )

    st.markdown(
        """
        <div class="legal-box">
        <strong>Mention légale obligatoire :</strong><br>
        Cet outil est un outil de sensibilisation. Il ne constitue pas un avis médical.
        En cas de doute, consultez un professionnel de santé.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Politique de confidentialité simplifiée</div>', unsafe_allow_html=True)
    st.write(
        """
        Les données saisies ne sont ni stockées, ni enregistrées, ni partagées.
        Le traitement est réalisé uniquement en mémoire pendant votre session.
        Aucun historique nominatif n'est conservé.
        Cet outil ne remplace pas un professionnel de santé.
        """
    )

    consent = st.checkbox("J’ai lu les informations ci-dessus et je consens à utiliser cet outil.")
    if consent:
        st.session_state.consent_given = True
        st.success("Consentement enregistré. Vous pouvez maintenant utiliser l'application.")
    else:
        st.session_state.consent_given = False
        st.info("Veuillez cocher la case pour accéder aux fonctionnalités.")

# =========================
# Page 2 — Mon profil de risque
# =========================
elif page == "Mon profil de risque":
    st.markdown('<div class="main-title">Mon profil de risque</div>', unsafe_allow_html=True)

    if not st.session_state.consent_given:
        st.warning("Veuillez d'abord donner votre consentement sur la page Accueil.")
        st.stop()

    with st.form("risk_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies_na = st.checkbox("Grossesses : non applicable", value=False)
            pregnancies = st.number_input(
                "Nombre de grossesses",
                min_value=0,
                max_value=20,
                value=0,
                help="Nombre total de grossesses. Si non applicable, cochez la case ci-dessus."
            )

            glucose = st.number_input(
                "Taux de glucose",
                min_value=0,
                max_value=500,
                value=120,
                help="Valeur biologique plausible : environ 40 à 250."
            )

            blood_pressure = st.number_input(
                "Pression artérielle diastolique",
                min_value=0,
                max_value=150,
                value=70,
                help="Valeur en mmHg."
            )

            skin_thickness = st.number_input(
                "Épaisseur du pli cutané",
                min_value=0,
                max_value=100,
                value=20,
                help="Mesure en mm."
            )

        with col2:
            insulin = st.number_input(
                "Insuline",
                min_value=0,
                max_value=900,
                value=80,
                help="Valeur mesurée dans le dataset d'origine."
            )

            bmi = st.number_input(
                "Indice de masse corporelle",
                min_value=0.0,
                max_value=70.0,
                value=28.0,
                step=0.1,
                help="IMC = poids / taille²"
            )

            dpf = st.number_input(
                "Antécédents familiaux (score pedigree diabète)",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.01,
                help="Score indicatif lié aux antécédents familiaux."
            )

            age = st.number_input(
                "Âge",
                min_value=18,
                max_value=100,
                value=35,
                help="Âge en années."
            )

        submitted = st.form_submit_button("Analyser mon profil")

    if submitted:
        if pregnancies_na:
            pregnancies = 0

        user_input = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }

        errors = validate_inputs(user_input)

        if errors:
            for err in errors:
                st.error(err)
        else:
            X_user = prepare_input_data(user_input)
            probability = predict_risk(model, scaler, X_user)
            category = risk_label(probability)

            st.session_state.user_input = user_input
            st.session_state.prediction_done = True
            st.session_state.risk_probability = probability
            st.session_state.risk_category = category
            st.session_state.processed_input = X_user

            st.success("Analyse terminée.")

    if st.session_state.prediction_done:
        st.subheader("Résultat")
        fig = plot_risk_gauge(st.session_state.risk_probability, st.session_state.risk_category)
        st.pyplot(fig)

        st.write(f"**Niveau de risque estimé : {st.session_state.risk_category}**")
        st.caption("Le résultat est présenté comme un niveau de risque et non comme un diagnostic médical.")



# =========================
# Page 3 — Comprendre ma prédiction
# =========================

elif page == "Comprendre ma prédiction":
    # from src.explain import get_shap_values,
    from src.explain import get_shap_values, generate_model_decision_explanation

    st.markdown('<div class="main-title">Comprendre ma prédiction</div>', unsafe_allow_html=True)

    if not st.session_state.prediction_done:
        st.info("Veuillez d'abord effectuer une analyse sur la page 'Mon profil de risque'.")
        st.stop()

    st.subheader("Interprétation du résultat")

    probability = st.session_state.risk_probability
    risk = st.session_state.risk_category

    st.info(
        f"Votre estimation de risque est **{risk.lower()}** "
        f"(probabilité estimée : {probability:.2f}).\n\n"
        "Cette estimation est produite par un modèle de machine learning "
        "entraîné sur le dataset Pima Indians Diabetes."
    )

    # -----------------------------
    # Calcul SHAP
    # -----------------------------

    X_user = st.session_state.processed_input

    shap_df, explainer, shap_values = get_shap_values(
        model,
        scaler,
        X_user
    )

    # -----------------------------
    # Variables importantes
    # -----------------------------

    st.subheader("Variables ayant le plus influencé la prédiction")

    st.dataframe(shap_df, use_container_width=True)

    # -----------------------------
    # Explication décision modèle
    # -----------------------------

    st.subheader("Comment le modèle a pris sa décision")

    explanation = generate_model_decision_explanation(
        shap_df,
        probability,
        risk
    )

    st.write(explanation)

    # -----------------------------
    # Graphique SHAP
    # -----------------------------

    st.subheader("Contribution des variables à la prédiction")

    import shap

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_user.iloc[0],
            feature_names=X_user.columns
        )
    )

    st.pyplot(plt.gcf())

    st.caption(
        "Ce graphique montre comment chaque variable influence la prédiction : "
        "les variables en rouge augmentent le risque estimé, celles en bleu le réduisent."
    )

    # -----------------------------
    # Comparaison dataset
    # -----------------------------

    st.subheader("Comparaison avec le dataset")

    fig = plot_feature_distributions(df, st.session_state.user_input)

    st.pyplot(fig)

    # -----------------------------
    # Explication naturelle
    # -----------------------------

    st.subheader("Explication en langage simple")

    top_features = shap_df.head(3)

    increase_risk = top_features[top_features["Impact_SHAP"] > 0]
    decrease_risk = top_features[top_features["Impact_SHAP"] < 0]

    explanation = f"Votre estimation de risque est **{risk.lower()}**.\n\n"

    if not increase_risk.empty:
        explanation += "Les facteurs qui augmentent cette estimation sont :\n"
        for _, row in increase_risk.iterrows():
            explanation += f"- **{row['Variable']}** (valeur : {row['Valeur']})\n"

    if not decrease_risk.empty:
        explanation += "\nLes facteurs qui réduisent cette estimation sont :\n"
        for _, row in decrease_risk.iterrows():
            explanation += f"- **{row['Variable']}** (valeur : {row['Valeur']})\n"

    explanation += """

Ces variables influencent la prédiction car elles sont associées,
dans les données utilisées pour entraîner le modèle, à un risque
plus ou moins élevé de diabète.

Il est important de rappeler que cette estimation est statistique
et ne constitue pas un diagnostic médical.
"""

    st.write(explanation)
    # -----------------------------
    # Recommandations
    # -----------------------------

    st.subheader("Recommandations générales")

    st.write(
        """
        - Un taux de glucose élevé peut être un facteur de risque important.
        - Maintenir un poids équilibré peut réduire le risque de diabète.
        - Une activité physique régulière est recommandée.
        - Une alimentation équilibrée peut contribuer à améliorer la santé métabolique.

        ⚠️ Cet outil est uniquement un outil de sensibilisation.
        En cas d'inquiétude, consultez un professionnel de santé.
        """
    )
# =========================
# Page 4 — Explorer les données
# =========================
elif page == "Explorer les données":
    st.markdown('<div class="main-title">Explorer les données</div>', unsafe_allow_html=True)

    if not st.session_state.consent_given:
        st.warning("Veuillez d'abord donner votre consentement sur la page Accueil.")
        st.stop()
    
    st.subheader("Aperçu du dataset")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Corrélations")
    fig_corr = plot_correlation_heatmap(df)
    st.pyplot(fig_corr)

    st.subheader("Performance du modèle")
    st.write(f"Accuracy : {MODEL_METRICS['accuracy']:.2f}")
    st.write(f"Précision : {MODEL_METRICS['precision']:.2f}")
    st.write(f"Rappel : {MODEL_METRICS['recall']:.2f}")
    st.write(f"F1-score : {MODEL_METRICS['f1']:.2f}")
    st.write(f"AUC-ROC : {MODEL_METRICS['auc']:.2f}")

    if y_true_demo is not None and y_pred_demo is not None:
        fig_cm = plot_confusion_matrix(y_true_demo, y_pred_demo)
        st.pyplot(fig_cm)

    if y_true_demo is not None and y_proba_demo is not None:
        fig_roc = plot_roc_curve(y_true_demo, y_proba_demo)
        st.pyplot(fig_roc)

    st.subheader("Transparence")
    st.write(
        """
        Le modèle utilisé est un modèle supervisé entraîné sur le dataset Pima Indians Diabetes.
        Ce jeu de données ne représente pas parfaitement la population française, ce qui limite
        la généralisation. Certaines variables peuvent aussi introduire des biais.
        Le résultat doit être interprété comme une aide à la sensibilisation.
        """
    )