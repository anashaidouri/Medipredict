
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_risk_gauge(probability, category):
    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.barh([0], [1], color="#dddddd")
    ax.barh([0], [probability], color="#4c78a8")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.3, 0.6, 1.0])
    ax.set_xticklabels(["0", "Faible", "Modéré", "Élevé"])
    ax.set_title(f"Niveau de risque : {category}")
    return fig

def plot_feature_distributions(df, user_input):
    cols = ["Glucose", "BMI", "Age", "BloodPressure"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for i, col in enumerate(cols):
        axes[i].hist(df[col], bins=20)
        axes[i].axvline(user_input[col], linestyle="--")
        axes[i].set_title(col)

    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Matrice de corrélation", pad=20)
    return fig

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    fig.colorbar(im)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title("Matrice de confusion")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    return fig

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend()
    return fig