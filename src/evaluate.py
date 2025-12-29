import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve, 
    auc, roc_curve, roc_auc_score, accuracy_score, balanced_accuracy_score
)

def evaluate_model(model, X_test, y_test, exp_name):
    """
    Calcule les métriques avancées, génère les courbes de performance 
    et sauvegarde les scores de probabilité.
    """
    # 1. Prédictions
    y_pred = model.predict(X_test)
    
    # Récupération des probabilités (indispensable pour les courbes et l'AUC)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred # Fallback si le modèle ne supporte pas les probabilités

    # 2. Sauvegarde des scores en CSV
    scores_path = os.path.join("Outputs", f"scores_{exp_name}.csv")
    pd.DataFrame({
        'true_label': y_test,
        'predicted_score': y_score
    }).to_csv(scores_path, index=False)

    # 3. Calcul des métriques
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Métriques des courbes
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall, precision)
    
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)

    # 4. Visualisations (Multi-plot)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Évaluation Expérience : {exp_name}", fontsize=16)

    # A. Matrice de Confusion
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax1)
    ax1.set_title("Confusion Matrix")
    ax1.set_ylabel('Réel')
    ax1.set_xlabel('Prédit')

    # B. Precision-Recall Curve (La plus importante pour la fraude)
    ax2.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}', color='blue')
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")
    ax2.grid(True)

    # C. ROC Curve
    ax3.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}', color='darkorange')
    ax3.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax3.set_title("ROC Curve")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.legend(loc="lower right")
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    img_path = os.path.join("Outputs", "Images", f"eval_{exp_name}.png")
    plt.savefig(img_path)
    plt.close()

    # 5. Retour des résultats pour le tableau final
    return {
        "Experience": exp_name,
        "Accuracy": acc,
        "Balanced_Acc": balanced_acc,
        "Precision_Fraud": report['1']['precision'],
        "Recall_Fraud": report['1']['recall'],
        "F1_Fraud": report['1']['f1-score'],
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc
    }