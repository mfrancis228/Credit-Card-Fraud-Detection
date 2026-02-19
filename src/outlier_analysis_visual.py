import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from load_data import load_raw_data

def detect_and_plot_outliers():
    # 1. Chargement des données
    df = load_raw_data()
    if df is None: return
    
    # Sélection des colonnes : V1 à V28 + Amount
    features = [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # 2. Configuration de la grille (6 lignes x 5 colonnes pour 29 features)
    n_features = len(features)
    n_cols = 5
    n_rows = (n_features // n_cols) + (1 if n_features % n_cols > 0 else 0)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 25))
    fig.suptitle("Analyse des Outliers par Box Plot (Méthode IQR)", fontsize=20, y=1.02)
    
    axes = axes.flatten() # Aplatir pour itérer facilement

    output_dir = "Outputs/outlier_analysis"
    os.makedirs(output_dir, exist_ok=True)

    print("Génération des Box Plots...")

    for i, col in enumerate(features):
        # Calcul des statistiques IQR pour info
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Plot sur l'axe correspondant
        sns.boxplot(data=df, y=col, ax=axes[i], palette="Set2", fliersize=2)
        
        # Ajout d'un titre et nettoyage des labels
        axes[i].set_title(f"{col}", fontsize=12, fontweight='bold')
        axes[i].set_ylabel("")
        
        # Optionnel : Colorer le fond si la colonne a beaucoup d'outliers
        n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if n_outliers > (len(df) * 0.05): # Si plus de 5% d'outliers
            axes[i].set_facecolor('#fff0f0') 

    # Cacher les subplots vides à la fin de la grille
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    # Sauvegarde
    save_path = os.path.join(output_dir, "all_features_boxplots.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"✅ Analyse terminée. Box plots sauvegardés dans : {save_path}")

if __name__ == "__main__":
    detect_and_plot_outliers()