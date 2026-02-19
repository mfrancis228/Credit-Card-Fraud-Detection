import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from load_data import load_raw_data
import os

def detect_outliers():
    # 1. Chargement des données
    df = load_raw_data()
    if df is None: return
    
    # Sélection des colonnes cibles : V1 jusqu'à Amount
    features = [f'V{i}' for i in range(1, 29)] + ['Amount']
    data_subset = df[features]
    
    # --- MÉTHODE 1 : ÉCART INTERQUARTILE (IQR) ---
    print("Analyse IQR en cours...")
    outliers_iqr = pd.DataFrame(index=df.index)
    
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Marquer comme 1 si c'est un outlier
        outliers_iqr[col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)

    # --- MÉTHODE 2 : ISOLATION FOREST (CLUSTERING AUTOMATIQUE) ---
    print("Analyse Isolation Forest en cours...")
    # contamination=0.05 suppose qu'environ 5% des données sont des anomalies
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    # -1 pour anomalie, 1 pour normal
    iso_outliers = iso_forest.fit_predict(data_subset)
    df['iso_outlier'] = [1 if x == -1 else 0 for x in iso_outliers]

    # --- VISUALISATION DES RÉSULTATS ---
    output_dir = "Outputs/outlier_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Graphique 1 : Nombre d'outliers par colonne (IQR)
    plt.figure(figsize=(15, 6))
    outlier_counts = outliers_iqr.sum().sort_values(ascending=False)
    sns.barplot(x=outlier_counts.index, y=outlier_counts.values, palette='viridis')
    plt.title("Nombre d'Outliers détectés par colonne (Méthode IQR)")
    plt.xticks(rotation=45)
    plt.ylabel("Nombre de points")
    plt.savefig(os.path.join(output_dir, "iqr_outliers_count.png"))
    
    # Graphique 2 : Comparaison Isolation Forest vs Class (Fraude)
    plt.figure(figsize=(10, 6))
    confusion = pd.crosstab(df['Class'], df['iso_outlier'], normalize='index')
    sns.heatmap(confusion, annot=True, cmap='Blues', fmt='.2%')
    plt.title("Corrélation entre Anomalies (IsoForest) et Fraudes Réelles")
    plt.xlabel("Détecté comme Outlier (1=Oui)")
    plt.ylabel("Est une Fraude (1=Oui)")
    plt.savefig(os.path.join(output_dir, "isoforest_correlation.png"))
    
    plt.show()

    print(f"\n✅ Analyse terminée. Graphiques sauvegardés dans : {output_dir}")
    print(f"Total outliers détectés par Isolation Forest : {df['iso_outlier'].sum()}")

if __name__ == "__main__":
    detect_outliers()