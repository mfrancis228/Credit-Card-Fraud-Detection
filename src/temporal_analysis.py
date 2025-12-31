import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from load_data import load_raw_data
import os

def analyze_fraud_frequency():
    # 1. Chargement des données
    df = load_raw_data()
    if df is None: return
    
    # Extraire uniquement les fraudes et leur temps
    frauds = df[df['Class'] == 1][['Time', 'Amount']].copy()
    frauds = frauds.sort_values('Time')
    
    # 2. Calcul des intervalles entre fraudes (Inter-arrival times)
    frauds['diff'] = frauds['Time'].diff().dropna()
    
    # --- VISUALISATION ---
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Distribution temporelle brute
    plt.subplot(2, 1, 1)
    plt.scatter(frauds['Time'], np.ones(len(frauds)), alpha=0.5, marker='|', color='red', s=500)
    plt.title("Occurrence des fraudes sur l'axe du Temps (Time)")
    plt.xlabel("Temps (secondes)")
    plt.yticks([]) # Cacher l'axe Y
    
    # Subplot 2: Distribution des intervalles (IAT)
    plt.subplot(2, 1, 2)
    sns.histplot(frauds['diff'].dropna(), kde=True, color='blue', bins=50)
    plt.title("Distribution du temps entre deux fraudes (Inter-Arrival Time)")
    plt.xlabel("Secondes entre deux fraudes")
    
    plt.tight_layout()

    # SAUVEGARDE SUR LE DISQUE
    output_dir = os.path.join("Outputs", "temporal_analysis")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "fraud_temporal_distribution.png")
    plt.savefig(save_path)
    print(f"✅ Graphique de distribution temporelle sauvegardé : {save_path}")
    plt.show()
    plt.close()

    # --- TESTS STATISTIQUES ---
    print("\n--- Statistiques Temporelles ---")
    print(f"Nombre total de fraudes : {len(frauds)}")
    print(f"Intervalle moyen entre fraudes : {frauds['diff'].mean():.2f} secondes")
    print(f"Écart-type des intervalles : {frauds['diff'].std():.2f}")
    
    # Coefficient de Variation (CV)
    # Si CV ~ 1 : Aléatoire (Poisson)
    # Si CV > 1 : Burstiness (Regroupements/Rafales)
    cv = frauds['diff'].std() / frauds['diff'].mean()
    print(f"Coefficient de Variation (CV) : {cv:.2f}")
    
    if cv > 1.2:
        print("💡 Conclusion : Les fraudes semblent survenir par 'rafales' (Clusters).")
    elif cv < 0.8:
        print("💡 Conclusion : Les fraudes semblent suivre un motif périodique régulier.")
    else:
        print("💡 Conclusion : Les fraudes semblent survenir de manière aléatoire (Processus de Poisson).")

if __name__ == "__main__":
    analyze_fraud_frequency()