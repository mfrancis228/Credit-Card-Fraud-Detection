import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve
from calculate_business_costs import calculate_operational_costs
import os
import json

def save_threshold_to_json(threshold, exp_name):
    """Sauvegarde le seuil optimal dans un fichier JSON pour le module predict."""
    output_dir = "Outputs/threshold_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, f"best_threshold_{exp_name}.json")
    
    data = {
        "experience": exp_name,
        "threshold": float(threshold) # conversion en float standard pour le JSON
    }
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"✅ Seuil sauvegardé avec succès dans : {file_path}")

def optimize_and_visualize_threshold(exp_name="Stacking_Standard"):
    """
    Calcule le seuil optimal via les coûts métier et visualise les courbes de décision.
    """
    # 1. Récupérer les coûts calculés dynamiquement via load_raw_data()
    cost_fn, cost_fp = calculate_operational_costs()
    
    # 2. Charger les scores du modèle
    file_path = f"Outputs/scores_{exp_name}.csv"
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} est introuvable.")
        return
        
    df = pd.read_csv(file_path)
    y_true = df['true_label']
    y_scores = df['predicted_score']

    # 3. Simulation sur une plage de seuils
    thresholds = np.linspace(0.001, 0.999, 100)
    results = []
    
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calcul du coût total et du score F1
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        f1 = f1_score(y_true, y_pred)
        
        results.append({
            'threshold': t,
            'total_cost': total_cost,
            'f1_score': f1,
            'fp_count': fp,
            'fn_count': fn
        })
    
    res_df = pd.DataFrame(results)
    
    # Identification des points optimaux
    best_f1_row = res_df.loc[res_df['f1_score'].idxmax()]
    best_cost_row = res_df.loc[res_df['total_cost'].idxmin()]

    # 4. Visualisation
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Axe 1 : F1-Score (Performance Statistique)
    color_f1 = 'tab:blue'
    ax1.set_xlabel('Seuil de probabilité (Threshold)')
    ax1.set_ylabel('F1-Score', color=color_f1)
    ax1.plot(res_df['threshold'], res_df['f1_score'], color=color_f1, linewidth=2, label='F1-Score')
    ax1.tick_params(axis='y', labelcolor=color_f1)
    ax1.grid(alpha=0.3)

    # Axe 2 : Coût Métier (Performance Financière)
    ax2 = ax1.twinx()
    color_cost = 'tab:red'
    ax2.set_ylabel('Coût métier total (€)', color=color_cost)
    ax2.plot(res_df['threshold'], res_df['total_cost'], color=color_cost, linewidth=2, linestyle='--', label='Coût métier')
    ax2.tick_params(axis='y', labelcolor=color_cost)

    # Annotations des seuils optimaux
    ax1.axvline(best_f1_row['threshold'], color='blue', linestyle=':', alpha=0.7)
    ax1.text(best_f1_row['threshold'], 0.1, f" Max F1: {best_f1_row['threshold']:.3f}", color='blue', rotation=90)
    
    ax2.axvline(best_cost_row['threshold'], color='red', linestyle=':', alpha=0.7)
    ax2.text(best_cost_row['threshold'], res_df['total_cost'].max()*0.8, f" Min Cost: {best_cost_row['threshold']:.3f}", color='red', rotation=90)

    plt.title(f"Optimisation du Seuil : {exp_name}\n(FN Cost: {cost_fn:.2f}€ | FP Cost: {cost_fp:.2f}€)")
    
    # Sauvegarde
    os.makedirs("Outputs/threshold_analysis", exist_ok=True)
    save_path = f"Outputs/threshold_analysis/optimized_threshold_{exp_name}.png"
    plt.savefig(save_path)
    plt.close()

    print(f"--- Rapport d'Optimisation ---")
    print(f"Seuil F1 optimal : {best_f1_row['threshold']:.4f} (Score: {best_f1_row['f1_score']:.3f})")
    print(f"Seuil Coût optimal : {best_cost_row['threshold']:.4f} (Coût: {best_cost_row['total_cost']:.2f}€)")
    print(f"Économie réalisée vs seuil 0.5 : {res_df.iloc[50]['total_cost'] - best_cost_row['total_cost']:.2f} €")
    
    best_threshold = best_cost_row['threshold']
    
    save_threshold_to_json(best_threshold, exp_name)
    
    return best_threshold

if __name__ == "__main__":
    optimize_and_visualize_threshold("Stacking_Standard")