import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def run_full_error_analysis(threshold_low=0.25, threshold_high=0.75):
    """
    Parcourt tous les fichiers scores_*.csv et analyse les échecs critiques.
    """
    # 1. Configuration des dossiers
    input_dir = "Outputs"
    output_dir = os.path.join(input_dir, "error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Récupération de tous les fichiers de scores
    score_files = glob.glob(os.path.join(input_dir, "scores_*.csv"))
    
    if not score_files:
        print("Aucun fichier de scores trouvé dans Outputs/. Lancez d'abord main.py.")
        return

    summary_errors = []

    # 3. Boucle sur chaque expérience
    for file_path in score_files:
        # Extraire le nom de l'expérience du nom de fichier
        filename = os.path.basename(file_path)
        exp_name = filename.replace("scores_", "").replace(".csv", "")
        
        print(f"Analyse des erreurs pour : {exp_name}...")
        df = pd.read_csv(file_path)

        # Identification des échecs drastiques
        # FN Drastique : C'est une fraude (1), mais le modèle est sûr à 90% que c'est sain (<0.1)
        drastic_fn = df[(df['true_label'] == 1) & (df['predicted_score'] < threshold_low)]
        
        # FP Drastique : C'est sain (0), mais le modèle est sûr à 90% que c'est une fraude (>0.9)
        drastic_fp = df[(df['true_label'] == 0) & (df['predicted_score'] > threshold_high)]

        # Enregistrement des statistiques
        summary_errors.append({
            "Experience": exp_name,
            "Drastic_FN": len(drastic_fn),
            "Drastic_FP": len(drastic_fp),
            "Total_Errors": len(drastic_fn) + len(drastic_fp)
        })

        # 4. Génération d'un graphique de distribution des scores pour cette expérience
        if len(df) > 0:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='predicted_score', hue='true_label', bins=50, element="step", common_norm=False)
            plt.axvline(threshold_low, color='red', linestyle='--', label='Seuil FN Drastique')
            plt.axvline(threshold_high, color='green', linestyle='--', label='Seuil FP Drastique')
            plt.title(f"Distribution des Scores : {exp_name}")
            plt.xlabel("Probabilité de Fraude prédite")
            plt.ylabel("Nombre de transactions")
            plt.legend()
            plt.yscale('log') # Log scale car les classes sont très déséquilibrées
            
            plt.savefig(os.path.join(output_dir, f"dist_{exp_name}.png"))
            plt.close()

    # 5. Création d'un graphique comparatif global
    df_summary = pd.DataFrame(summary_errors).sort_values(by="Drastic_FN")
    
    plt.figure(figsize=(12, 8))
    df_melted = df_summary.melt(id_vars="Experience", value_vars=["Drastic_FN", "Drastic_FP"])
    sns.barplot(data=df_melted, y="Experience", x="value", hue="variable", palette={"Drastic_FN": "red", "Drastic_FP": "orange"})
    
    plt.title("Comparaison des Échecs Drastiques par Modèle\n(Plus c'est bas, mieux c'est)")
    plt.xlabel("Nombre d'erreurs à haute confiance")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "global_comparison_errors.png"))
    plt.close()

    # Sauvegarde du rapport en CSV
    df_summary.to_csv(os.path.join(output_dir, "summary_drastic_errors.csv"), index=False)
    print(f"\nAnalyse terminée. Résultats sauvegardés dans : {output_dir}")

if __name__ == "__main__":
    run_full_error_analysis()