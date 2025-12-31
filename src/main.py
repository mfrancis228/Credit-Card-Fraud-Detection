import os
import sys

# Ajouter le dossier parent au path pour s'assurer que les imports fonctionnent 
# si vous lancez le script depuis la racine du projet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_data import load_raw_data, load_config
from train_test_split import prepare_base_split, get_experiment_data
from model_training import get_models, train_model, save_trained_model
from evaluate import evaluate_model
import pandas as pd
from ensemble_model import get_stacking_ensemble, save_ensemble

def main():
    # 1. Initialisation
    config = load_config()
    data = load_raw_data(config['data_path'])
    if data is None: return

    # 2. Split de base
    X_train_base, X_test, y_train_base, y_test = prepare_base_split(data) 
    
    sampling_methods = ['standard', 'undersampling', 'smote']
    models_to_test = get_models()
    all_results = []

    # 3. Boucle des expériences
    for method in sampling_methods:
        print(f"\n--- Préparation des données : {method.upper()} ---")
        X_train, y_train = get_experiment_data(X_train_base, y_train_base, method=method)
        
        for model_name, model_obj in models_to_test.items():
            exp_id = f"{model_name}_{method}"
            print(f"Entraînement de {exp_id}...")
            
            # Entraînement
            trained_model = train_model(model_obj, X_train, y_train)
            
            # Évaluation
            results = evaluate_model(trained_model, X_test, y_test, exp_id)
            all_results.append(results)
            
            # Sauvegarde
            save_trained_model(trained_model, exp_id)

    # 4. Résumé final
    df_results = pd.DataFrame(all_results)
    print("\n" + "="*50)
    print("RÉSULTATS COMPARATIFS DES EXPÉRIENCES")
    print("="*50)
    print(df_results.sort_values(by="F1_Fraud", ascending=False).to_string(index=False))
    output_df_path = r"D:\Project\Data science\Learning\Credit Card Fraud Detection\Outputs\Training result.csv"
    df_results.to_csv(output_df_path)
    
    print("\n--- Entraînement de l'Ensemble Learning (Stacking) ---")
    ensemble_model = get_stacking_ensemble()

    # Option A : Entraînement sur données Standard
    ensemble_model.fit(X_train_base, y_train_base)
    res_stack = evaluate_model(ensemble_model, X_test, y_test, "Stacking_Standard")

    # Option B : On peut aussi l'entraîner sur SMOTE pour voir si le Recall décolle
    # ensemble_model.fit(X_train_smote, y_train_smote)
    # res_stack_smote = evaluate_model(ensemble_model, X_test, y_test, "Stacking_SMOTE")

    save_ensemble(ensemble_model)

if __name__ == "__main__":
    main()