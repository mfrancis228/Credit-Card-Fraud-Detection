import pandas as pd
import joblib
import os
import json
from load_data import load_config

def load_production_model(model_name="Ensemble_Stacking"):
    """Charge le modèle entraîné et les métadonnées associées."""
    model_path = os.path.join("Outputs", "Models", f"{model_name}.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_path} est introuvable. "
                                "Assurez-vous d'avoir entraîné l'ensemble.")
    
    model = joblib.load(model_path)
    print(f"✅ Modèle {model_name} chargé avec succès.")
    return model

def get_optimized_threshold(exp_name="Stacking_Standard"):
    """Récupère le seuil optimisé (par défaut 0.03 si non trouvé)."""
    # Note : Dans un flux réel, ce seuil proviendrait d'un fichier de config 
    # généré par threshold_optimization.py
    threshold_path = os.path.join("Outputs", "threshold_analysis", f"best_threshold_{exp_name}.json")
    
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            config = json.load(f)
            return config.get('threshold', 0.03)
    
    
    print("⚠️ Seuil optimisé non trouvé, utilisation du seuil de sécurité par défaut : 0.03")
    print(threshold_path)
    return 0.03

def make_predictions(new_data, threshold=0.03):
    """
    Prend de nouvelles données et retourne les prédictions finales.
    """
    model = load_production_model()
    
    # --- CORRECTION ICI ---
    # On s'assure de ne garder que les colonnes sur lesquelles le modèle a été entraîné.
    # Si votre modèle attend 29 features, c'est probablement V1 à V28 + Amount.
    # On retire 'Time' et 'Class' s'ils sont présents.
    features_to_drop = ['Class', 'Time']
    existing_drops = [c for c in features_to_drop if c in new_data.columns]
    X_input = new_data.drop(columns=existing_drops)
    
    # Vérification de sécurité
    if X_input.shape[1] != 29:
        print(f"⚠️ Attention: Le modèle attend 29 colonnes, X_input en a {X_input.shape[1]}")
    
    # Utilisation de .values pour éviter le UserWarning sur les feature names (Optionnel)
    probabilities = model.predict_proba(X_input.values)[:, 1]
    
    predictions = (probabilities >= threshold).astype(int)
    
    results = pd.DataFrame({
        'Probability': probabilities,
        'Is_Fraud': predictions,
        'Decision': ["BLOCK / VERIFY" if p == 1 else "APPROVE" for p in predictions]
    })
    
    return results

if __name__ == "__main__":
    # Simulation avec quelques lignes du dataset de test
    from sklearn.model_selection import train_test_split
    from load_data import load_raw_data
    
    # 1. Chargement des données pour le test
    df = load_raw_data()
    X = df.drop('Class', axis=1)
    _, X_test_sample = train_test_split(X, test_size=0.0001, random_state=42) # 28 lignes
    
    # 2. Prédiction
    print("\n--- Analyse de nouvelles transactions ---")
    threshold = get_optimized_threshold()
    final_results = make_predictions(X_test_sample, threshold=threshold)
    
    # 3. Affichage des alertes
    print(f"\nSeuil appliqué : {threshold}")
    print(final_results[final_results['Is_Fraud'] == 1])
    
    if not final_results['Is_Fraud'].any():
        print("Aucune fraude détectée dans cet échantillon.")