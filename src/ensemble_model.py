
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import joblib
import os

def get_stacking_ensemble():
    """
    Crée un modèle Stacking basé sur les meilleurs résultats obtenus.
    """
    # 1. Définition des modèles de base (Base Learners)
    base_models = [
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ]
    
    # 2. Définition du méta-modèle (Meta-Learner)
    # Il va apprendre à combiner les probabilités des 3 modèles ci-dessus
    meta_model = LogisticRegression()
    
    ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,          # Utilise la validation croisée pour l'entraînement du méta-modèle
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    return ensemble

def save_ensemble(model, name="Ensemble_Stacking"):
    path = os.path.join("Outputs", "Models", f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Modèle ensemble sauvegardé : {path}")