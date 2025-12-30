from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import os

def get_models():
    """Retourne un dictionnaire des modèles à tester, incluant des méthodes d'ensemble."""
    
    # Note pour XGBoost : scale_pos_weight aide à gérer le déséquilibre 
    # (Ratio de classes négatives / classes positives)
    
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        
        # --- Modèles ajoutés ---
        
        # Random Forest : Très robuste au surapprentissage (Overfitting)
        "RandomForest": RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced', 
            n_jobs=-1
        ),
        
        # XGBoost : L'état de l'art pour les données tabulaires
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        
        # LightGBM : Souvent plus rapide que XGBoost sur les gros datasets
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            class_weight='balanced',
            importance_type='gain'
        ),
        
        # AdaBoost : Une autre méthode de boosting classique
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            random_state=42
        )
    }

def train_model(model, X_train, y_train):
    """Entraîne le modèle fourni."""
    model.fit(X_train, y_train)
    return model

def save_trained_model(model, name):
    """Sauvegarde le modèle dans le dossier dédié."""
    path = os.path.join("Outputs", "Models", f"{name}.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)