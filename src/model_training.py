from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

def get_models():
    """Retourne un dictionnaire des modèles à tester."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "KNN": KNeighborsClassifier(n_neighbors=5) # Note: KNN peut être lent sur de gros datasets
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