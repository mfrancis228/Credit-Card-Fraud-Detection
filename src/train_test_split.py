import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def prepare_base_split(df, target_col='Class', test_size=0.2, random_state=42):
    """
    Sépare les données et applique une normalisation.
    Utilise 'stratify' pour maintenir le ratio de fraude dans les deux ensembles.
    """
    X = df.drop(columns=[target_col, 'Time'], errors='ignore')
    y = df[target_col]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalisation (indispensable pour la plupart des modèles)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test

def get_experiment_data(X_train, y_train, method='standard'):
    """
    Génère différentes versions des données d'entraînement selon l'expérience.
    """
    if method == 'standard':
        # Expérience 1 : Données originales (déséquilibrées)
        return X_train, y_train

    elif method == 'undersampling':
        # Expérience 2 : Réduction de la classe majoritaire (Non-Fraude)
        # On réduit les 0 pour qu'il y en ait autant que les 1
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        return X_res, y_res

    elif method == 'smote':
        # Expérience 3 : Augmentation synthétique de la classe minoritaire (Fraude)
        # On crée des fausses fraudes mathématiquement crédibles
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res

    else:
        raise ValueError("Méthode inconnue. Choisissez 'standard', 'undersampling' ou 'smote'.")

if __name__ == "__main__":
    # Test rapide si le fichier est lancé seul
    print("Module train_test_split prêt.")