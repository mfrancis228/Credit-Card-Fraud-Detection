import pandas as pd
import yaml
import os

def load_config(config_file="local"):
    """
    Charge la configuration depuis le dossier Config.
    config_file: "local" ou "prod"
    """
    config_path = os.path.join("Config", f"{config_file}.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Le fichier de configuration {config_path} est introuvable.")
        
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_raw_data(file_path=None):
    """
    Charge le dataset principal. 
    Si file_path n'est pas fourni, on peut le chercher dans la config.
    """
    if file_path is None:
        # Par défaut, on charge la config locale pour trouver le chemin
        config = load_config("local")
        file_path = config.get('data_path', 'Data/creditcard.csv')
    
    print(f"Chargement des données depuis : {file_path}")
    
    try:
        data = pd.read_csv(file_path)
        print(f"Données chargées avec succès : {data.shape[0]} lignes, {data.shape[1]} colonnes.")
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None

if __name__ == "__main__":
    # Test rapide du module
    df = load_raw_data()
    if df is not None:
        print(df.head())