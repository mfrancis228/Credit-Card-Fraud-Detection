from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import uvicorn

# 1. Définition du format de la requête (Schéma de données)
class Transaction(BaseModel):
    # Les 28 variables PCA + Amount
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

# 2. Initialisation de l'application
app = FastAPI(
    title="Sentinelle Fraud Detection API",
    description="API de détection de fraude en temps réel basée sur un modèle de Stacking.",
    version="1.0.0"
)

# 3. Chargement des ressources
MODEL_PATH = "Outputs/Models/Ensemble_Stacking.pkl"
THRESHOLD = 0.03  # Seuil optimisé précédemment

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur de chargement du modèle : {e}")
    model = None

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API Sentinelle. Utilisez /docs pour la documentation."}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non disponible.")
    
    # Conversion de la requête Pydantic en DataFrame (format attendu par sklearn)
    data = pd.DataFrame([transaction.model_dump()])
    
    # Prédiction des probabilités
    # Note : .values évite le warning sur les feature names si nécessaire
    prob = model.predict_proba(data.values)[:, 1][0]
    
    # Décision métier
    decision = "BLOCK" if prob >= THRESHOLD else "APPROVE"
    
    return {
        "fraud_probability": round(float(prob), 4),
        "threshold_used": THRESHOLD,
        "decision": decision,
        "action_required": decision == "BLOCK"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)