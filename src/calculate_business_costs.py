from load_data import load_raw_data
import os

def calculate_operational_costs():
    """
    Calcule les coûts FN et FP basés sur les données réelles du projet.
    """
    # Utilisation de votre fonction existante pour charger les données
    df = load_raw_data()
    
    if df is None:
        print("Impossible de calculer les coûts : données non trouvées.")
        return 100, 10 # Valeurs par défaut en cas d'erreur
    
    # --- CALCUL DE COST_FN (Faux Négatif) ---
    # On cible les fraudes réelles pour voir ce qu'elles coûtent en moyenne
    fraud_amounts = df[df['Class'] == 1]['Amount']
    avg_fraud_loss = fraud_amounts.mean()
    
    # Ajout d'un forfait de gestion de litige (frais bancaires, dossiers)
    administrative_fee = 50.0 
    cost_fn = avg_fraud_loss + administrative_fee
    
    # --- CALCUL DE COST_FP (Faux Positif) ---
    # Coût d'un analyste (10 min) + frais de communication client
    analyst_rate_per_hour = 30.0
    cost_fp = (analyst_rate_per_hour / 60 * 10) + 1.5 
    
    print("\n" + "="*40)
    print("📊 SYNTHÈSE DES COÛTS FINANCIERS")
    print("="*40)
    print(f"Perte moyenne par fraude : {avg_fraud_loss:.2f} €")
    print(f"Coût total d'un FN (Perte + Frais) : {cost_fn:.2f} €")
    print(f"Coût total d'un FP (Alerte inutile) : {cost_fp:.2f} €")
    print(f"Ratio de sévérité (FN/FP) : {cost_fn/cost_fp:.2f}")
    print("="*40 + "\n")
    
    return cost_fn, cost_fp

if __name__ == "__main__":
    calculate_operational_costs()