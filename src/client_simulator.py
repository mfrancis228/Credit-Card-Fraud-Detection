import pandas as pd
import asyncio
import httpx
import time
import os
from datetime import datetime
from load_data import load_raw_data
import traceback

class FraudClient:
    def __init__(self, api_url, duration_seconds):
        self.api_url = f"{api_url}/predict"
        self.duration = duration_seconds
        self.results = []
        self.start_time = None

    async def send_transaction(self, client, row_dict, true_label):
        """Envoie une requête unique et stocke le résultat."""
        try:
            # Préparer les données (on enlève 'Class' car l'API ne l'attend pas)
            payload = {k: v for k, v in row_dict.items() if k != 'Class'}
            
            response = await client.post(self.api_url, json=payload)
            
            if response.status_code == 200:
                res_json = response.json()
                # On combine l'entrée, le vrai label et la prédiction
                record = {**row_dict, 
                          "true_label": true_label, 
                          "pred_prob": res_json['fraud_probability'],
                          "decision": res_json['decision']}
                self.results.append(record)
        except Exception as e:
            print(f"\n\n\nErreur lors de l'envoi : {e}")
            traceback.print_exc()

    async def run_simulation(self):
        df = load_raw_data()
        if df is None: return
        
        # On mélange les données pour avoir un échantillon varié
        test_data = df.sample(frac=1).to_dict('records')
        
        print(f"🚀 Démarrage de la simulation pendant {self.duration} secondes...")
        self.start_time = time.time()
        
        # Utilisation d'un client HTTP asynchrone pour la performance
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        async with httpx.AsyncClient(limits=limits, timeout=10.0) as client:
            tasks = []
            idx = 0
            
            while time.time() - self.start_time < self.duration:
                row = test_data[idx % len(test_data)]
                task = asyncio.create_task(self.send_transaction(client, row, row['Class']))
                tasks.append(task)
                idx += 1
                
                # Petit délai pour ne pas saturer la boucle locale d'événements
                if idx % 50 == 0:
                    await asyncio.sleep(0.1)
            
            # Attendre que toutes les requêtes en cours se terminent
            await asyncio.gather(*tasks)

    def save_results(self):
        if not self.results:
            print("Aucun résultat à sauvegarder.")
            return

        # Création du dossier dédié
        output_dir = "Outputs/client_logs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Génération du nom de fichier avec date
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"simulation_results_{timestamp}.xlsx"
        filepath = os.path.join(output_dir, filename)
        
        # Sauvegarde en Excel
        df_res = pd.DataFrame(self.results)
        df_res.to_excel(filepath, index=False)
        print(f"✅ Simulation terminée : {len(self.results)} requêtes traitées.")
        print(f"📂 Résultats sauvegardés dans : {filepath}")

if __name__ == "__main__":
    # Remplacez par l'URL fournie par 'minikube service fds-service --url'
    # URL_K8S = "http://127.0.0.1:8000" # <--- À CHANGER
    URL_K8S = "http://127.0.0.1:50606" # <--- À CHANGER
    DUREE = 12 # secondes
    
    client_sim = FraudClient(URL_K8S, DUREE)
    asyncio.run(client_sim.run_simulation())
    client_sim.save_results()