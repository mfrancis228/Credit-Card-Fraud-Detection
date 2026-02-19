# 1. Image de base légère (Python 3.11 Slim)
FROM python:3.11-slim

# 2. Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 3. Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copier le fichier requirements.txt en premier (optimisation du cache)
COPY requirements.txt .

# 5. Installer les bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copier tout le contenu du projet dans le conteneur
COPY . .

# 7. Exposer le port que FastAPI va utiliser
EXPOSE 8000

# 8. Commande pour lancer l'API au démarrage du conteneur
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]