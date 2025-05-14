# =================================================
# Dockerfile (Beispiel) für SmartComplete-LSTM-Projekt
# =================================================

# 1) Grundlegendes Python-Image
#    Du kannst auch python:3.9-slim oder eine andere Variante verwenden.
FROM python:3.9

# 2) Arbeitsverzeichnis im Container
WORKDIR /app

# 3) Anforderungen kopieren (requirements.txt)
COPY requirements.txt .

# 4) Installieren der Python-Abhängigkeiten
#    --no-cache-dir: Platzersparnis
RUN pip install --no-cache-dir -r requirements.txt

# 5) Projektcode kopieren
#    Damit gelangen alle relevanten Dateien (src/, data/, notebooks/, etc.) in den Container
COPY . .

# 6) Expose: Falls du einen Webserver (Streamlit, FastAPI, etc.) startest, kannst du den Port öffnen.
#    Hier z. B. 8501 für Streamlit:
# EXPOSE 8501

# 7) Standardbefehl
#    Du kannst z. B. ein Trainingsskript starten oder den Inferenz-/API-Prozess
#    Hier ein Beispiel, das "src/train.py" ausführt, falls du den Container
#    für Training bauen möchtest:
CMD ["python", "src/train.py", 
     "--train_csv", "data/processed/train.csv", 
     "--val_csv", "data/processed/val.csv", 
     "--vocab_json", "data/processed/vocab.json",
     "--epochs", "5"]

# Falls du einen Inferenzserver (API) starten willst, 
# könntest du stattdessen z. B.:
# CMD ["python", "api/main.py"]
