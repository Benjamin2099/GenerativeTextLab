# ChatPal – Ein Einfacher Chatbot

**ChatPal** ist ein Educational-Projekt, das einen einfachen Chatbot implementiert, der auf Dialogdaten trainiert wird. Das Ziel ist es, grundlegende Konzepte der Generativen KI (mit LSTM und optional Transformer) verständlich zu machen und einen interaktiven Chatbot zu demonstrieren.

## Projektziel

- **Verständnis vermitteln**: Die Studierenden lernen, wie man ein LSTM-basiertes Sprachmodell für Chat-Daten aufbaut, trainiert und einsetzt.
- **Interaktive Anwendung**: ChatPal soll es ermöglichen, in Echtzeit mit dem Chatbot zu "chatten" und so die Prinzipien von Dialogsystemen zu erfassen.
- **Grundlagen der NLP-Pipeline**: Von der Datenaufbereitung über das Training bis zur Inferenz.

## Verzeichnisstruktur

chatpal/
├── data/
│   ├── raw/               # Unbearbeitete Chatlogs oder Q&A-Daten (CSV, TXT, etc.)
│   ├── processed/         # Vorverarbeitete Daten (tokenisiert, in Train/Val Splits)
│   └── README.md          # Beschreibung der Dialogdaten, Quellen & Lizenzen
├── notebooks/
│   ├── 1_data_exploration.ipynb    # Erste Analyse der Rohdaten
│   ├── 2_data_preprocessing.ipynb  # Datenbereinigung, Tokenisierung & Splitting
│   └── 3_training_demo.ipynb       # Prototypischer Trainings- und Inferenzablauf
├── src/
│   ├── dataset.py         # PyTorch Dataset & Hilfsfunktionen, speziell für Dialoge
│   ├── model.py           # LSTM-/Transformer-Modelle (oder Wrapper für GPT-2)
│   ├── train.py           # Haupt-Trainingsskript (CLI)
│   ├── chat.py            # Inferenzskript (CLI) zum interaktiven Chatten mit dem Bot
│   └── __init__.py        # Markiert das src-Verzeichnis als Python-Paket
├── .gitignore             # Ausschlussliste (z. B. __pycache__, Modelle, Daten etc.)
├── requirements.txt       # Liste der benötigten Python-Pakete (torch, numpy, nltk, etc.)
└── README.md              # Diese Datei – Hauptdokumentation, Setup, Usage, Projektziel

## Installation
Repository klonen
git clone https://github.com/Benjmain2099/GenerativeTextLab.git
cd chatpal
Virtuelle Umgebung einrichten (optional, aber empfohlen)

python3 -m venv venv
source venv/bin/activate      # Für Linux/MacOS
Windows: venv\Scripts\activate

## Abhängigkeiten installieren
pip install -r requirements.txt

## Datenaufbereitung
Lege deine Rohdaten (Chatlogs, Q&A-Paare) im Ordner data/raw/ ab.

Führe das Notebook notebooks/1_data_exploration.ipynb aus, um einen Überblick über die Daten zu erhalten.

Nutze notebooks/2_data_preprocessing.ipynb, um die Daten zu bereinigen, zu tokenisieren und in Trainings-/Validierungssplits aufzuteilen.

Die verarbeiteten Daten (sowie das Vokabular als vocab.json) werden im Ordner data/processed/ gespeichert.

## Training
Interaktives Training im Notebook:
Öffne und führe notebooks/3_training_demo.ipynb aus, um einen ersten Prototypen des Chatbots zu trainieren und erste Inferenz-Ergebnisse zu sehen.

Kommandozeilenbasiertes Training:
Starte das Trainingsskript über die Kommandozeile:


python src/train.py --train_csv data/processed/train.csv \
                    --val_csv data/processed/val.csv \
                    --vocab_json data/processed/vocab.json \
                    --batch_size 32 --epochs 5 --lr 0.001 \
                    --embed_dim 128 --hidden_dim 256 --num_layers 2 --dropout 0.1 \
                    --save_dir models --log_interval 50
Chat – Interaktive Inferenz
Chatbot starten:
Um mit dem Chatbot zu interagieren, nutze das CLI-Skript:

python src/chat.py --model_path models/chatpal_epoch3_valloss2.1357.pt \
                   --vocab_json data/processed/vocab.json \
                   --max_length 20 --top_k 1 --temperature 1.0
Tippe deine Eingaben ein und erhalte direkt Antworten vom Bot.

## Projektziel und Lerninhalte
Grundlegende NLP-Pipeline: Von der Datenaufbereitung über das Training bis zur Inferenz.

LSTM vs. Transformer: Verstehe die Unterschiede zwischen klassischen LSTM-Modellen und modernen Transformer-Ansätzen (optional).

Interaktive Anwendung: Erlebe, wie ein Chatbot funktioniert und lerne, wie man Modelle für Dialoge trainiert und einsetzt.

Didaktische Struktur: Die einzelnen Notebooks und Skripte sind so aufgebaut, dass sie den Lernprozess Schritt für Schritt unterstützen.

## Weiterentwicklung
Verbessere die Inferenz: Experimentiere mit Top-k- oder Temperature-Sampling.

Erweitere den Kontext: Füge Mechanismen hinzu, um den Chatverlauf zu speichern und den Kontext über mehrere Turns zu berücksichtigen.

Transformer-Ansätze: Fine-tune GPT-2 oder GPT-3 auf Dialogdaten für noch flüssigere und kontextbewusstere Antworten.


Viel Spaß beim Lernen und Experimentieren mit ChatPal!

