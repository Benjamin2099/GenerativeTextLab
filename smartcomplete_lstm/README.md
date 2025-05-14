# SmartComplete – Einfache Text-Vervollständigung mit LSTM

Willkommen zu **SmartComplete**, einem Projekt, das ein **LSTM-Sprachmodell** einsetzt, um Texte basierend auf einem **Prompt** automatisch zu vervollständigen. Ziel ist es, eine leicht verständliche, aber erweiterbare **Generative-KI-Lösung** zu demonstrieren. Die Struktur umfasst:

- **Datenerfassung & -vorbereitung** (`data/`)
- **Notebooks** für Exploration und Preprocessing (`notebooks/`)
- **Trainings- und Inferenzscripte** (`src/`)
- **Tests** (`tests/`)

## Inhalt
1. Projektziel
2. Features
3. Verzeichnisstruktur
4. Installation
5. Lizenz
6. Kontakt
---

## Projektziel
SmartComplete soll zeigen, wie man:
- Ein **LSTM-Sprachmodell** in **PyTorch** aufbaut und trainiert.  
- Daten von Grund auf **vorbereitet** (Tokenisierung, Vokabularbildung).  
- Eine **Basisauswahl** an Techniken (z. B. greedy, top-k) zur Textgenerierung umsetzt.  

Zudem dient es als **Beispielstruktur** für generative KI-Projekte, die du leicht erweitern kannst (z. B. mit **Transformern**).

---

## Features
- **Einfache Autovervollständigung**: LSTM generiert Wort für Wort.
- **CLI-Skripte** für Training und Generierung.
- **Interaktive Jupyter Notebooks** für Explorations- und Preprocessing-Schritte.
- **Unit- und Integrationstests** (Pytest) für zuverlässige und reproduzierbare Ergebnisse.
- **Optionale Dockerisierung** für einheitliches Deployment.

---

## Verzeichnisstruktur
smartcomplete_lstm/
├── data/
│   ├── raw/               # Unverarbeitete Originaldaten
│   ├── processed/         # Aufbereitete Daten (tokenisiert, Splits)
│   └── README.md          # Beschreibung & Herkunft der Daten
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_data_preprocessing.ipynb
│   └── 3_training_demo.ipynb
├── src/
│   ├── dataset.py         # Datenlogik (PyTorch Dataset, DataLoader, etc.)
│   ├── model.py           # LSTM-Modell
│   ├── train.py           # Haupt-Trainingsskript (CLI)
│   ├── generate.py        # Inferenzskript (CLI)
│   └── __init__.py
├── tests/
│   ├── test_dataset.py    # Tests für dataset.py
│   ├── test_model.py      # Tests für model.py
│   └── test_integration.py# Integrationstests (End-to-End)
├── .gitignore
├── requirements.txt
├── Dockerfile             # Optional: Containerisierung
└── README.md              # Du befindest dich hier 

## Installation
Repository klonen:
git clone https://github.com/Benjamin2099/GenerativeTextLab.git
cd mailassist
Virtuelle Umgebung einrichten (empfohlen):

python3 -m venv venv
source venv/bin/activate      # Für Linux/MacOS
Für Windows: venv\Scripts\activate


## Lizenz
Dieses Projekt steht unter der MIT License.

## Kontakt
Hauptautor: Benjamin

GitHub: https://github.com/Benjamin2099

Viel Erfolg beim Experimentieren und Weiterentwickeln von MailAssist!
