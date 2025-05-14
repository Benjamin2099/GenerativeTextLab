# MailAssist – Automatisierte E-Mail-/Textvorschläge

**MailAssist** ist ein Educational-Projekt, das sich mit der automatisierten Generierung von E-Mail- und Textvorschlägen beschäftigt. Ziel ist es, ein System zu entwickeln, das in der Lage ist, unvollständige E-Mail-Texte oder Eingaben zu vervollständigen – ähnlich wie Gmail Smart Compose. Dabei werden sowohl klassische LSTM-basierte Seq2Seq-Modelle als auch moderne Transformer-Ansätze (z. B. GPT-2 oder T5) erforscht.

---

## Projektziel

- **Lernziel:**  
  - Verstehen der kompletten NLP-Pipeline – von der Datenaufbereitung über das Training bis zur Inferenz.
  - Vergleich zwischen traditionellen LSTM-basierten Ansätzen und modernen Transformer-Methoden.
  
- **Anwendungsziel:**  
  - Entwicklung eines Systems, das automatisierte Vorschläge für E-Mail-Texte generiert.
  - Unterstützung von Benutzern, die schnelle und passende Formulierungen für ihre E-Mails benötigen.

---

## Features

- **Datenaufbereitung:**  
  - Rohdaten (E-Mail-Betreff und -Body) werden aus verschiedenen Quellen gesammelt.
  - Bereinigung, Tokenisierung und Aufteilung in Trainings-, Validierungs- und Test-Sets.
  - Erstellung eines Vokabulars (z. B. `vocab.json`).

- **Modellierung:**  
  - **Seq2Seq-LSTM-Modell:** Klassischer Encoder-Decoder-Ansatz zur E-Mail-Vervollständigung.
  - **Transformer-Wrapper:** Optionaler Einsatz von vortrainierten Modellen (z. B. GPT-2 oder T5) für natürlichere Textvorschläge.

- **Training und Evaluierung:**  
  - CLI-Skript (`train.py`) zum Training des Modells.
  - Notebooks zur interaktiven Datenexploration und zum Training (`3_training_demo.ipynb`).

- **Inferenz:**  
  - CLI-Skript (`suggest.py`) zur Generierung von E-Mail-/Textvorschlägen basierend auf einem unvollständigen Input.

---

## Verzeichnisstruktur

mailassist/
├── data/
│   ├── raw/               # Unbearbeitete E-Mail-Daten (formell, informell)
│   ├── processed/         # Tokenisierte E-Mail-Daten, Splits
│   └── README.md          # Beschreibung der E-Mail-Daten, Quellen & Lizenzen
│
├── notebooks/
│   ├── 1_data_exploration.ipynb  # Analyse der E-Mail-Texte
│   ├── 2_data_preprocessing.ipynb  # Bereinigung, Tokenisierung & Splitting
│   └── 3_training_demo.ipynb       # Training & Evaluierung der E-Mail-Vervollständigung
│
├── src/
│   ├── dataset.py         # Dataset & Hilfsfunktionen für E-Mail-Daten
│   ├── model.py           # Seq2Seq-LSTM &/oder Transformer-Wrapper für E-Mail-Vervollständigung
│   ├── train.py           # Trainingsskript (CLI)
│   ├── suggest.py         # Inferenzskript (CLI) für E-Mail-/Textvorschläge
│   └── __init__.py        # Markiert src als Python-Paket
│
├── .gitignore             # Ausschlussliste (z. B. __pycache__, temporäre Dateien)
├── requirements.txt       # Liste der benötigten Python-Pakete (torch, transformers, nltk, etc.)
└── README.md              # Diese Hauptdokumentation


## Installation
Repository klonen:
git clone https://github.com/Benjamin2099/GenerativeTextLab.git
cd mailassist
Virtuelle Umgebung einrichten (empfohlen):

python3 -m venv venv
source venv/bin/activate      # Für Linux/MacOS
Für Windows: venv\Scripts\activate

## Abhängigkeiten installieren:

pip install -r requirements.txt
Datenaufbereitung
Rohdaten:
Lege deine unbearbeiteten E-Mail-Daten (z. B. Betreff und Body) im Ordner data/raw/ ab.

Verarbeitung:
Öffne das Notebook notebooks/2_data_preprocessing.ipynb, um die Daten zu bereinigen, zu tokenisieren und in Trainings-, Validierungs- und Test-Sets aufzuteilen. Das Vokabular wird dabei als vocab.json in data/processed/ gespeichert.

## Training
Interaktives Training (Notebook)
Öffne notebooks/3_training_demo.ipynb und führe die Zellen schrittweise aus, um den Trainingsprozess zu beobachten und erste Ergebnisse zu evaluieren.

## CLI-Training
Alternativ kannst du das Trainingsskript per Kommandozeile starten:

python src/train.py --train_csv data/processed/train.csv \
                    --val_csv data/processed/val.csv \
                    --vocab_json data/processed/vocab.json \
                    --batch_size 32 --epochs 10 --lr 0.001 \
                    --embed_dim 128 --hidden_dim 256 --num_layers 2 --dropout 0.1 \
                    --teacher_forcing_ratio 0.5 --save_dir models --log_interval 50
Inferenz – E-Mail-/Textvorschläge generieren
Um Vorschläge zu generieren, nutze das CLI-Skript:

python src/suggest.py --model_path models/mailassist_epochX.pt \
                      --vocab_json data/processed/vocab.json \
                      --input_text "sehr geehrte damen und herren, ich möchte sie darüber informieren" \
                      --max_length 30
Das Skript gibt dann einen vervollständigten E-Mail-Text als Vorschlag aus.

## Weiterentwicklung & Ideen
Erweiterung der Datenbasis:
Nutze mehr E-Mail-Daten, um die Vielfalt der Vorschläge zu erhöhen.

Modelloptimierung:
Experimentiere mit verschiedenen Hyperparametern, Sampling-Methoden oder erweitere das Modell um Transformer-basierte Ansätze für noch natürlichere Vorschläge.

Anpassung an den Kontext:
Entwickle Mechanismen, die den Kontext (z. B. bisherige E-Mail-Konversationen) berücksichtigen, um konsistentere und passgenauere Vorschläge zu liefern.

## Projektziel & Lerninhalte
Verständnis der NLP-Pipeline:
Vom Sammeln und Vorverarbeiten von E-Mail-Daten über das Training eines Seq2Seq-Modells bis hin zur Inferenz.

Modellvergleich:
Vergleich zwischen klassischen LSTM-basierten Ansätzen und modernen Transformer-Methoden.

Praktische Anwendung:
Entwicklung eines Systems, das Benutzern hilft, schneller passende E-Mail-Formulierungen zu finden.

## Lizenz
Dieses Projekt steht unter der MIT License.

## Kontakt
Hauptautor: Benjamin

GitHub: https://github.com/Benjamin2099

Viel Erfolg beim Experimentieren und Weiterentwickeln von MailAssist!

