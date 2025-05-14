# StyleShift – Text-Stil-Transfer

**StyleShift** ist ein Educational-Projekt, das sich mit dem Transfer von Textstilen beschäftigt. Ziel ist es, Texte von einem Ausgangsstil in einen anderen zu transformieren – etwa einen modernen Satz in den Shakespeare-Stil oder zwischen formell und informell. Dieses Projekt demonstriert sowohl klassische Ansätze mit Seq2Seq-LSTM als auch moderne Transformer-basierte Modelle (z. B. GPT-2, T5), um die Unterschiede und Vorteile beider Ansätze praxisnah zu vermitteln.

---

## Projektziel

- **Lernziel:**  
  Den kompletten Workflow im Bereich Text-Stil-Transfer verstehen – von der Datenaufbereitung über das Modelltraining bis hin zur Inferenz.
  
- **Anwendungsziel:**  
  Ein System zu entwickeln, das in der Lage ist, Texte automatisch in einen gewünschten Stil zu übertragen.  
  Beispielsweise:
  - Modern → Shakespeare
  - Formell → Informell
  - Neutral → Poetisch

---

## Features

- **Datenaufbereitung:**  
  - Rohdaten in unterschiedlichen Stilen (z. B. moderne Texte vs. klassische Texte) werden gesammelt.
  - Bereinigung, Tokenisierung und Aufteilung in Trainings-, Validierungs- und Test-Sets.
  - Erstellung eines gemeinsamen Vokabulars.

- **Modellierung:**  
  - **Seq2Seq-LSTM-Modell:**  
    Ein klassischer Encoder-Decoder-Ansatz, der einfache Stiltransfers demonstriert.
  - **Transformer-Modelle:**  
    Einsatz von vortrainierten Modellen wie GPT-2 oder T5, die durch Fine-Tuning noch realistischere Ergebnisse liefern können.
  
- **Training und Evaluierung:**  
  - Interaktive Notebooks und CLI-Skripte für Training und Evaluierung.
  - Möglichkeit zum Vergleich verschiedener Ansätze.

- **Inferenz:**  
  - CLI-Skript, das es ermöglicht, einen gegebenen Text in den Zielstil zu übertragen.

---

## Verzeichnisstruktur

styleshift/
├── data/
│   ├── raw/               # Unbearbeitete Textpaare (z.B. modern vs. Shakespeare)
│   ├── processed/         # Tokenisierte und aufbereitete Daten (Train/Val/Test)
│   └── README.md          # Beschreibung der Datensätze & Stilarten
│
├── notebooks/
│   ├── 1_data_exploration.ipynb   # Untersuchung der Textstile, Längenverteilung & Tokenstatistiken
│   ├── 2_data_preprocessing.ipynb # Datenbereinigung, Tokenisierung & Split-Erstellung
│   └── 3_training_demo.ipynb      # Prototypisches Training & erste Inferenz-Demos
│
├── src/
│   ├── dataset.py         # PyTorch Dataset-Klassen & Hilfsfunktionen
│   ├── model.py           # Seq2Seq-LSTM-Modell & Transformer-Wrapper (GPT-2/T5)
│   ├── train.py           # Haupt-Trainingsskript (CLI) für den Stiltransfer
│   ├── transfer.py        # Inferenzskript (CLI) zur Übertragung eines Textes in den Zielstil
│   └── __init__.py        # Markiert src als Python-Paket
│
├── .gitignore             # Ausschlussliste (z. B. __pycache__, temporäre Dateien)
├── requirements.txt       # Liste der benötigten Python-Pakete (torch, transformers, nltk, etc.)
└── README.md              # Diese Hauptdokumentation (Setup, Usage, Projektziel, Weiterentwicklung)


## Installation
Repository klonen:

git clone https://github.com/Benjamin2099/GenerativeTextLab.git
cd styleshift
Virtuelle Umgebung einrichten (empfohlen):


python3 -m venv venv
source venv/bin/activate      # Linux/MacOS
Für Windows: venv\Scripts\activate

## Abhängigkeiten installieren:

pip install -r requirements.txt
Datenaufbereitung
Rohdaten bereitstellen:

Lege deine Originaltextpaare im Ordner data/raw/ ab (z. B. in einer CSV-Datei, in der jede Zeile einen modernen Text und seinen stilisierten Pendant enthält).

Datenexploration:

Öffne und führe notebooks/1_data_exploration.ipynb aus, um einen Überblick über die Textlängen, den Wortschatz und Unterschiede zwischen den Stilen zu gewinnen.

Datenverarbeitung:

Nutze notebooks/2_data_preprocessing.ipynb zur Bereinigung (Entfernen von HTML-Tags, überflüssigen Leerzeichen), zur Tokenisierung und zum Split in Trainings-, Validierungs- und Test-Sets.

Dabei wird auch ein Vokabular (z. B. vocab.json) erstellt, das du im Training und in der Inferenz verwendest.

## Training
Interaktives Training (Notebook)
Öffne notebooks/3_training_demo.ipynb und führe die Zellen schrittweise aus, um den Trainingsprozess zu beobachten und erste Stiltransfer-Demos zu erhalten.

Kommandozeilenbasiertes Training
Alternativ kannst du das CLI-Skript nutzen:


python src/train.py --train_csv data/processed/train.csv \
                    --val_csv data/processed/val.csv \
                    --vocab_json data/processed/vocab.json \
                    --batch_size 32 --epochs 10 --lr 0.001 \
                    --embed_dim 128 --hidden_dim 256 --num_layers 2 --dropout 0.1 \
                    --teacher_forcing_ratio 0.5 --save_dir models --log_interval 50
Inferenz – Stiltransfer
Um einen modernen Text in einen anderen Stil zu übertragen, nutze das CLI-Skript:


python src/transfer.py --model_path models/styleshift_epochX.pt \
                       --vocab_json data/processed/vocab.json \
                       --input_text "ich freue mich auf den sommer" \
                       --max_length 20
Das Skript lädt das vortrainierte Modell, tokenisiert den Input und generiert schrittweise den Text im Zielstil (z. B. Shakespeare).

## Projektziel & Lerninhalte
Lernziel:
Verstehe den kompletten Workflow von der Datenaufbereitung über das Training bis zur Inferenz im Bereich Text-Stil-Transfer.

Vergleich von Ansätzen:

LSTM-basierte Modelle: Einfacher zu implementieren, geringerer Hardwarebedarf, aber begrenztes Kontextverständnis.

Transformer-basierte Modelle: Bieten State-of-the-Art-Qualität, können längere Kontexte verarbeiten, sind jedoch hardwareintensiver.

Praktische Anwendung:
Entwickle ein System, das Texte von einem Ausgangsstil in einen gewünschten Zielstil überträgt – ideal für kreative Textanpassungen, Marketing oder literarische Anwendungen.

## Weiterentwicklung & Ideen
Hyperparameter-Tuning:
Experimentiere mit unterschiedlichen Embedding- und Hidden-Dimensionen, LSTM-Schichten, Dropout und Teacher Forcing.

Transformer-Ansätze:
Erweitere das Projekt um Transformer-Modelle (z. B. GPT-2 oder T5) für noch realistischere Stiltransfers.

Daten erweitern:
Integriere weitere Stilpaare, um das Modell auf unterschiedliche Sprachstile zu trainieren (z. B. modern, formell, poetisch, umgangssprachlich).

Evaluation:
Implementiere Metriken zur Bewertung der Stilübertragung (z. B. BLEU, ROUGE) und führe qualitative Vergleiche durch.

## Lizenz
Dieses Projekt steht unter der MIT License.

## Kontakt
Hauptautor: Benjamin

GitHub: https://github.com/Benjamin2099

Viel Erfolg beim Experimentieren, Trainieren und Weiterentwickeln von StyleShift!