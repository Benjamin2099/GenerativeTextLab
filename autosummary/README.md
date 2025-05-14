# AutoSummary – Automatische Zusammenfassungen

**AutoSummary** ist ein Educational-Projekt zur automatischen Zusammenfassung von langen Texten. Ziel ist es, Studierenden und Interessierten die grundlegende NLP-Pipeline zu vermitteln – von der Datenaufbereitung über den Modellaufbau (entweder mit einem Seq2Seq-LSTM oder Transformer-Ansätzen wie T5/BART) bis hin zur Inferenz. Das Projekt ermöglicht es, komplexe Texte wie Nachrichtenartikel, wissenschaftliche Berichte oder Blogbeiträge in kompakte, verständliche Zusammenfassungen zu verwandeln.

---

## Projektziel

- **Lernziel:**  
  - Verstehen, wie man lange Texte automatisch zusammenfassen kann.
  - Vergleich verschiedener Modellansätze (klassisches Seq2Seq-LSTM vs. moderne Transformer-Modelle).
  - Vermittlung einer kompletten NLP-Pipeline (Datenaufbereitung, Training, Inferenz).

- **Anwendungsziel:**  
  - Entwicklung eines Prototyps, der in der Lage ist, Artikel oder Berichte automatisch und abstrakt zusammenzufassen.
  - Basis für weiterführende Projekte in der generativen KI und im Bereich der automatischen Textzusammenfassung.

---

## Verzeichnisstruktur
autosummary/
├── data/
│   ├── raw/               # Unbearbeitete Textdaten (z.B. Artikel, Berichte)
│   ├── processed/         # Vorverarbeitete Daten (tokenisiert, Train/Val/Test-Splits)
│   └── README.md          # Beschreibung der Datensätze, Quellen & Lizenzen
├── notebooks/
│   ├── 1_data_exploration.ipynb   # Erste Analyse der Rohdaten (Inhalte, Längen, Tokenstatistiken)
│   ├── 2_data_preprocessing.ipynb # Datenbereinigung, Tokenisierung & Split-Erstellung
│   └── 3_training_demo.ipynb      # Prototypisches Training & erste Inferenz-Demos
├── src/
│   ├── dataset.py         # PyTorch Dataset & Hilfsfunktionen für Summaries
│   ├── model.py           # Seq2Seq-LSTM-Modell (Encoder, Decoder) und/oder Transformer-Wrapper (T5/BART)
│   ├── train.py           # Haupt-Trainingsskript (CLI) zum Trainieren des Modells
│   ├── summarize.py       # Inferenzskript (CLI) zur Generierung von Zusammenfassungen
│   └── __init__.py        # Markiert das src-Verzeichnis als Python-Paket
├── .gitignore             # Ausschlussliste (z.B. __pycache__, temporäre Dateien)
├── requirements.txt       # Liste der benötigten Python-Pakete (torch, numpy, nltk, transformers, etc.)
└── README.md              # Diese Hauptdokumentation (Setup, Usage, Projektziel, Weiterentwicklung)


## Installation
Repository klonen:
git clone https://github.com/Benjamin2099/GenerativeTextLab.git
cd autosummary
Virtuelle Umgebung einrichten (empfohlen):

python3 -m venv venv
source venv/bin/activate     # Für Linux/MacOS
Für Windows: venv\Scripts\activate


## Abhängigkeiten installieren:
pip install -r requirements.txt

## Datenaufbereitung
Rohdaten bereitstellen:

Lege deine Rohtexte (z.B. Artikel, Berichte) im Ordner data/raw/ ab.

Eine Übersicht der Quellen und Lizenzinformationen findest du in data/README.md.

Daten explorieren:

Öffne das Notebook notebooks/1_data_exploration.ipynb, um einen ersten Überblick über die Inhalte, Textlängen und Tokenstatistiken zu erhalten.

Daten verarbeiten:

Führe das Notebook notebooks/2_data_preprocessing.ipynb aus, um die Texte zu bereinigen, zu tokenisieren und in Trainings-, Validierungs- und Test-Sets aufzuteilen.

Dabei wird auch ein Vokabular (vocab.json) erstellt und in data/processed/ gespeichert.

## Training
Interaktives Training (Notebook)
Öffne notebooks/3_training_demo.ipynb und führe die Zellen schrittweise aus, um einen Prototypen des Zusammenfassungsmodells (Seq2Seq-LSTM oder Transformer) zu trainieren.

Kommandozeilenbasiertes Training
Alternativ kannst du das Training über das CLI-Skript starten:

python src/train.py --train_csv data/processed/train.csv \
                    --val_csv data/processed/val.csv \
                    --vocab_json data/processed/vocab.json \
                    --batch_size 32 --epochs 5 --lr 0.001 \
                    --embed_dim 128 --hidden_dim 256 --num_layers 2 --dropout 0.1 \
                    --save_dir models --log_interval 50
Das Skript speichert das beste Modell basierend auf dem Validierungs-Loss im Ordner models/.

## Inferenz – Zusammenfassungen generieren
Um eine Zusammenfassung zu generieren, nutze das CLI-Skript:

python src/summarize.py --model_path models/autosummary_epoch3_valloss2.1234.pt \
                        --vocab_json data/processed/vocab.json \
                        --article "Dein langer Artikeltext, der zusammengefasst werden soll." \
                        --max_length 150
Das Skript gibt eine automatisch generierte Zusammenfassung des eingegebenen Artikels aus.

## Weiterentwicklung & Ideen
Hyperparameter-Tuning:
Experimentiere mit verschiedenen Einstellungen für Embedding-Dimensionen, Hidden-Dimensionen, Anzahl der LSTM-Schichten und Dropout.

Verbesserte Sampling-Methoden:
Ersetze das einfache Greedy-Decoding durch Top-k- oder Nucleus (Top-p)-Sampling, um variablere Zusammenfassungen zu erhalten.

Transformer-Ansatz:
Fine-tune vortrainierte Modelle wie T5 oder BART, um noch qualitativ hochwertigere Zusammenfassungen zu generieren.

Domänenspezifische Anpassungen:
Trainiere das Modell auf spezifischen Textkorpora (z.B. medizinische Berichte oder technische Dokumentationen), um den Output weiter zu optimieren.

Evaluation und Feedback:
Implementiere Metriken wie ROUGE oder BLEU, um die Qualität der generierten Zusammenfassungen objektiv zu bewerten.

## Projektziel & Lerninhalte
Verständnis der NLP-Pipeline:
Von der Rohdatenerfassung über die Vorverarbeitung bis hin zum Training und der Inferenz.

Vergleich verschiedener Modellansätze:
LSTM-basierte Seq2Seq-Modelle versus Transformer-Modelle (z.B. T5/BART) für die Zusammenfassung.

Praktische Anwendung:
Entwicklung eines Prototyps, der es ermöglicht, lange Texte automatisch zusammenzufassen – ideal für den Einsatz in Bildungs- und Forschungsprojekten.


Viel Spaß beim Erkunden, Trainieren und Weiterentwickeln von AutoSummary!
