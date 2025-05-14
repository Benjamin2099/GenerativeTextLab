# StoryWeaver – Kreative Geschichtengenerierung mit KI

StoryWeaver ist ein Educational-Projekt zur Erzeugung von kreativen Geschichten mithilfe von Generative KI. Das Projekt zeigt, wie man mithilfe eines LSTM-Sprachmodells (und optional auch Transformer-Modelle) Texte generieren kann – beginnend bei der Datenaufbereitung über das Training bis hin zur Inferenz.

## Inhaltsverzeichnis

1. Projektziel
2. Features
3. Verzeichnisstruktur
4. Installation
5. Datenaufbereitung
6. Training
7. Textgenerierung (Inference)
8. Weitere Hinweise & Roadmap
9. Lizenz

## Projektziel

Das Ziel von StoryWeaver ist es, Studierenden und Interessierten einen praxisnahen Einblick in generative KI zu bieten – insbesondere in den Bereich der kreativen Textgenerierung. Anhand eines LSTM-Modells (und optional Transformer-Ansätzen) lernst du, wie man:
- Rohdaten (z. B. Märchen, Fantasy-Geschichten, Romane) säubert und tokenisiert,
- ein Sprachmodell trainiert und
- auf Basis eines Start-Prompts fortlaufend Texte generiert.

## Features

- **Datenaufbereitung**: Verarbeitung von Rohdaten, Tokenisierung und Vokabularerstellung.
- **Modelltraining**: Aufbau und Training eines LSTM-Sprachmodells (optional mit Erweiterung auf Transformer).
- **Textgenerierung**: CLI-basierte Generierung von Geschichten, die mit einem einfachen Prompt starten.
- **Educational Setup**: Klare Struktur und Beispielcode, der sich ideal für Lehrzwecke eignet.
- **Einfachheit & Erweiterbarkeit**: Projekt ohne Docker und umfangreiche Tests, um den Fokus auf die Kernkonzepte zu legen.

## Verzeichnisstruktur

storyweaver/
├── data/
│   ├── raw/               # Unbearbeitete Originaltexte (Märchen, Fantasy-Geschichten etc.)
│   ├── processed/         # Vorverarbeitete Daten (tokenisiert, aufgeteilt in train/val/test)
│   └── README.md          # Infos zu Quellen, Lizenzen und Aufbereitungsschritten
├── notebooks/
│   ├── 1_data_exploration.ipynb   # Erste Analyse und Visualisierung der Rohdaten
│   ├── 2_data_preprocessing.ipynb   # Datenbereinigung, Tokenisierung & Vokabularbildung
│   └── 3_training_demo.ipynb        # Prototypisches Training und erste Generierungstests
├── src/
│   ├── dataset.py         # PyTorch Dataset-/DataLoader-Klassen & Hilfsfunktionen
│   ├── model.py           # LSTM-Modell (und optional Transformer-Wrapper)
│   ├── train.py           # Haupt-Trainingsskript (CLI)
│   ├── generate.py        # Inferenzskript (CLI) für Geschichtengenerierung
│   └── __init__.py        # Markierung des src-Verzeichnisses als Python-Paket
├── .gitignore             # Ausschlussliste für temporäre Dateien, Modelle etc.
├── requirements.txt       # Liste aller benötigten Python-Pakete
└── README.md              # Diese Hauptdokumentation

## Installation

Repository klonen
git clone https://github.com/Benjamin2099/GenerativeTextLab.git
cd storyweaver
Virtuelle Umgebung erstellen (empfohlen)

python3 -m venv venv
source venv/bin/activate    # Linux/Mac
Windows: venv\Scripts\activate

## Abhängigkeiten installieren
pip install -r requirements.txt
Datenaufbereitung
Lege deine Rohtexte (z. B. Märchen, Fantasy-Kapitel) in den Ordner data/raw/ ab.

Öffne das Notebook notebooks/1_data_exploration.ipynb, um einen ersten Überblick über die Daten zu erhalten.

Führe das Notebook notebooks/2_data_preprocessing.ipynb aus, um:

Die Texte zu säubern (HTML-Tags, Sonderzeichen entfernen, etc.),

Die Texte zu tokenisieren,

Ein Vokabular zu erstellen (vocab.json) und

Die Daten in Trainings-, Validierungs- und Test-Sets aufzuteilen.

Die vorverarbeiteten Daten werden im Ordner data/processed/ gespeichert.

Training
Du kannst dein Modell entweder über die Notebooks oder per Kommandozeile trainieren.

Über das Notebook
Öffne notebooks/3_training_demo.ipynb und führe die Zellen schrittweise aus. Dort findest du auch Beispiele zur Generierung.

Über die Kommandozeile
Führe das Skript src/train.py aus:

python src/train.py \
    --train_csv data/processed/train.csv \
    --val_csv data/processed/val.csv \
    --vocab_json data/processed/vocab.json \
    --batch_size 32 \
    --epochs 5 \
    --lr 0.001 \
    --embed_dim 128 \
    --hidden_dim 256 \
    --num_layers 2 \
    --dropout 0.1 \
    --save_dir models \
    --log_interval 50
Das Skript trainiert das LSTM-Modell und speichert bei Verbesserung des Validierungs-Loss Checkpoints im angegebenen Verzeichnis.

Textgenerierung (Inference)
Nach dem Training kannst du das Modell verwenden, um Geschichten zu generieren. Rufe dazu das CLI-Skript src/generate.py auf:

python src/generate.py \
    --model_path models/lstm_epoch5_valloss2.1234.pt \
    --vocab_json data/processed/vocab.json \
    --prompt "Es war einmal" \
    --max_length 50 \
    --top_k 5 \
    --temperature 1.0
Das Skript lädt das trainierte Modell, nimmt den Prompt entgegen und generiert fortlaufend Tokens, bis entweder <EOS> erreicht wird oder die maximale Länge überschritten wird.

Weitere Hinweise & Roadmap
Hyperparameter-Tuning: Experimentiere mit verschiedenen Embedding-Dimensionen, Hidden-Dimensions, Anzahl der LSTM-Schichten und Dropout-Werten.

Sampling-Methoden: Neben Greedy-Decoding kannst du Top-k oder Nucleus (Top-p) Sampling einsetzen, um kreativere Ergebnisse zu erhalten.

Transformer-Ansatz: Für fortgeschrittene Nutzer besteht die Möglichkeit, auf vortrainierte Transformer-Modelle (z. B. GPT-2) umzusteigen und diese für die Generierung feinzutunen.

Educational Workshops: Dieses Projekt eignet sich hervorragend, um Studierenden den kompletten Workflow von der Datenaufbereitung über das Training bis zur Inferenz einer generativen KI zu vermitteln.

## Lizenz
Dieses Projekt ist unter der MIT License lizenziert. Details findest du in der Datei LICENSE.

## Kontakt
Hauptautor: Benjamin

GitHub: https://github.com/Benjamin2099

Bei Fragen oder Interesse an Zusammenarbeit stehe ich gerne zur Verfügung.

Viel Spaß beim Erforschen und Entwickeln mit StoryWeaver!