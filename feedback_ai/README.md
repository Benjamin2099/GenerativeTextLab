# Feedback AI – Lernendes System mit Nutzerbewertung

**Feedback AI** ist ein Educational-Projekt, das ein generatives Textsystem mithilfe von menschlichem Feedback (z. B. Daumen hoch/runter) kontinuierlich verbessert. Das System generiert Texte – beispielsweise E-Mail-Vorschläge, Chat-Antworten oder allgemeine Texte – und nutzt das Nutzerfeedback, um während eines Fine-Tuning-Prozesses gezielt Modelle zu optimieren. Dadurch lernt das System, bevorzugt solche Texte zu generieren, die von den Nutzern als qualitativ hochwertig bewertet werden.

---

## Projektziel

- **Lernziel:**  
  - Verständnis der vollständigen NLP-Pipeline: von der Datensammlung über die Vorverarbeitung, das Training bis zur Inferenz.
  - Vergleich von klassischen LSTM-basierten Seq2Seq-Methoden mit modernen Transformer-Ansätzen (z. B. GPT-2/T5) und dem Einsatz von Reinforcement Learning from Human Feedback (RLHF).
  
- **Anwendungsziel:**  
  - Entwicklung eines Systems, das sich durch kontinuierliches Nutzerfeedback verbessert und somit fortlaufend qualitativ bessere Texte generiert.
  - Einsatzmöglichkeiten sind u.a. automatisierte E-Mail-Vervollständigungen, Chatbots oder andere generative Anwendungen.

---

## Verzeichnisstruktur

feedback_ai/
├── data/
│   ├── raw/               # Unbearbeitete generierte Texte und Nutzerfeedback (z. B. "thumbs_up" / "thumbs_down")
│   ├── processed/         # Vorverarbeitete Daten (Beispiele mit positiver/negativer Gewichtung)
│   └── README.md          # Beschreibung der Feedback-Daten & Gewichtungsmethoden
│
├── notebooks/
│   ├── 1_data_exploration.ipynb  # Analyse von generierten Texten & Nutzerfeedback
│   ├── 2_data_preprocessing.ipynb  # Bereinigung, Tokenisierung & Labeling (Feedback)
│   ├── 3_training_demo.ipynb       # Training mit Feedback-basiertem Fine-Tuning (LSTM/Transformer)
│   └── 4_rlhf_experiment.ipynb     # Experimentelles Reinforcement Learning mit menschlichem Feedback
│
├── src/
│   ├── dataset.py         # Dataset & Hilfsfunktionen für Feedback-Daten
│   ├── model.py           # Modelle mit Feedback-Anpassung (LSTM-Reader & Transformer-Wrapper für RLHF)
│   ├── train.py           # Trainingsskript (CLI) mit feedback-basiertem Fine-Tuning
│   ├── generate.py        # Inferenzskript (CLI) zur Textgenerierung mit Feedback
│   ├── feedback.py        # Skript zur Verarbeitung von Nutzerfeedback und Model-Updates
│   └── __init__.py        # Markiert src als Python-Paket (zentrale Importe, ohne "suggest")
│
├── .gitignore             # Ausschlussliste (z. B. __pycache__, temporäre Dateien)
├── requirements.txt       # Liste der benötigten Python-Pakete (torch, transformers, nltk, etc.)
└── README.md              # Diese Hauptdokumentation

## Installation
Repository klonen:
git clone https://github.com/Benjamin2099/GenerativeTextLab.git
cd feedback_ai
Virtuelle Umgebung einrichten (empfohlen):

python3 -m venv venv
source venv/bin/activate      # Für Linux/MacOS
Für Windows: venv\Scripts\activate

## Abhängigkeiten installieren:
pip install -r requirements.txt
Datenaufbereitung
Rohdaten:
Lege generierte Texte und das dazugehörige Nutzerfeedback (z. B. "thumbs_up" oder "thumbs_down") im Ordner data/raw/ ab.

Verarbeitung:
Öffne das Notebook notebooks/2_data_preprocessing.ipynb, um die Daten zu bereinigen, zu tokenisieren und um Labels bzw. Gewichtungen zu ergänzen. Die vorverarbeiteten Daten werden als CSV-Dateien in data/processed/ gespeichert. Ebenso wird ein Vokabular (z. B. vocab.json) erstellt.

## Training
Interaktives Training (Notebook)
Öffne notebooks/3_training_demo.ipynb und führe die Zellen schrittweise aus, um das Feedback-basierte Fine-Tuning des generativen Modells zu beobachten.

## CLI-Training
Alternativ kannst du das Trainingsskript per Kommandozeile starten:

python src/train.py --train_csv data/processed/train.csv \
                    --val_csv data/processed/val.csv \
                    --vocab_json data/processed/vocab.json \
                    --batch_size 32 --epochs 10 --lr 0.001 \
                    --embed_dim 128 --hidden_dim 256 --num_layers 2 --dropout 0.1 \
                    --teacher_forcing_ratio 0.5 --save_dir models --log_interval 50
Inferenz – Textgenerierung mit Feedback
Um Textvorschläge zu generieren, nutze das CLI-Skript:


python src/generate.py --model_path models/feedback_ai_epochX_vallossY.pt \
                       --vocab_json data/processed/vocab.json \
                       --input_text "Ihr Anfangstext hier" \
                       --max_length 30
Das Skript generiert einen Textvorschlag basierend auf dem unvollständigen Input. Das Modell hat bereits Feedback in den Trainingsprozess integriert.

## Nutzerfeedback & Model-Updates
Feedback-Verarbeitung:
Das Skript feedback.py ermöglicht es, zusätzliche Feedback-Daten (z. B. in einer JSON-Datei) zu laden und das Modell basierend auf diesen Daten weiter zu fine-tunen.

Anwendung:

python src/feedback.py --feedback_json data/processed/feedback.json \
                       --vocab_json data/processed/vocab.json \
                       --model_path models/feedback_ai_best.pt \
                       --epochs 3
Projektziel & Lerninhalte
Verständnis der NLP-Pipeline:
Von der Sammlung und Aufbereitung von generierten Texten und Nutzerfeedback bis hin zum Training und der Inferenz eines generativen Systems, das sich durch menschliches Feedback verbessert.

Modellvergleich:
Vergleich zwischen klassischen LSTM-basierten Seq2Seq-Methoden und modernen Transformer-Ansätzen (unter Einbeziehung von RLHF).

Praktische Anwendung:
Entwicklung eines Systems, das kontinuierlich lernt und bessere, nutzerfreundliche Texte generiert – ideal für Anwendungen wie automatisierte E-Mail-Vervollständigungen oder Chatbots.

Weiterentwicklung & Ideen
Datenbasis erweitern:
Sammle mehr Feedback-Daten, um die Vielfalt und Qualität des Trainings zu verbessern.

Modelloptimierung:
Experimentiere mit verschiedenen Hyperparametern und Sampling-Methoden.

Transformer und RLHF:
Erweitere das Projekt um Transformer-Modelle (z. B. GPT-2/T5) und integriere fortgeschrittene RL-Methoden (Reinforcement Learning from Human Feedback), um die Textqualität weiter zu steigern.

Evaluation:
Implementiere Metriken und manuelle Reviews, um die Wirkung des Feedbacks quantitativ und qualitativ zu evaluieren.

## Lizenz
Dieses Projekt steht unter der MIT License.

## Kontakt
Hauptautor: Benjamin sat

GitHub: https://github.com/Benjamin2099

Viel Erfolg beim Experimentieren, Trainieren und Weiterentwickeln von Feedback AI!
