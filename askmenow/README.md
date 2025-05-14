# AskMeNow – FAQ-/Wissens-Q&A mit Retrieval-Augmented Generation (RAG)

**AskMeNow** ist ein Educational-Projekt, das ein Frage-Antwort-System entwickelt, das mithilfe von Retrieval-Augmented Generation (RAG) relevante Informationen aus einer Wissensbasis abruft und diese zur Generierung fundierter Antworten verwendet. Das System kombiniert klassische Retrieval-Methoden (wie BM25 oder FAISS) mit modernen Transformer-Modellen (z. B. T5), um Fragen möglichst präzise zu beantworten.

---

## Projektziel

- **Lernziel:**  
  - Den kompletten Workflow eines modernen Q&A-Systems verstehen – von der Datenaufbereitung über das Retrieval bis hin zur Antwortgenerierung.
  - Vergleich und Einsatz von klassischen Ansätzen (BM25) und Transformer-basierten Methoden (z. B. T5) im Rahmen eines RAG-Systems.
  
- **Anwendungsziel:**  
  - Entwicklung eines Systems, das Nutzerfragen automatisch beantwortet, indem es relevante Passagen aus einer Wissensbasis abruft und diese in eine zusammenhängende Antwort integriert.
  - Nutzung in Bereichen wie FAQ-Systemen, Kundensupport oder Wissensdatenbanken.

---

## Verzeichnisstruktur

askmenow/
├── data/
│   ├── raw/               # Unbearbeitete FAQ-Daten und Wissensbasis (z. B. CSV-Dateien mit Dokumenten)
│   ├── processed/         # Bereinigte, tokenisierte und ggf. vektorisierte Daten
│   ├── embeddings/        # Vorab berechnete Embeddings für die Wissensbasis (z. B. FAISS-Index-Dateien)
│   └── README.md          # Beschreibung der Wissensquellen, Retrieval-Methoden & Lizenzen
│
├── notebooks/
│   ├── 1_data_exploration.ipynb  # Analyse der FAQ-Daten und Wissensbasis
│   ├── 2_data_preprocessing.ipynb  # Bereinigung, Tokenisierung & Vektorisierung (Embeddings)
│   ├── 3_training_demo.ipynb       # Training von Q&A-Modellen (Reader)
│   ├── 4_retrieval_demo.ipynb      # Vergleich von Retrieval-Methoden (BM25, FAISS, etc.)
│   └── 5_rag_generation.ipynb      # Kombination von Retrieval und Transformer-Generierung (RAG)
│
├── src/
│   ├── dataset.py         # Dataset-Klassen & Hilfsfunktionen für Q&A-Daten
│   ├── retriever.py       # Implementierung von BM25, FAISS oder Elasticsearch für Retrieval
│   ├── model.py           # Modelle: Seq2Seq-LSTM und/oder Transformer-Reader (z. B. T5) für RAG
│   ├── train.py           # Trainingsskript (CLI) für Q&A-Generierung
│   ├── ask.py             # CLI-Skript zur Beantwortung von Nutzerfragen (Inferenz mit Retrieval)
│   ├── build_index.py     # Skript zum Aufbau eines Retrieval-Indexes (z. B. FAISS)
│   └── __init__.py        # Markiert src als Python-Paket (zentrale Importe)
│
├── .gitignore             # Ausschlussliste (z. B. __pycache__, temporäre Dateien, große Daten)
├── requirements.txt       # Liste der benötigten Python-Pakete (torch, transformers, nltk, rank_bm25, etc.)
└── README.md              # Diese Dokumentation


## Installation
Repository klonen:
git clone https://github.com/Benjamin2099/GenerativeTextLab.git
cd askmenow
Virtuelle Umgebung einrichten (empfohlen):

python3 -m venv venv
source venv/bin/activate      # Für Linux/MacOS
Für Windows: venv\Scripts\activate

## Abhängigkeiten installieren:
pip install -r requirements.txt

## Datenaufbereitung
Rohdaten:

Lege deine FAQ-Daten und Wissensquellen im Ordner data/raw/ ab.

Die Daten sollten mindestens die Fragen, Antworten oder Dokumentpassagen enthalten.

Verarbeitung:

Nutze die Notebooks in notebooks/1_data_exploration.ipynb und notebooks/2_data_preprocessing.ipynb, um die Daten zu bereinigen, zu tokenisieren und ggf. Vektoren (Embeddings) zu berechnen.

Die vorverarbeiteten Daten werden in data/processed/ und data/embeddings/ gespeichert.

## Training
Interaktives Training (Notebook)
Öffne notebooks/3_training_demo.ipynb und führe die Zellen schrittweise aus, um das Q&A-Modell (Reader) zu trainieren.

## CLI-Training
Alternativ kannst du das Trainingsskript per Kommandozeile starten:

python src/train.py --train_csv data/processed/train.csv \
                    --val_csv data/processed/val.csv \
                    --vocab_json data/processed/vocab.json \
                    --batch_size 32 --epochs 10 --lr 0.001 \
                    --embed_dim 128 --hidden_dim 256 --num_layers 2 \
                    --teacher_forcing_ratio 0.5 --save_dir models --log_interval 50
                    
## Retrieval & Inferenz
Aufbau eines Retrieval-Indexes
Nutze src/build_index.py, um einen FAISS-Index oder einen anderen Retrieval-Index auf Basis der Embeddings deiner Wissensbasis zu erstellen.

python src/build_index.py --embeddings_file data/embeddings/faq_embeddings.npy \
                          --index_output data/embeddings/faiss_index.bin \
                          --nlist 100
Beantwortung von Fragen
Um eine Frage zu beantworten, wird das Skript src/ask.py genutzt:


python src/ask.py --question "Wie funktionieren neuronale Netze?" \
                  --knowledge_csv data/processed/knowledge.csv \
                  --model_name t5-small \
                  --max_length 100
Das Skript ruft zunächst relevante Dokumente mit BM25 ab und kombiniert diese mit der Frage zu einem Prompt, der an ein vortrainiertes Transformer-Modell (z. B. T5) übergeben wird, um eine Antwort zu generieren.

## Weiterentwicklung & Ideen
Erweiterung der Wissensbasis:
Integriere weitere Datenquellen, um die Antworten zu verbessern.

Verbesserung der Retrieval-Methoden:
Experimentiere mit alternativen Ansätzen (z. B. FAISS, Elasticsearch) für eine schnellere und präzisere Suche.

Modelloptimierung:
Feintune Transformer-Modelle (z. B. T5) oder integriere RLHF, um die Antwortqualität weiter zu steigern.

Evaluation:
Implementiere automatische Metriken (wie BLEU oder ROUGE) sowie manuelle Reviews, um die generierten Antworten zu bewerten.

## Lizenz
Dieses Projekt steht unter der MIT License.

## Kontakt
Hauptautor: Benjamin

GitHub: https://github.com/Benjamin2099

Viel Erfolg beim Experimentieren, Trainieren und Weiterentwickeln von AskMeNow!