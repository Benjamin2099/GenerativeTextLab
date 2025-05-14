# Datenübersicht für ChatPal

In diesem Ordner liegen sämtliche Daten für deinen Chatbot:
- **`raw/`**: Unbearbeitete Dialogdaten (z. B. Chat-Logs, Q&A-Paare, E-Mail-Ketten).
- **`processed/`**: Aufbereitete Daten (tokenisiert, in Trainings-/Validierungs- (und optional Test-) Splits unterteilt).

## 1. Herkunft und Lizenz

- **Quelle**:  
  - Möglicherweise selbst erstellte Chat-Logs (Privat/Eigenes Team)  
  - Öffentliche Chat-Datensätze (z. B. [Reddit-Kommentare](https://www.reddit.com/), [OpenSubtitles](http://opus.nlpl.eu/OpenSubtitles.php) – allerdings oft sehr groß und evtl. lizenzrechtlich eingeschränkt)  
  - FAQ-Paare aus einer Firmendatenbank (Achtung auf NDA oder Datenschutz)  

- **Lizenz**:  
  - Bitte prüfen, ob du die Daten weitergeben darfst. Bei vielen Foren (z. B. Reddit) brauchst du Genehmigungen oder musst Nutzer anonymisieren.  
  - Selbst erstellte Daten kannst du frei verwenden, solange keine persönlichen Daten enthalten sind.  

## 2. Datendateien & Formate

- **`raw/`**  
  - Beispiel: `chatlogs_raw.csv` mit Spalten wie:
    - `user_input`: Was der User eingibt  
    - `bot_response` (nur bei Q&A-Paaren)  
    - oder Spalten für **Zeitstempel**, **Konversations-ID** etc.  
  - Evtl. mehrere Dateien, falls du mehrere Datenquellen mischst.

- **`processed/`**  
  - Enthält die **bereinigten** Datensätze (z. B. ohne HTML, ohne Sonderzeichen).  
  - Enthält bereits **tokenisierte** Versionen (`token_ids`) oder `train.csv`, `val.csv`.  
  - Beispiel:  
    ```
    train.csv
    val.csv
    vocab.json
    ```
  - `train.csv` könnte Spalten wie `user_token_ids`, `bot_token_ids` oder **ein gemeinsames Feld** haben, je nach Trainingsstrategie.

## 3. Verarbeitungsschritte

1. **Säubern**: Entfernen von unerwünschten Sonderzeichen (HTML, Emojis optional), Lowercasing, Pseudonymisierung (Name → „UserA“).  
2. **Tokenisierung**: Zerlegung in Wörter oder Subwörter (je nach Modell).  
3. **Sequenzbildung** (falls LSTM):  
   - Bilde `(input_seq, target_seq)` – z. B. `(Userfrage, Botantwort)`.  
   - Bei mehrschrittigen Dialogen ggf. Kontext in X vorherige Turns einbeziehen.  
4. **Split in train/val** (und test optional):  
   - Du könntest nach Konversationen splitten, damit du keine „Leakage“ hast (keine Konversationsteile im Train- und Val-Set).  

## 4. Tipps & Hinweise

- Ein **einfacher LSTM-Chatbot** kann gut auf kurzen, klar strukturierten Paaren (Frage→Antwort) funktionieren.  
- Für **längere Gespräche** ist ein reiner LSTM oft überfordert → begrenzter Kontext.  
- **Transformer-Ansätze** (GPT-2/3) erlauben größere Kontexte, aber benötigen meist mehr Rechenressourcen.  
- Achte auf **Datenschutz** und **Anonymisierung**, vor allem wenn du Chatlogs von Personen sammelst.

## 5. Beispieldateien & Größen

- **`chatlogs_raw.csv`** (in `raw/`):  
  - ~10.000 Zeilen, Spalten: `[conversation_id, user_message, bot_response, timestamp]`  
  - ~1MB Größe  
- **`train.csv`** (in `processed/`):  
  - ~8.000 Zeilen nach Tokenisierung  
  - Spalten: `user_token_ids, bot_token_ids` (Listenform wie `[2, 45, 60, 13]`)  
- **`val.csv`**: ~1.000 Zeilen  
- **`test.csv`** (falls vorhanden): ~1.000 Zeilen  
