# Datenübersicht für MailAssist

In diesem Ordner findest du alle Datensätze, die für das MailAssist-Projekt verwendet werden. Das Ziel von MailAssist ist es, automatisierte E-Mail- und Textvorschläge zu generieren – beispielsweise im Stil von formellen oder informellen E-Mails. Diese Daten dienen als Grundlage, um Modelle zu trainieren, die Alltagsformulierungen verstehen und passende Vorschläge generieren können.

## 1. Herkunft der Daten

Die E-Mail-Daten stammen aus unterschiedlichen Quellen, z. B.:
- **Öffentliche E-Mail-Datensätze:**  
  - Beispielsweise von öffentlich zugänglichen Archivierungen oder Forschungsprojekten (bei denen die Inhalte anonymisiert vorliegen).
- **Eigene Sammlungen:**  
  - Interne E-Mails oder Beispieltexte, die speziell für das Projekt erstellt wurden.  
- **Kombination aus formellen und informellen Beispielen:**  
  - Formelle E-Mails, wie sie in geschäftlichen Kontexten üblich sind.
  - Informelle E-Mails, die umgangssprachlich verfasst sind.

**Wichtig:**  
Bitte beachte, dass alle verwendeten Daten den jeweiligen Lizenz- und Datenschutzbestimmungen entsprechen müssen. Für interne oder sensible Daten sollte stets eine Anonymisierung erfolgen.

## 2. Struktur der Daten

- **Raw Data (`data/raw/`):**  
  - Enthält die unbearbeiteten E-Mail-Daten.  
  - Beispieldateien:  
    - `emails_formell.csv`: E-Mails im formellen Stil.
    - `emails_informell.csv`: E-Mails im informellen Stil.
  - In diesen Dateien sind in der Regel Spalten enthalten wie:  
    - `subject`: Betreff der E-Mail.
    - `body`: Inhalt der E-Mail.
    - Optionale Metadaten: Datum, Absender, etc.

- **Processed Data (`data/processed/`):**  
  - Hier befinden sich die vorverarbeiteten Daten, die für das Training verwendet werden.  
  - Diese Daten sind in der Regel tokenisiert und in Trainings-, Validierungs- und Test-Splits aufgeteilt.
  - Beispieldateien:  
    - `train.csv`
    - `val.csv`
    - `test.csv`
  - Zusätzlich wird ein Vokabular (z. B. als `vocab.json`) erstellt und gespeichert.

## 3. Datenaufbereitung

Die Rohdaten werden im Rahmen der Vorverarbeitung wie folgt bearbeitet:
- **Bereinigung:**  
  - Entfernen von HTML-Tags, überflüssigen Leerzeichen und unerwünschten Sonderzeichen.
  - Anpassen der Groß-/Kleinschreibung (z. B. Konvertierung in Kleinbuchstaben).
- **Tokenisierung:**  
  - Zerlegung der E-Mail-Texte in einzelne Tokens (Wörter oder Subwörter), um sie in numerische Sequenzen umzuwandeln.
- **Splitting:**  
  - Aufteilung der Daten in Trainings-, Validierungs- und Test-Sets, um einen robusten Trainings- und Evaluierungsprozess zu ermöglichen.

## 4. Hinweise zur Nutzung

- **Quellenangaben & Lizenz:**  
  - Die in `data/raw/` abgelegten Dateien enthalten Informationen zu den ursprünglichen Quellen der E-Mail-Daten.  
  - Bitte stelle sicher, dass du die Lizenzbedingungen der verwendeten Daten einhältst.
- **Vorverarbeitungsprozess:**  
  - Weitere Details zur Datenbereinigung und -aufbereitung findest du in den Notebooks unter `notebooks/2_data_preprocessing.ipynb`.

