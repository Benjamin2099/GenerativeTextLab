# Datenübersicht für StyleShift

In diesem Ordner befinden sich die Datensätze, die für den **Text-Stil-Transfer** verwendet werden. Ziel des Projekts ist es, Texte von einem Ausgangsstil in einen anderen zu transformieren – beispielsweise einen modernen Satz in den Stil von Shakespeare oder in einen anderen gewünschten Stil.

## 1. Herkunft der Datensätze

Die Rohdaten in diesem Projekt können aus unterschiedlichen Quellen stammen, z. B.:

- **Literatur und klassische Texte**:  
  - Beispielsweise Werke, die im Public Domain sind, wie klassische Dramen oder Gedichte von Shakespeare.
  
- **Moderne Texte**:  
  - Zeitgenössische Artikel, Blogs oder Dialoge, die den "modernen" Stil repräsentieren.
  
- **Parallelisierte Datensätze**:  
  - Eigene Sammlungen oder öffentlich verfügbare Datensätze, in denen Texte bereits in zwei unterschiedlichen Stilen (Quelltext und Zielstil) vorliegen.  
  - Beispiel: Jede Zeile enthält einen modernen Satz und den entsprechenden Satz im gewünschten Zielstil (z. B. formell vs. informell, modern vs. Shakespeare).

## 2. Datenaufbereitung

Die Daten werden in zwei Hauptkategorien organisiert:

- **Raw Data (`data/raw/`)**:  
  - Hier werden die Originaltexte gespeichert, bevor jegliche Bereinigung oder Tokenisierung erfolgt.  
  - Beispiele:
    - `modern_texts.txt` – Sammlung moderner Texte.
    - `shakespeare_texts.txt` – Sammlung klassischer, stilisierter Texte (z. B. von Shakespeare).

- **Processed Data (`data/processed/`)**:  
  - Hier befinden sich die vorverarbeiteten Datensätze, die für das Training verwendet werden.  
  - Diese Dateien enthalten in der Regel:
    - Tokenisierte Versionen der Texte.
    - Trainings-, Validierungs- und Test-Splits.
    - Ein Vokabular (`vocab.json`), das für beide Stilarten erstellt wurde.
  - Beispiel-Dateien:
    - `train.csv` – Enthält parallelisierte Textpaare (z. B. moderne Texte und ihre stilisierten Versionen).
    - `val.csv`
    - `test.csv`

## 3. Beschreibung der Stilarten

Im Rahmen von **StyleShift** werden unterschiedliche Stile betrachtet. Beispiele für mögliche Stilpaare sind:

- **Modern vs. Shakespeare**:  
  - Modern: "Hey, was geht?"  
  - Shakespeare: "Pray, what doth thy spirit declare?"

- **Formell vs. Informell**:  
  - Formell: "Sehr geehrte Damen und Herren, ich bitte um Ihre Aufmerksamkeit."  
  - Informell: "Hey Leute, hört mal her!"

- **Neutral vs. Poetisch**:  
  - Neutral: "Die Sonne geht heute auf."  
  - Poetisch: "Mit goldnem Glanz erhebt sich der Morgen."

Die Wahl der Stilarten hängt von den zur Verfügung stehenden Daten ab. In den vorverarbeiteten Datensätzen sind die Texte in ihren jeweiligen Stilen bereits zugeordnet und parallelisiert, sodass das Modell lernen kann, wie ein Satz in den Zielstil transformiert wird.

## 4. Hinweise zur Nutzung der Daten

- **Datenquelle und Lizenz**:  
  - Stelle sicher, dass alle verwendeten Texte entsprechend der Lizenzbedingungen genutzt werden dürfen (z. B. Public Domain oder Creative Commons).
  
- **Vorverarbeitung**:  
  - Die im Ordner `data/processed/` abgelegten Daten sind bereits bereinigt und tokenisiert. Weitere Schritte (z. B. Padding, Sequenzbildung) erfolgen im Trainingscode.
  
- **Erweiterungsmöglichkeiten**:  
  - Du kannst zusätzliche Stilpaare integrieren oder den Datensatz erweitern, um den Stiltransfer weiter zu verfeinern.


