# Datenübersicht – SmartComplete / StoryWeaver

Dieses Verzeichnis enthält die Datensätze, die für das Training und die Evaluierung des SmartComplete-/StoryWeaver-Projekts verwendet werden.

## 1. Quellen der Rohdaten (data/raw/)
Die Rohdaten stammen aus verschiedenen öffentlich zugänglichen Quellen, beispielsweise:
- **Märchen und Sagen** aus dem Gutenberg-Projekt (Public Domain)
- **Fantasy-Geschichten** aus frei zugänglichen Sammlungen
- Eigene Sammlungen von Kurzgeschichten

Bitte beachte:
- Die verwendeten Texte in `data/raw/` sind **unverändert** und können noch Formatierungsfehler, HTML-Tags oder andere Artefakte enthalten.
- Für jeden Datensatz sollte eine separate Datei vorliegen (z. B. `maerchen.txt`, `fantasy.csv`).

## 2. Verarbeitung der Daten (data/processed/)
In diesem Ordner befinden sich die **aufbereiteten Daten**, die aus den Rohdaten erzeugt wurden:
- **Tokenisierte Texte**: Die Texte wurden in einzelne Token zerlegt.
- **Train/Val/Test-Splits**: Die Daten wurden in Trainings-, Validierungs- und Testmengen unterteilt.
- **Vokabular**: Das Vokabular (Mapping von Token zu ID) wird in der Datei `vocab.json` abgelegt.

## 3. Lizenz und Copyright
- **Öffentliche Domain**: Texte aus dem Gutenberg-Projekt und anderen Quellen, die in die Public Domain fallen, dürfen frei verwendet werden.
- **Eigene Sammlungen**: Texte, die selbst erstellt wurden, unterliegen dem Copyright des Autors.
- **Drittanbieter**: Bei der Nutzung von externen Textsammlungen wurde auf die jeweilige Lizenz geachtet. Weitere Informationen hierzu findest du in den Begleitdokumenten (z. B. in den jeweiligen Quellenangaben der Rohdaten).

## 4. Hinweise zur Vorverarbeitung
Die Daten wurden in folgenden Schritten vorverarbeitet:
- **Bereinigung**: Entfernen von HTML-Tags, überflüssigen Leerzeichen und Sonderzeichen.
- **Tokenisierung**: Zerlegung der Texte in Token (Wörter, Satzzeichen, etc.) mithilfe von [NLTK](https://www.nltk.org/).
- **Aufteilung**: Aufteilung in Trainings-, Validierungs- und Testmengen (Standard: 80/10/10).
- **Speicherung**: Die verarbeiteten Daten werden als CSV- oder Pickle-Dateien (z. B. `train_sequences.pt`) sowie das Vokabular als `vocab.json` abgelegt.

## 5. Nutzungshinweise
- Vor dem Training des Modells sollten die Daten in `data/processed/` überprüft werden.
- Änderungen an den Rohdaten erfordern eine erneute Verarbeitung in `2_data_preprocessing.ipynb`.


