# Datenübersicht für SmartComplete LSTM

## Rohdaten (raw/)
1. `emails_raw.csv` - ~10.000 E-Mail-Texte, anonymisiert
2. `chatlogs.txt`   - ~5.000 Zeilen aus Chat-Protokollen
3. `short_stories.txt` - Sammlung kurzer Geschichten (gemeinfreie Quellen)

## Verarbeitungsschritte
- **Säuberung**: Entfernen von doppelten Leerzeichen, Sonderzeichen, HTML-Tags
- **Tokenisierung**: Mithilfe von NLTK (word_tokenize)
- **Aufteilung**: 80% Training, 10% Validation, 10% Test
- **Speicherung**: In PyTorch-Tensors (`.pt`) in `data/processed/`
  - `train_sequences.pt`
  - `val_sequences.pt`
  - `test_sequences.pt`
- **Vokabular**: Abgelegt in `vocab.json` (Mapping Token <-> ID)

## Verarbeitete Daten (processed/)
- `train_sequences.pt` - Enthält ~100k Token-Sequenzen
- `val_sequences.pt`   - ~12k Sequenzen für Validierung
- `test_sequences.pt`  - ~12k Sequenzen für Finale Tests
- `vocab.json`         - Enthält das generierte Vokabular, ca. 15.000 Einträge
