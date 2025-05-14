"""
tests/test_dataset.py
---------------------
Unit-Tests für die Dataset- und DataLoader-Funktionalitäten aus src/dataset.py.
Diese Tests überprüfen:
  - Ob das Dataset korrekt aus einer CSV-Datei gelesen wird.
  - Ob die Token-IDs (als Liste) richtig interpretiert und in Input- und Target-Sequenzen umgewandelt werden.
  - Ob die Collate-Funktion die Sequenzen richtig padded.
"""

import pytest
import torch
import pandas as pd
from src.dataset import StoryDataset, collate_fn, create_dataloader, load_vocab

# Wir verwenden Pytests Fixture, um eine temporäre CSV-Datei zu erstellen
@pytest.fixture
def sample_csv(tmp_path):
    """
    Erstellt eine kleine Beispiel-CSV-Datei mit der Spalte 'token_ids'.
    Die 'token_ids'-Spalte enthält in diesem Fall als String gespeicherte Listen.
    """
    data = {
        "token_ids": [
            "[2, 45, 7, 13, 3]",   # Beispiel: <BOS>, 45, 7, 13, <EOS>
            "[2, 10, 22, 3]",      # Beispiel: <BOS>, 10, 22, <EOS>
            "[2, 99, 3]",         # Beispiel: <BOS>, 99, <EOS>
        ]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)
    return file_path

def test_story_dataset_loading(sample_csv):
    """
    Testet, ob das StoryDataset:
      - Die CSV-Datei korrekt einliest.
      - Den 'token_ids'-String in eine Python-Liste umwandelt.
      - Input- und Target-Sequenzen korrekt erstellt.
    """
    dataset = StoryDataset(csv_file=str(sample_csv))
    
    # Erwartete Länge: 3 Zeilen aus der CSV
    assert len(dataset) == 3, "Das Dataset sollte 3 Einträge enthalten."
    
    # Prüfe den ersten Eintrag
    input_seq, target_seq = dataset[0]
    # Für die Liste [2, 45, 7, 13, 3]:
    # Erwartet: Input = [2, 45, 7, 13], Target = [45, 7, 13, 3]
    assert input_seq.tolist() == [2, 45, 7, 13], "Input-Sequenz stimmt nicht."
    assert target_seq.tolist() == [45, 7, 13, 3], "Target-Sequenz stimmt nicht."

def test_collate_fn_padding():
    """
    Testet die Funktion 'collate_fn', die verschiedene Sequenzlängen
    auf die gleiche Länge auffüllt (Padding).
    """
    # Beispiel-Batch: Zwei Tupel (Input, Target)
    batch = [
        (torch.tensor([2, 45, 7]), torch.tensor([45, 7, 13])),
        (torch.tensor([2, 10]), torch.tensor([10, 3]))
    ]
    # Wir verwenden pad_id = 0 für das Padding
    inp_padded, tgt_padded = collate_fn(batch, pad_id=0)
    
    # Erwartete Form: 2 Zeilen, die Länge entspricht der längsten Sequenz (hier: 3)
    assert inp_padded.shape == (2, 3), "Die Eingabesequenz sollte die Form (2,3) haben."
    assert tgt_padded.shape == (2, 3), "Die Zielsequenz sollte die Form (2,3) haben."
    
    # Überprüfe, ob das Padding korrekt erfolgt:
    # Die zweite Sequenz soll auf [2, 10, 0] gepaddet werden.
    assert inp_padded[1, 2].item() == 0, "Padding-Wert (0) fehlt in der zweiten Sequenz."

def test_create_dataloader(sample_csv):
    """
    Testet, ob create_dataloader einen funktionsfähigen DataLoader erzeugt.
    """
    # Erzeuge einen DataLoader mit Batch-Größe 2
    loader = create_dataloader(csv_file=str(sample_csv), batch_size=2, shuffle=False, pad_id=0)
    
    # Hole einen Batch
    for batch_input, batch_target in loader:
        # batch_input und batch_target sollten Tensoren mit Batch-Dimension 2 haben
        assert batch_input.shape[0] == 2, "Batch-Größe sollte 2 sein."
        assert batch_target.shape[0] == 2, "Batch-Größe sollte 2 sein."
        # Länge der Sequenzen sollte gleich sein (aufgrund von Padding)
        assert batch_input.shape[1] == batch_target.shape[1], "Input und Target sollten dieselbe Sequenzlänge haben."
        break  # Nur den ersten Batch testen

def test_load_vocab(tmp_path):
    """
    Testet, ob load_vocab das Vokabular aus einer JSON-Datei korrekt lädt.
    """
    import json
    # Erstelle ein kleines Vokabular als Beispiel
    vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "es": 4, "war": 5, "einmal": 6}
    vocab_file = tmp_path / "vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    
    loaded_vocab = load_vocab(str(vocab_file))
    assert loaded_vocab["<PAD>"] == 0, "Der Wert für <PAD> sollte 0 sein."
    assert loaded_vocab["einmal"] == 6, "Der Wert für 'einmal' sollte 6 sein."
    assert len(loaded_vocab) == 7, "Das Vokabular sollte 7 Einträge enthalten."
