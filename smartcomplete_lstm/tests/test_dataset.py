"""
test_dataset.py
---------------
Unit-Tests für das Dataset und DataLoader-Verhalten 
aus src/dataset.py.
"""

import pytest
import torch
import os
import pandas as pd

from src.dataset import TextDataset, create_dataloader, collate_fn, load_vocab

@pytest.fixture
def mock_csv(tmp_path):
    """
    Erstellt eine temporäre CSV-Datei mit Beispieldaten und gibt den Pfad zurück.
    """
    df = pd.DataFrame({
        "token_ids": [
            "[2, 45, 7, 13]",    # 4 Tokens
            "[2, 10, 10, 10, 3]",# 5 Tokens
            "[]",                # Leere Liste
            "[2]"                # Nur 1 Token
        ]
    })
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return file_path

def test_text_dataset_basic(mock_csv):
    """
    Prüft, ob das TextDataset grundlegende Funktionen erfüllt:
     - Korrekte Länge
     - Umwandlung von Strings zu Python-Listen
     - Ausgabe (input_seq, target_seq) mit erwarteter Größe.
    """
    dataset = TextDataset(csv_file=mock_csv)

    # Insgesamt haben wir 4 Zeilen in der CSV
    assert len(dataset) == 4, "Dataset sollte 4 Reihen enthalten"

    # Prüfe das erste Sample
    input_seq, target_seq = dataset[0]
    # token_ids laut CSV: [2, 45, 7, 13]
    # => input_seq: [2, 45, 7], target_seq: [45, 7, 13]
    assert input_seq.tolist() == [2, 45, 7], "Input-Seq sollte [2, 45, 7] sein"
    assert target_seq.tolist() == [45, 7, 13], "Target-Seq sollte [45, 7, 13] sein"

def test_text_dataset_short_sequences(mock_csv):
    """
    Prüft Verhalten, wenn token_ids sehr kurz oder leer sind.
    """
    dataset = TextDataset(csv_file=mock_csv)
    # Zeile 2 in CSV: "[]"
    inp, tgt = dataset[2]
    # Da wir in dataset.py (Example) den Fall abfangen mit:
    # tokens_list = [0,0] wenn len < 2
    # => input_seq=[0], target_seq=[0]
    assert inp.tolist() == [0], "Bei leerer Liste sollte Input 0 enthalten."
    assert tgt.tolist() == [0], "Bei leerer Liste sollte Target 0 enthalten."

    # Zeile 3 in CSV: "[2]" => 1 Token
    inp2, tgt2 = dataset[3]
    # => [0,0] oder Ähnliches je nach Implementation
    assert len(inp2) == 1 and len(tgt2) == 1, "Sollte automatisch angepasste Längen haben."

def test_collate_fn():
    """
    Prüft, ob collate_fn korrektes Padding durchführt.
    """
    batch = [
        (torch.tensor([2, 45, 7]), torch.tensor([45, 7, 13])),
        (torch.tensor([2, 10]), torch.tensor([10, 3]))
    ]
    input_padded, target_padded = collate_fn(batch, pad_id=0)
    
    # input_padded sollte Form (2, max_seq_len)
    # max_seq_len = 3 (längste Input-Sequenz)
    assert input_padded.shape == (2, 3)
    # 1. Zeile -> [2, 45, 7]
    # 2. Zeile -> [2, 10, 0] (Padding)
    assert input_padded[1, 2].item() == 0, "Drittes Token für zweite Zeile sollte 0 (Pad) sein."

    # target_padded -> (2, max_seq_len=3)
    # 1. Zeile: [45, 7, 13]
    # 2. Zeile: [10, 3, 0]
    assert target_padded.shape == (2, 3)
    assert target_padded[1, 2].item() == 0, "Letztes Token zweiter Zeile => Padding=0."

def test_create_dataloader(mock_csv):
    """
    Testet, ob create_dataloader einen funktionsfähigen DataLoader zurückgibt.
    """
    from src.dataset import create_dataloader
    loader = create_dataloader(csv_file=mock_csv, batch_size=2, shuffle=False, pad_id=0)
    
    data_iter = iter(loader)
    input_batch, target_batch = next(data_iter)
    
    # Wir haben 2 Samples pro Batch -> shape [2, max_seq_len]
    assert input_batch.shape[0] == 2, "Batchsize=2 in create_dataloader."
    assert target_batch.shape[0] == 2, "Batchsize=2 in create_dataloader."

def test_load_vocab(tmp_path):
    """
    Prüft, ob load_vocab die JSON-Datei korrekt lädt.
    """
    vocab_dict = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2}
    vocab_json = tmp_path / "vocab.json"

    import json
    with open(vocab_json, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f)

    loaded = load_vocab(vocab_json)
    assert loaded["<PAD>"] == 0
    assert loaded["<BOS>"] == 2
    assert len(loaded) == 3, "3 Token im Vokabular erwartet."
