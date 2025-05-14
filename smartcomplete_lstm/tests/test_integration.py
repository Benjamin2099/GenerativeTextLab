"""
test_integration.py
-------------------
Integrationstests, die sicherstellen, dass Dataset, Modell und ggf. Trainings-
oder Generierungs-Workflow zusammen funktionieren.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.dataset import create_dataloader, load_vocab
from src.model import LSTMTextModel

@pytest.fixture
def vocab_dict(tmp_path):
    """
    Erstellt ein kleines Vokabular und speichert es als vocab.json.
    Gibt Pfad zurück.
    """
    import json
    vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "hello": 4, "world": 5}
    vocab_path = tmp_path / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    return vocab_path

@pytest.fixture
def small_csv(tmp_path):
    """
    Erstellt eine CSV-Datei für Trainingsdaten mit 2 Beispielsätzen.
    """
    import pandas as pd
    df = pd.DataFrame({
        "token_ids": [
            "[2, 4, 5, 3]",   # <BOS> hello world <EOS>
            "[2, 5, 5, 5, 3]" # <BOS> world world world <EOS>
        ]
    })
    csv_path = tmp_path / "train_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def test_full_training_integration(small_csv, vocab_dict, tmp_path):
    """
    Testet, ob ein kleiner Trainings-Durchlauf mit Dataset und LSTM-Modell
    ohne Fehler durchläuft und ob der Loss sinkt.
    """
    # 1) Vokabular laden
    word2id = load_vocab(vocab_dict)
    vocab_size = len(word2id)

    # 2) DataLoader erstellen
    loader = create_dataloader(
        csv_file=small_csv,
        batch_size=2, 
        shuffle=False,
        pad_id=word2id["<PAD>"]
    )

    # 3) LSTM-Modell initialisieren
    model = LSTMTextModel(
        vocab_size=vocab_size,
        embed_dim=16,
        hidden_dim=32,
        num_layers=1,
        pad_idx=word2id["<PAD>"]
    )

    # 4) Loss & Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=word2id["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 5) Mini-Trainingsloop
    initial_loss, final_loss = None, None
    EPOCHS = 3

    for epoch in range(EPOCHS):
        total_loss = 0.0
        model.train()
        for batch_input, batch_target in loader:
            logits, _ = model(batch_input)
            # logits: [Batch, SeqLen, vocab_size]
            vocab_s = logits.size(-1)
            logits = logits.reshape(-1, vocab_s)
            batch_target = batch_target.view(-1)

            loss = criterion(logits, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        if epoch == 0:
            initial_loss = avg_loss
        if epoch == EPOCHS - 1:
            final_loss = avg_loss
    
    # Prüfe, ob Loss sich verringert hat
    assert final_loss is not None and initial_loss is not None, "Loss-Werte fehlen."
    assert final_loss < initial_loss, (
        f"Erwartet, dass der Loss nach {EPOCHS} Epochen sinkt. "
        f"Initial={initial_loss:.4f}, Final={final_loss:.4f}"
    )

def test_generation_integration(small_csv, vocab_dict, tmp_path):
    """
    Testet, ob nach einem kurzen Training eine Textgenerierung (Inference)
    möglich ist, ohne Fehler.
    """
    from src.generate import generate_text

    # 1) Vokabular laden
    word2id = load_vocab(vocab_dict)
    id2word = {v:k for k,v in word2id.items()}
    vocab_size = len(word2id)

    # 2) DataLoader (Train + Quick Train)
    loader = create_dataloader(small_csv, batch_size=2, shuffle=False, pad_id=word2id["<PAD>"])

    model = LSTMTextModel(
        vocab_size=vocab_size,
        embed_dim=16,
        hidden_dim=32,
        num_layers=1,
        pad_idx=word2id["<PAD>"]
    )

    criterion = nn.CrossEntropyLoss(ignore_index=word2id["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Kurz trainieren (1 Epoche)
    for batch_input, batch_target in loader:
        logits, _ = model(batch_input)
        vocab_s = logits.size(-1)
        logits = logits.view(-1, vocab_s)
        batch_target = batch_target.view(-1)
        loss = criterion(logits, batch_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 3) Inference
    # Wir simulieren, dass <BOS> im Prompt steht. 
    # generate_text: tokens=[<BOS>, ...]
    generated_ids = generate_text(
        model=model,
        word2id=word2id,
        id2word=id2word,
        prompt="<BOS>",
        max_length=5,
        top_k=1,
        temperature=1.0,
        device="cpu"
    )
    
    # Nur Test, ob wir keine Exception bekommen und eine Ausgabe vorliegt
    assert len(generated_ids) > 0, "Es sollte mindestens ein generiertes Token geben."
    print("Generated sequence:", generated_ids)
