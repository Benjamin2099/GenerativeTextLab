"""
tests/test_integration.py
-------------------------
Integrationstests für das StoryWeaver-Projekt.
Diese Tests führen einen kleinen End-to-End-Durchlauf durch:
  - Laden eines kleinen Datensatzes (CSV mit token_ids)
  - Training eines LSTM-Modells über ein paar Epochen
  - Generierung eines Textes (Inference)
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from src.dataset import create_dataloader, load_vocab
from src.model import LSTMTextModel

import pytest

# Fixture zum Erzeugen einer kleinen Beispieldatei für den Datensatz
@pytest.fixture
def small_csv(tmp_path):
    """
    Erstellt eine temporäre CSV-Datei mit Beispieldaten.
    Die CSV enthält eine Spalte 'token_ids' mit Beispielen.
    """
    data = {
        "token_ids": [
            "[2, 10, 20, 30, 3]",  # Beispiel: <BOS>, 10, 20, 30, <EOS>
            "[2, 15, 25, 35, 3]"
        ]
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "small_train.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

# Fixture zum Erzeugen eines kleinen Vokabulars als JSON-Datei
@pytest.fixture
def small_vocab(tmp_path):
    """
    Erstellt ein kleines Vokabular und speichert es als vocab.json.
    """
    vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "10": 4, "20": 5, "30": 6, "15": 7, "25": 8, "35": 9}
    vocab_path = tmp_path / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    return str(vocab_path)

def test_integration_training_and_generation(small_csv, small_vocab):
    """
    Führt einen kleinen End-to-End-Durchlauf durch:
      - Laden eines Mini-Datensatzes
      - Training eines LSTM-Modells für wenige Epochen
      - Generierung eines Textes basierend auf einem Prompt
    Überprüft, ob der Trainingsloss sinkt und ob die Generierung nicht leer ist.
    """
    # 1) Vokabular laden
    word2id = load_vocab(small_vocab)
    vocab_size = len(word2id)
    id2word = {v: k for k, v in word2id.items()}
    pad_id = word2id.get("<PAD>", 0)
    
    # 2) DataLoader erstellen (wir nutzen den gleichen Datensatz für Training und Validierung)
    train_loader = create_dataloader(csv_file=small_csv, batch_size=2, shuffle=True, pad_id=pad_id)
    val_loader = create_dataloader(csv_file=small_csv, batch_size=2, shuffle=False, pad_id=pad_id)
    
    # 3) Modell initialisieren
    model = LSTMTextModel(
        vocab_size=vocab_size,
        embed_dim=16,
        hidden_dim=32,
        num_layers=1,
        pad_idx=pad_id,
        dropout=0.0
    )
    
    # 4) Loss und Optimizer definieren
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # 5) Training (simpler Mini-Trainingsloop über 3 Epochen)
    def train_one_epoch(model, loader):
        model.train()
        total_loss = 0.0
        for inp, tgt in loader:
            logits, _ = model(inp)
            logits = logits.view(-1, vocab_size)
            tgt = tgt.view(-1)
            loss = criterion(logits, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    initial_loss = train_one_epoch(model, train_loader)
    final_loss = train_one_epoch(model, train_loader)
    
    # Test: Der Loss sollte sinken (bei solch einem einfachen Beispiel oft gut erkennbar)
    assert final_loss < initial_loss, f"Der Loss sollte sinken. Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    
    # 6) Inferenz: Textgenerierung testen
    def generate_text(model, prompt_ids, max_length=10):
        model.eval()
        tokens = prompt_ids[:]
        inp = torch.tensor([tokens], dtype=torch.long)
        hidden = None
        for _ in range(max_length):
            logits, hidden = model(inp, hidden)
            last_logits = logits[0, -1, :]
            next_id = torch.argmax(last_logits).item()  # Greedy-Decoding
            tokens.append(next_id)
            inp = torch.tensor([tokens], dtype=torch.long)
            if next_id == word2id.get("<EOS>", 3):
                break
        return tokens
    
    # Starte mit <BOS> als Prompt
    prompt_ids = [word2id.get("<BOS>", 2)]
    generated_ids = generate_text(model, prompt_ids, max_length=5)
    generated_tokens = [id2word.get(i, "<UNK>") for i in generated_ids]
    
    # Test: Es sollte mindestens ein zusätzliches Token generiert werden.
    assert len(generated_ids) > len(prompt_ids), "Die Generierung sollte zusätzliche Tokens produzieren."
    print("Generierte Sequenz:", generated_tokens)
