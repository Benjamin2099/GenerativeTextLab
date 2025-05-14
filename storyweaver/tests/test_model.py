"""
tests/test_model.py
-------------------
Unit-Tests für das LSTM-/Transformer-Modell aus src/model.py.
Diese Tests überprüfen:
  - Ob der Forward-Pass des LSTM-Modells die erwarteten Ausgaben (logits) liefert.
  - Ob die Shape der logits und der Hidden-States den Erwartungen entspricht.
  - Ob die Funktion init_hidden() korrekt initialisiert.
"""

import torch
import torch.nn as nn
import pytest
from src.model import LSTMTextModel

# Fixture für ein kleines LSTM-Modell, um Tests durchzuführen.
@pytest.fixture
def small_lstm_model():
    """
    Erzeugt ein kleines LSTM-Modell mit festgelegten Parametern.
    """
    vocab_size = 50   # kleines Demo-Vokabular
    embed_dim = 16
    hidden_dim = 32
    num_layers = 2
    pad_idx = 0
    model = LSTMTextModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pad_idx=pad_idx,
        dropout=0.0  # Für Tests ist Dropout nicht erforderlich
    )
    return model

def test_lstm_forward_shape(small_lstm_model):
    """
    Testet, ob das LSTM-Modell beim Forward-Pass:
      - logits mit der Form [Batch, SeqLen, vocab_size] zurückgibt.
      - den Hidden-State als Tupel (h_n, c_n) in der erwarteten Form liefert.
    """
    batch_size = 4
    seq_len = 5
    vocab_size = 50  # Sollte mit dem Modell übereinstimmen

    # Erzeuge Dummy-Daten als Eingabe
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    
    # Forward-Pass ohne explizite Initialisierung des Hidden-States (wird intern auf 0 gesetzt)
    logits, hidden = small_lstm_model(x)
    
    # Überprüfe, ob die logits-Shape [Batch, SeqLen, vocab_size] entspricht
    assert logits.shape == (batch_size, seq_len, vocab_size), \
           f"Erwartet: {(batch_size, seq_len, vocab_size)}, erhalten: {logits.shape}"
    
    # Überprüfe das Hidden-State-Tupel
    assert isinstance(hidden, tuple) and len(hidden) == 2, "hidden sollte ein Tupel (h_n, c_n) sein."
    h_n, c_n = hidden
    num_layers = small_lstm_model.lstm.num_layers
    hidden_dim = small_lstm_model.lstm.hidden_size

    # h_n und c_n sollten die Form [num_layers, batch_size, hidden_dim] haben
    assert h_n.shape == (num_layers, batch_size, hidden_dim), \
           f"Erwartete h_n-Shape: {(num_layers, batch_size, hidden_dim)}, erhalten: {h_n.shape}"
    assert c_n.shape == (num_layers, batch_size, hidden_dim), \
           f"Erwartete c_n-Shape: {(num_layers, batch_size, hidden_dim)}, erhalten: {c_n.shape}"

def test_lstm_init_hidden(small_lstm_model):
    """
    Testet, ob init_hidden() korrekte Zero-Tensoren für den Hidden- und Cell-State erzeugt.
    """
    batch_size = 3
    device = "cpu"
    h_0, c_0 = small_lstm_model.init_hidden(batch_size, device=device)
    
    num_layers = small_lstm_model.lstm.num_layers
    hidden_dim = small_lstm_model.lstm.hidden_size

    # Überprüfe die Shape der h_0- und c_0-Tensoren
    assert h_0.shape == (num_layers, batch_size, hidden_dim), \
           f"h_0 sollte die Form {(num_layers, batch_size, hidden_dim)} haben."
    assert c_0.shape == (num_layers, batch_size, hidden_dim), \
           f"c_0 sollte die Form {(num_layers, batch_size, hidden_dim)} haben."
    
    # Alle Werte in h_0 und c_0 sollten 0 sein
    assert torch.all(h_0 == 0), "h_0 sollte ausschließlich Nullen enthalten."
    assert torch.all(c_0 == 0), "c_0 sollte ausschließlich Nullen enthalten."

def test_lstm_overfit_small_batch(small_lstm_model):
    """
    Simuliert ein kleines Training auf einem Mini-Datensatz, um zu testen,
    ob das Modell in der Lage ist, den Loss zu reduzieren.
    Hier lernen wir, das Modell soll Input in sich selbst abbilden (Identitätsfunktion).
    """
    vocab_size = 50
    batch_size = 2
    seq_len = 4

    # Erzeuge einen kleinen Dummy-Datensatz: Input-Daten als zufällige Token-IDs.
    inputs = torch.randint(low=0, high=vocab_size, size=(batch_size * 2, seq_len))
    targets = inputs.clone()  # Identitätsabbildung: Das Modell soll lernen, das Gleiche zurückzugeben.

    # Simuliere einen DataLoader, der zwei Batches liefert.
    def get_batches():
        for i in range(0, batch_size * 2, batch_size):
            yield (inputs[i:i+batch_size], targets[i:i+batch_size])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(small_lstm_model.parameters(), lr=0.1)

    initial_losses = []
    final_losses = []

    # Trainiere über 3 Epochen
    for epoch in range(3):
        for inp, tgt in get_batches():
            logits, _ = small_lstm_model(inp)
            logits = logits.view(-1, vocab_size)
            tgt = tgt.view(-1)
            loss = criterion(logits, tgt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch == 0:
                initial_losses.append(loss.item())
            elif epoch == 2:
                final_losses.append(loss.item())

    avg_initial = sum(initial_losses) / len(initial_losses)
    avg_final = sum(final_losses) / len(final_losses)
    
    # Wir erwarten, dass der durchschnittliche Loss nach 3 Epochen sinkt.
    assert avg_final < avg_initial, f"Loss sollte sinken (Initial: {avg_initial:.4f}, Final: {avg_final:.4f})."
