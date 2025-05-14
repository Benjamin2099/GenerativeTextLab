"""
test_model.py
-------------
Enthält Unit-Tests für das LSTM-Modell aus src/model.py.
Prüft vor allem Forward-Pass, Output-Shapes und Hidden-States.
"""

import pytest
import torch
import torch.nn as nn

from src.model import LSTMTextModel

@pytest.fixture
def small_lstm_model():
    """
    Stellt ein kleines LSTM-Modell für Tests bereit.
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
        dropout=0.0
    )
    return model

def test_lstm_forward_shape(small_lstm_model):
    """
    Testet, ob das LSTM-Modell bei einem Forward-Pass
    die erwarteten Formen für logits und hidden state zurückgibt.
    """
    batch_size = 4
    seq_len = 5
    vocab_size = 50  # muss mit dem in small_lstm_model übereinstimmen

    # Dummy-Eingabe: [Batch, SeqLen]
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    
    # Forward
    logits, hidden = small_lstm_model(x)
    # logits soll [batch_size, seq_len, vocab_size] haben
    assert logits.shape == (batch_size, seq_len, vocab_size), (
        f"Erwartete Logits-Form: {(batch_size, seq_len, vocab_size)}, "
        f"aber erhalten: {logits.shape}"
    )

    # hidden ist ein Tupel (h_n, c_n)
    # Jeweils [num_layers, batch_size, hidden_dim]
    assert isinstance(hidden, tuple) and len(hidden) == 2, "hidden sollte (h_n, c_n) sein"
    h_n, c_n = hidden
    num_layers = small_lstm_model.lstm.num_layers
    hidden_dim = small_lstm_model.lstm.hidden_size

    assert h_n.shape == (num_layers, batch_size, hidden_dim), (
        f"h_n Form erwartet: {(num_layers, batch_size, hidden_dim)}, erhalten: {h_n.shape}"
    )
    assert c_n.shape == (num_layers, batch_size, hidden_dim), (
        f"c_n Form erwartet: {(num_layers, batch_size, hidden_dim)}, erhalten: {c_n.shape}"
    )

def test_lstm_init_hidden(small_lstm_model):
    """
    Testet, ob init_hidden eine korrekte Null-Initialisierung
    für den Hidden- und Cell-State liefert.
    """
    batch_size = 3
    device = "cpu"
    hidden = small_lstm_model.init_hidden(batch_size, device=device)
    
    # hidden => (h_0, c_0)
    assert isinstance(hidden, tuple) and len(hidden) == 2
    h_0, c_0 = hidden
    
    num_layers = small_lstm_model.lstm.num_layers
    hidden_dim = small_lstm_model.lstm.hidden_size
    
    # Formen checken
    assert h_0.shape == (num_layers, batch_size, hidden_dim)
    assert c_0.shape == (num_layers, batch_size, hidden_dim)
    
    # Werte checken (sollten alle 0 sein)
    assert torch.all(h_0 == 0), "h_0 sollte nur Nullwerte enthalten."
    assert torch.all(c_0 == 0), "c_0 sollte nur Nullwerte enthalten."

def test_lstm_overfit_small_batch(small_lstm_model):
    """
    Testet, ob das Modell auf einem winzigen Datensatz 
    (z.B. 2 Batches) in wenigen Schritten spürbar 
    den Loss verringern kann (Mini-Overfit-Test).
    """
    # Künstlicher Datensatz
    vocab_size = 50
    batch_size = 2
    seq_len = 4
    
    # Wir erstellen 2 Batches, je 2 Samples => total 4 Trainingssamples
    input_data = torch.randint(low=0, high=vocab_size, size=(batch_size*2, seq_len))
    target_data = input_data.clone()  # Wir lassen das Modell lernen, x -> x (Id. Mapping)
    
    # Einfache DataLoader-Simulation
    # => 2 Batches: 
    #    Batch1 -> items 0 und 1
    #    Batch2 -> items 2 und 3
    def get_batches():
        for i in range(0, batch_size*2, batch_size):
            yield (input_data[i:i+batch_size], target_data[i:i+batch_size])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(small_lstm_model.parameters(), lr=0.1)
    
    initial_losses = []
    final_losses = []
    
    # Training über 3 Epochen
    for epoch in range(3):
        for inp, tgt in get_batches():
            logits, _ = small_lstm_model(inp)
            # logits: [batch, seq_len, vocab_size]
            vocab_size = logits.size(-1)
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
    
    avg_initial_loss = sum(initial_losses) / len(initial_losses)
    avg_final_loss = sum(final_losses) / len(final_losses)
    
    # Wir erwarten, dass der Loss sich deutlich reduziert
    # (wenn unser LSTM halbwegs lernt) 
    assert avg_final_loss < avg_initial_loss, (
        f"Loss sollte nach 3 Epochen geringer sein; initial={avg_initial_loss:.4f}, "
        f"final={avg_final_loss:.4f}"
    )
