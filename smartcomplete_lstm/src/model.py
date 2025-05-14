"""
model.py
--------
Enthält die LSTM-Modellklasse (PyTorch), die für das 
SmartComplete-Sprachmodell genutzt wird.
"""

import torch
import torch.nn as nn

class LSTMTextModel(nn.Module):
    """
    Ein einfaches LSTM-basiertes Sprachmodell für Textgenerierung.
    
    Aufbau:
      1) Embedding: Wandelt Token-IDs in hochdimensionale Vektoren
      2) LSTM: RNN, das sequentiell die Embeddings verarbeitet
      3) FC-Layer: Gibt pro Zeitschritt Wahrscheinlichkeiten für das nächste Wort (Vokabulargröße)
    
    Beispielverwendung:
      model = LSTMTextModel(vocab_size=10000, embed_dim=128, hidden_dim=256, num_layers=2)
      logits, hidden = model(input_tensor) 
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, pad_idx=0, dropout=0.0):
        """
        Args:
            vocab_size  (int): Größe des Vokabulars (inkl. Sondertokens)
            embed_dim   (int): Dimension der Embedding-Vektoren
            hidden_dim  (int): Größe des verborgenen Zustands (Hidden) in den LSTM-Schichten
            num_layers  (int): Anzahl der LSTM-Layer (Stacking)
            pad_idx     (int): Index des <PAD>-Tokens (wichtig, um Padding im Embedding zu ignorieren)
            dropout     (float): Dropout-Wahrscheinlichkeit zwischen den LSTM-Schichten
        """
        super(LSTMTextModel, self).__init__()
        
        # 1) Embedding-Layer: Wandelt Token-IDs in Vektoren
        # pad_idx stellt sicher, dass keine Gradientenupdates für <PAD> berechnet werden
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        # 2) LSTM-Schichten
        # batch_first=True → Eingabeform [Batch, SeqLen, EmbeddingDim]
        # dropout - nur zwischen den Schichten, sofern num_layers > 1
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 3) FC-Layer: Von Hidden-Dimension auf Vokabulargröße
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Optional: Du kannst hier z. B. init_weights() aufrufen,
        # um Gewichte speziell zu initialisieren, falls nötig.
    
    def forward(self, x, hidden=None):
        """
        Führt einen Vorwärtsdurchlauf durch.
        
        Args:
            x      (Tensor): Eingabesequenz [Batch, SeqLen] (Token-IDs)
            hidden (Tuple): (h_0, c_0) - Initialer verborgener und Zellzustand, 
                            wenn None → wird von PyTorch intern auf 0 initiiert.
        
        Returns:
            logits (Tensor): Ausgabe [Batch, SeqLen, vocab_size] 
                             mit den Wahrscheinlichkeiten pro Zeitschritt.
            hidden (Tuple): (h_n, c_n) - Letzter verborgener und Zellzustand.
        """
        # Embedding
        emb = self.embedding(x)  # [Batch, SeqLen, embed_dim]
        
        # LSTM
        out, hidden = self.lstm(emb, hidden)  # out: [Batch, SeqLen, hidden_dim]
        
        # FC-Layer
        logits = self.fc(out)  # [Batch, SeqLen, vocab_size]
        
        return logits, hidden

    def init_hidden(self, batch_size, device='cpu'):
        """
        Erstellt einen Null-Zustand für hidden und cell, passend zu num_layers und hidden_dim.
        
        Returns:
            Tuple (h_0, c_0), beide von Shape:
               [num_layers, batch_size, hidden_dim]
        """
        # LSTM-Parameter aus self.lstm
        num_layers = self.lstm.num_layers
        hidden_dim = self.lstm.hidden_size
        
        # h_0, c_0 auf 0 initialisieren
        h_0 = torch.zeros(num_layers, batch_size, hidden_dim, device=device)
        c_0 = torch.zeros(num_layers, batch_size, hidden_dim, device=device)
        return (h_0, c_0)


if __name__ == "__main__":
    # Kleiner Test, wenn man das Skript direkt ausführt
    vocab_size_test = 500
    model = LSTMTextModel(vocab_size=vocab_size_test, embed_dim=32, hidden_dim=64, num_layers=2, pad_idx=0, dropout=0.1)
    
    batch_size_test = 4
    seq_len_test = 10
    
    # Dummy-Eingabe: [Batch, SeqLen] - Zufällige Token-IDs
    x_test = torch.randint(low=0, high=vocab_size_test, size=(batch_size_test, seq_len_test))
    print("x_test shape:", x_test.shape)
    
    # Forward Pass
    logits, hidden = model(x_test)
    print("logits shape:", logits.shape)    # erwarteter Output: [4, 10, 500]
    print("Hidden states shapes:", [h.shape for h in hidden]) 
    print("Test Forward Pass erfolgreich!")
