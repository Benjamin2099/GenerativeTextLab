"""
train.py
--------
Haupt-Trainingsskript (CLI) für das StoryWeaver-Projekt.
Trainiert ein LSTM-Sprachmodell auf Märchen-/Fantasy-Texten.

Beispielaufruf:
    python train.py --train_csv ../data/processed/train.csv \
                    --val_csv ../data/processed/val.csv \
                    --vocab_json ../data/processed/vocab.json \
                    --batch_size 32 --epochs 5 --lr 0.001
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Eigene Module importieren
from dataset import create_dataloader, load_vocab
from model import LSTMTextModel

def parse_args():
    """
    Liest die Kommandozeilenargumente ein und gibt sie zurück.
    """
    parser = argparse.ArgumentParser(description="Train StoryWeaver LSTM Model.")
    
    # Daten
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Pfad zur Trainings-CSV-Datei (z.B. 'train.csv').")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Pfad zur Validierungs-CSV-Datei (z.B. 'val.csv').")
    parser.add_argument("--vocab_json", type=str, required=True,
                        help="Pfad zur vocab.json (Mapping Token->ID).")
    
    # Hyperparameter
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch-Größe.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Anzahl der Trainings-Epochen.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Lernrate.")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Embedding-Dimension im LSTM.")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Größe der Hidden-States im LSTM.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Anzahl gestackter LSTM-Schichten.")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout (wirkt erst ab num_layers>1).")
    
    # Speichern
    parser.add_argument("--save_dir", type=str, default="../models",
                        help="Zielordner, um Modellcheckpoints abzulegen.")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Alle wieviele Batches wird ein Zwischenstand ausgegeben?")
    
    args = parser.parse_args()
    return args

def train_one_epoch(model, loader, optimizer, criterion, device, log_interval=100):
    """
    Führt einen Trainingsdurchlauf über loader durch.
    """
    model.train()
    total_loss = 0.0
    for batch_idx, (inp, tgt) in enumerate(loader):
        inp, tgt = inp.to(device), tgt.to(device)
        
        # Forward
        logits, _ = model(inp)
        vocab_size = logits.size(-1)
        
        # Shape anpassen für CrossEntropy
        logits = logits.view(-1, vocab_size)   # [Batch*SeqLen, vocab_size]
        tgt    = tgt.view(-1)                  # [Batch*SeqLen]
        
        loss = criterion(logits, tgt)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  [Batch {batch_idx+1}/{len(loader)}] Loss = {avg_loss:.4f}")

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    """
    Evaluierung auf Validation-Loader. Gibt den Durchschnitts-Loss zurück.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            logits, _ = model(inp)
            vocab_size = logits.size(-1)
            logits = logits.view(-1, vocab_size)
            tgt = tgt.view(-1)
            loss = criterion(logits, tgt)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    args = parse_args()
    
    # 1) Gerät festlegen
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Trainiere auf Gerät:", device)
    
    # 2) Vokabular laden
    word2id = load_vocab(args.vocab_json)
    vocab_size = len(word2id)
    print("Vokabulargröße:", vocab_size)
    pad_id = word2id.get("<PAD>", 0)
    
    # 3) DataLoader erzeugen
    train_loader = create_dataloader(
        csv_file=args.train_csv,
        batch_size=args.batch_size,
        shuffle=True,
        pad_id=pad_id
    )
    val_loader = create_dataloader(
        csv_file=args.val_csv,
        batch_size=args.batch_size,
        shuffle=False,
        pad_id=pad_id
    )
    
    # 4) Modell initialisieren
    model = LSTMTextModel(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pad_idx=pad_id,
        dropout=args.dropout
    ).to(device)
    
    # 5) Loss und Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 6) Trainingsloop
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoche {epoch}/{args.epochs} ===")
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, 
            criterion, device, 
            log_interval=args.log_interval
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
        
        # Speichern, wenn neue Bestmarke
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.save_dir, f"lstm_epoch{epoch}_valloss{val_loss:.4f}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"** Modell mit neuem Bestloss gespeichert unter: {model_path}")

if __name__ == "__main__":
    main()
