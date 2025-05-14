"""
train.py
--------
Haupt-Trainingsskript (CLI) für ChatPal – den einfachen Chatbot.
Dieses Skript trainiert ein LSTM-Modell, um aus User-Eingaben passende Bot-Antworten zu generieren.

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

# Eigene Module aus dem src-Verzeichnis importieren
from dataset import create_dataloader, load_vocab
from model import ChatLSTMModel

def parse_args():
    """
    Liest Kommandozeilen-Argumente ein.
    """
    parser = argparse.ArgumentParser(description="Train ChatPal LSTM Model")
    
    # Pfade zu den CSV-Daten und dem Vokabular
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Pfad zur Trainings-CSV-Datei (z.B. 'train.csv').")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Pfad zur Validierungs-CSV-Datei (z.B. 'val.csv').")
    parser.add_argument("--vocab_json", type=str, required=True,
                        help="Pfad zur vocab.json-Datei (Mapping Token->ID).")
    
    # Hyperparameter
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch-Größe.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Anzahl der Trainings-Epochen.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Lernrate.")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Dimension der Embeddings.")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Größe der Hidden-States im LSTM.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Anzahl der LSTM-Schichten.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout zwischen LSTM-Schichten (wirkt erst ab num_layers>1).")
    
    # Speicherort für das Modell
    parser.add_argument("--save_dir", type=str, default="../models",
                        help="Ordner, in dem Modell-Checkpoints gespeichert werden.")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Alle wieviele Batches wird ein Zwischenstand ausgegeben.")
    
    args = parser.parse_args()
    return args

def train_one_epoch(model, loader, optimizer, criterion, device, log_interval=50):
    """
    Trainiert das Modell für eine Epoche und gibt den durchschnittlichen Loss zurück.
    """
    model.train()
    total_loss = 0.0
    for batch_idx, (user_batch, bot_batch) in enumerate(loader):
        user_batch, bot_batch = user_batch.to(device), bot_batch.to(device)
        
        # Forward-Pass: Das Modell nimmt die User-Sequenz als Input,
        # und gibt Logits zurück, die mit der Bot-Sequenz verglichen werden.
        logits, _ = model(user_batch)
        vocab_size = logits.size(-1)
        # Reshape, damit CrossEntropyLoss angewendet werden kann.
        logits = logits.view(-1, vocab_size)   # [Batch * SeqLen, vocab_size]
        bot_flat = bot_batch.view(-1)            # [Batch * SeqLen]
        
        loss = criterion(logits, bot_flat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  [Batch {batch_idx+1}/{len(loader)}] Loss: {avg_loss:.4f}")
    
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    """
    Berechnet den durchschnittlichen Loss auf dem Validierungs-Datensatz.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for user_batch, bot_batch in loader:
            user_batch, bot_batch = user_batch.to(device), bot_batch.to(device)
            logits, _ = model(user_batch)
            vocab_size = logits.size(-1)
            logits = logits.view(-1, vocab_size)
            bot_flat = bot_batch.view(-1)
            loss = criterion(logits, bot_flat)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    args = parse_args()
    
    # 1) Geräteauswahl: GPU oder CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Trainiere auf Gerät:", device)
    
    # 2) Vokabular laden
    word2id = load_vocab(args.vocab_json)
    vocab_size = len(word2id)
    pad_id = word2id.get("<PAD>", 0)
    print("Vokabulargröße:", vocab_size)
    
    # 3) DataLoader erstellen
    train_loader = create_dataloader(
        csv_file=args.train_csv,
        batch_size=args.batch_size,
        shuffle=True,
        pad_id=pad_id,
        user_col="user_ids",
        bot_col="bot_ids"
    )
    val_loader = create_dataloader(
        csv_file=args.val_csv,
        batch_size=args.batch_size,
        shuffle=False,
        pad_id=pad_id,
        user_col="user_ids",
        bot_col="bot_ids"
    )
    
    # 4) Modell initialisieren
    model = ChatLSTMModel(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pad_idx=pad_id,
        dropout=args.dropout
    ).to(device)
    
    # 5) Loss-Funktion und Optimizer definieren
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 6) Speicherordner erstellen, falls nicht vorhanden
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    
    # 7) Trainings- und Validierungsschleife
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoche {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, log_interval=args.log_interval)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Speichere das Modell, wenn der Validierungs-Loss verbessert wurde.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.save_dir, f"chatpal_epoch{epoch}_valloss{val_loss:.4f}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"** Neues bestes Modell gespeichert: {model_path}")
    
    print("\nTraining abgeschlossen.")

if __name__ == "__main__":
    main()
