"""
train.py
--------
Haupt-Trainingsskript (CLI) für das LSTM-Sprachmodell im SmartComplete-Projekt.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Eigene Module importieren
from dataset import create_dataloader, load_vocab
from model import LSTMTextModel

def parse_args():
    """
    Liest die Kommandozeilen-Argumente ein und gibt sie zurück.
    Beispiele:
      python train.py --train_csv ../data/processed/train.csv --val_csv ../data/processed/val.csv
                      --vocab_json ../data/processed/vocab.json
                      --batch_size 32 --epochs 5 --lr 0.001
    """
    parser = argparse.ArgumentParser(description="Train LSTM language model (SmartComplete).")
    
    # Daten & Vokabular
    parser.add_argument("--train_csv", type=str, required=True, 
                        help="Pfad zur Trainings-CSV-Datei (enthält token_ids).")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Pfad zur Validierungs-CSV-Datei (enthält token_ids).")
    parser.add_argument("--vocab_json", type=str, required=True,
                        help="Pfad zur vocab.json-Datei (Mapping token->ID).")
    
    # Hyperparameter
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch-Größe.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Anzahl der Trainings-Epochen.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Lernrate (Learning Rate) des Optimizers.")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Dimension der Embeddings.")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden-Dimension in den LSTM-Schichten.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Anzahl gestackter LSTM-Layer.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout zwischen LSTM-Schichten (wirkt erst ab num_layers > 1).")
    
    # Sonstiges
    parser.add_argument("--save_dir", type=str, default="../models",
                        help="Ordner, in dem das trainierte Modell gespeichert wird.")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Nach wie vielen Batches ein Log ausgegeben wird.")
    
    args = parser.parse_args()
    return args

def train_one_epoch(model, loader, optimizer, criterion, device, log_interval=100):
    """Eine Epoche trainieren und den durchschnittlichen Loss zurückgeben."""
    model.train()
    total_loss = 0.0
    for batch_idx, (batch_input, batch_target) in enumerate(loader):
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        
        # Forward
        logits, _ = model(batch_input)  # logits: [Batch, SeqLen, vocab_size]
        
        # Reshape für CrossEntropy
        vocab_size = logits.size(-1)
        logits = logits.view(-1, vocab_size)      # -> [Batch*SeqLen, vocab_size]
        batch_target = batch_target.view(-1)      # -> [Batch*SeqLen]
        
        loss = criterion(logits, batch_target)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  [Batch {batch_idx+1}/{len(loader)}] Durchschnittlicher Loss: {avg_loss:.4f}")
    
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    """Evaluierung auf dem Validierungs-Dataloader. Gibt Loss zurück."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_input, batch_target in loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            logits, _ = model(batch_input)
            vocab_size = logits.size(-1)
            logits = logits.view(-1, vocab_size)
            batch_target = batch_target.view(-1)
            
            loss = criterion(logits, batch_target)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    args = parse_args()
    
    # 1) Gerät festlegen (GPU oder CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training auf Gerät:", device)
    
    # 2) Vokabular laden
    word2id = load_vocab(args.vocab_json)
    vocab_size = len(word2id)
    print("Vokabulargröße:", vocab_size)
    
    # 3) DataLoader erstellen
    train_loader = create_dataloader(
        csv_file=args.train_csv,
        batch_size=args.batch_size,
        shuffle=True, 
        pad_id=word2id.get("<PAD>", 0)  # Standard: <PAD>=0
    )
    val_loader = create_dataloader(
        csv_file=args.val_csv,
        batch_size=args.batch_size,
        shuffle=False,
        pad_id=word2id.get("<PAD>", 0)
    )
    
    # 4) Modell definieren
    model = LSTMTextModel(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pad_idx=word2id.get("<PAD>", 0),
        dropout=args.dropout
    ).to(device)
    
    # 5) Loss & Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=word2id.get("<PAD>", 0))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 6) Trainingsschleife
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoche {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, log_interval=args.log_interval)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Bestes Modell speichern
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.save_dir, f"lstm_epoch{epoch}_valloss{val_loss:.4f}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"** Modell gespeichert unter: {model_path}")
    
    print("\nTraining abgeschlossen.")
    print(f"Bestes Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
