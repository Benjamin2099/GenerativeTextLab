"""
train.py
--------
Haupt-Trainingsskript (CLI) für den Stiltransfer im StyleShift-Projekt.
Dieses Skript trainiert ein Seq2Seq-LSTM-Modell, das moderne Texte in einen 
anderen Stil (z. B. Shakespeare) überträgt.

Beispielaufruf:
    python train.py --train_csv data/processed/train.csv \
                    --val_csv data/processed/val.csv \
                    --vocab_json data/processed/vocab.json \
                    --batch_size 32 --epochs 10 --lr 0.001 \
                    --embed_dim 128 --hidden_dim 256 --num_layers 2 --dropout 0.1 \
                    --teacher_forcing_ratio 0.5 --save_dir models --log_interval 50
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Eigene Module importieren
from dataset import create_dataloader, load_vocab
from model import Encoder, Decoder, Seq2Seq

def parse_args():
    parser = argparse.ArgumentParser(description="Trainiere das StyleShift Seq2Seq-Modell für den Stiltransfer.")
    
    # Pfade
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Pfad zur Trainings-CSV-Datei (z.B. 'train.csv').")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Pfad zur Validierungs-CSV-Datei (z.B. 'val.csv').")
    parser.add_argument("--vocab_json", type=str, required=True,
                        help="Pfad zur vocab.json-Datei (Mapping Token->ID).")
    
    # Hyperparameter
    parser.add_argument("--batch_size", type=int, default=32, help="Batch-Größe.")
    parser.add_argument("--epochs", type=int, default=10, help="Anzahl der Trainings-Epochen.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Lernrate.")
    parser.add_argument("--embed_dim", type=int, default=128, help="Dimension der Embeddings.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Größe der Hidden-States im LSTM.")
    parser.add_argument("--num_layers", type=int, default=2, help="Anzahl der LSTM-Schichten.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout, wirkt ab num_layers>1.")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5,
                        help="Wahrscheinlichkeit, das wahre nächste Token beim Decoder zu verwenden (Teacher Forcing).")
    
    # Speicherparameter
    parser.add_argument("--save_dir", type=str, default="models",
                        help="Ordner, in dem Modell-Checkpoints gespeichert werden.")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Intervall (in Batches) für Zwischen-Log-Ausgaben.")
    
    return parser.parse_args()

def train_one_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio, log_interval):
    model.train()
    total_loss = 0.0
    for batch_idx, (src, trg) in enumerate(loader):
        # src: moderne Text-Token-IDs, trg: Zielstil-Token-IDs
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        
        # Forward-Pass mit Teacher Forcing
        outputs = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
        # outputs: [Batch, trg_len, vocab_size]
        # Ignoriere das erste Token (<BOS>) bei der Loss-Berechnung
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(outputs, trg)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"[Batch {batch_idx+1}/{len(loader)}] Loss: {avg_loss:.4f}")
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            outputs = model(src, trg, teacher_forcing_ratio=0.0)
            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(outputs, trg)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Trainiere auf Gerät:", device)
    
    # Vokabular laden
    word2id = load_vocab(args.vocab_json)
    vocab_size = len(word2id)
    pad_id = word2id.get("<PAD>", 0)
    print("Vokabulargröße:", vocab_size)
    
    # DataLoader erstellen
    train_loader = create_dataloader(
        csv_file=args.train_csv,
        batch_size=args.batch_size,
        shuffle=True,
        pad_id=pad_id,
        article_col="modern_ids",    # Input: moderner Stil
        summary_col="shakespeare_ids"  # Ziel: stilisierter Text (z.B. Shakespeare)
    )
    val_loader = create_dataloader(
        csv_file=args.val_csv,
        batch_size=args.batch_size,
        shuffle=False,
        pad_id=pad_id,
        article_col="modern_ids",
        summary_col="shakespeare_ids"
    )
    
    # Modell initialisieren: Seq2Seq-LSTM-Modell
    encoder = Encoder(vocab_size, args.embed_dim, args.hidden_dim, args.num_layers, pad_id)
    decoder = Decoder(vocab_size, args.embed_dim, args.hidden_dim, args.num_layers, pad_id)
    model = Seq2Seq(encoder, decoder, pad_id).to(device)
    print(model)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoche {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.teacher_forcing_ratio, args.log_interval)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.save_dir, f"styleshift_epoch{epoch}_valloss{val_loss:.4f}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"** Neues bestes Modell gespeichert: {model_path}")
    
    print("\nTraining abgeschlossen.")

if __name__ == "__main__":
    main()
