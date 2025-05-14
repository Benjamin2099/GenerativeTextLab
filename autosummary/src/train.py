"""
train.py
--------
Haupt-Trainingsskript (CLI) für das AutoSummary-Projekt.
Trainiert ein Seq2Seq-Modell (Encoder-Decoder-LSTM) zur abstrakten Zusammenfassung.
Beispielaufruf:
    python train.py --train_csv ../data/processed/train.csv \
                    --val_csv ../data/processed/val.csv \
                    --vocab_json ../data/processed/vocab.json \
                    --batch_size 32 --epochs 5 --lr 0.001 \
                    --embed_dim 128 --hidden_dim 256 --num_layers 2 --dropout 0.1 \
                    --save_dir ../models --log_interval 50
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Eigene Module aus src importieren
from dataset import create_dataloader, load_vocab
from model import Seq2Seq, Encoder, Decoder

def parse_args():
    """
    Liest Kommandozeilen-Argumente ein.
    """
    parser = argparse.ArgumentParser(description="Train AutoSummary Seq2Seq Model")
    
    # Pfade zu den Daten
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Pfad zur Trainings-CSV-Datei (z.B. 'train.csv').")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Pfad zur Validierungs-CSV-Datei (z.B. 'val.csv').")
    parser.add_argument("--vocab_json", type=str, required=True,
                        help="Pfad zur vocab.json-Datei (Mapping Token->ID).")
    
    # Hyperparameter
    parser.add_argument("--batch_size", type=int, default=32, help="Batch-Größe.")
    parser.add_argument("--epochs", type=int, default=5, help="Anzahl der Trainings-Epochen.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Lernrate.")
    parser.add_argument("--embed_dim", type=int, default=128, help="Dimension der Embeddings.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Größe der Hidden-States im LSTM.")
    parser.add_argument("--num_layers", type=int, default=2, help="Anzahl der LSTM-Schichten.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout (wirkt ab num_layers>1).")
    
    # Weitere Parameter
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5,
                        help="Wahrscheinlichkeit, dass Teacher Forcing angewendet wird.")
    
    # Speicherort für Modell-Checkpoints
    parser.add_argument("--save_dir", type=str, default="../models",
                        help="Ordner, in dem Modell-Checkpoints gespeichert werden.")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Intervall (in Batches) für Zwischen-Log-Ausgaben.")
    
    args = parser.parse_args()
    return args

def train_one_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio, log_interval):
    """
    Trainiert das Modell für eine Epoche und gibt den durchschnittlichen Loss zurück.
    """
    model.train()
    total_loss = 0.0
    for batch_idx, (src, trg) in enumerate(loader):
        # src: Artikel, trg: Zusammenfassung (inkl. <BOS> am Anfang)
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        # Forward-Pass: Teacher Forcing wird angewendet
        outputs = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
        # outputs: [Batch, trg_len, vocab_size]
        # Wir ignorieren das erste Token (<BOS>) beim Loss
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(outputs, trg)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  [Batch {batch_idx+1}/{len(loader)}] Loss: {avg_loss:.4f}")
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    """
    Berechnet den durchschnittlichen Loss auf dem Validierungsdatensatz.
    """
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
    
    # 1) Geräteauswahl
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training auf Gerät:", device)
    
    # 2) Vokabular laden
    word2id = load_vocab(args.vocab_json)
    vocab_size = len(word2id)
    pad_id = word2id.get("<PAD>", 0)
    print("Vokabulargröße:", vocab_size)
    
    # 3) DataLoader erstellen
    # Hier wird angenommen, dass die CSVs die Spalten "article_ids" und "summary_ids" enthalten.
    train_loader = create_dataloader(
        csv_file=args.train_csv,
        batch_size=args.batch_size,
        shuffle=True,
        pad_id=pad_id,
        article_col="article_ids",
        summary_col="summary_ids"
    )
    val_loader = create_dataloader(
        csv_file=args.val_csv,
        batch_size=args.batch_size,
        shuffle=False,
        pad_id=pad_id,
        article_col="article_ids",
        summary_col="summary_ids"
    )
    
    # 4) Modell initialisieren: Seq2Seq-Modell (Encoder-Decoder)
    EMBED_DIM = args.embed_dim
    HIDDEN_DIM = args.hidden_dim
    NUM_LAYERS = args.num_layers
    encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, pad_id)
    decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, pad_id)
    model = Seq2Seq(encoder, decoder, pad_id).to(device)
    print(model)
    
    # 5) Loss und Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 6) Speicherordner erstellen, falls nicht vorhanden
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    
    # 7) Trainings- und Validierungsschleife
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoche {epoch}/{args.epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.teacher_forcing_ratio, args.log_interval)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Speichern des besten Modells
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.save_dir, f"autosummary_epoch{epoch}_valloss{val_loss:.4f}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"** Neues bestes Modell gespeichert: {model_path}")
    
    print("\nTraining abgeschlossen.")

if __name__ == "__main__":
    main()
