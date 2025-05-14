"""
feedback.py
-----------
CLI-Skript zur Verarbeitung von Nutzerfeedback und zur Aktualisierung (Fine-Tuning) des Modells im Feedback AI-Projekt.

Beispielaufruf:
    python feedback.py --feedback_csv data/processed/feedback.csv \
                       --vocab_json data/processed/vocab.json \
                       --model_path models/feedback_ai_epoch3_valloss2.1357.pt \
                       --update_epochs 3 --lr 5e-4 --save_updated_model True
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from tqdm import tqdm

# Importiere benötigte Module aus src
from dataset import create_dataloader, load_vocab
from model import Encoder, Decoder, Seq2Seq

def parse_args():
    parser = argparse.ArgumentParser(description="Verarbeite Nutzerfeedback und update das Modell (Feedback AI).")
    parser.add_argument("--feedback_csv", type=str, required=True,
                        help="Pfad zur CSV-Datei mit Feedback-Daten (z.B. 'data/processed/feedback.csv').")
    parser.add_argument("--vocab_json", type=str, required=True,
                        help="Pfad zur vocab.json-Datei (Mapping Token -> ID).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Pfad zur aktuellen Modellgewichtsdatei (.pt).")
    parser.add_argument("--update_epochs", type=int, default=3,
                        help="Anzahl der Fine-Tuning-Epochen mit Feedback.")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Lernrate für das Fine-Tuning.")
    parser.add_argument("--save_updated_model", type=bool, default=True,
                        help="Ob das aktualisierte Modell gespeichert werden soll.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Gerät: 'cuda' oder 'cpu'.")
    parser.add_argument("--log_interval", type=int, default=20,
                        help="Intervall für Zwischen-Log-Ausgaben.")
    return parser.parse_args()

# Für dieses Skript verwenden wir denselben FeedbackDataset-Ansatz aus dataset.py
# und damit auch die create_dataloader()-Funktion. (Es wird erwartet, dass die CSV
# Spalten "token_ids" und "feedback_label" enthält.)

def train_feedback_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio):
    model.train()
    total_loss = 0.0
    for src, trg, feedback in tqdm(loader, desc="Feedback-Training", leave=False):
        src, trg, feedback = src.to(device), trg.to(device), feedback.to(device)
        optimizer.zero_grad()
        outputs = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss_per_token = criterion(outputs, trg)  # [Batch*seq_len]
        batch_size = src.size(0)
        loss_per_example = loss_per_token.view(batch_size, -1).sum(dim=1)
        # Beispiel-Gewichtung: Positives Feedback (Label 1) erhält Gewicht 1.0, negatives (0) 0.5
        weights = torch.where(feedback == 1, torch.tensor(1.0, device=device), torch.tensor(0.5, device=device))
        weighted_loss = (loss_per_example * weights).mean()
        weighted_loss.backward()
        optimizer.step()
        total_loss += weighted_loss.item()
    return total_loss / len(loader)

def main():
    args = parse_args()
    device = args.device
    print(f"Feedback-Update läuft auf: {device}")
    
    # Vokabular laden
    word2id = load_vocab(args.vocab_json)
    id2word = {v: k for k, v in word2id.items()}
    vocab_size = len(word2id)
    pad_id = word2id.get("<PAD>", 0)
    print("Vokabulargröße:", vocab_size)
    
    # Feedback-Daten laden
    # Wir nutzen hier create_dataloader, wobei Feedback-Daten die Spalte "token_ids" (Text)
    # und "feedback_label" enthalten.
    feedback_loader = create_dataloader(
        csv_file=args.feedback_csv,
        batch_size=32,
        shuffle=True,
        pad_id=pad_id,
        text_col="token_ids",
        label_col="feedback_label"
    )
    
    # Modellarchitektur initialisieren
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, pad_id)
    decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, pad_id)
    model = Seq2Seq(encoder, decoder, pad_id).to(device)
    
    # Lade existierende Modellgewichte
    if not os.path.exists(args.model_path):
        print(f"Modellpfad nicht gefunden: {args.model_path}")
        return
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Aktuelle Modellgewichte erfolgreich geladen.")
    
    # Setup für Fine-Tuning mit Feedback
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Führe Fine-Tuning-Epochen durch
    for epoch in range(1, args.update_epochs + 1):
        print(f"\n=== Feedback Fine-Tuning Epoche {epoch}/{args.update_epochs} ===")
        epoch_loss = train_feedback_epoch(model, feedback_loader, optimizer, criterion, device, teacher_forcing_ratio=0.5)
        print(f"Epoch {epoch} Feedback-Loss: {epoch_loss:.4f}")
    
    if args.save_updated_model:
        updated_model_path = os.path.join(os.path.dirname(args.model_path), "feedback_updated.pt")
        torch.save(model.state_dict(), updated_model_path)
        print(f"Updated Modell wurde unter {updated_model_path} gespeichert.")
    else:
        print("Kein Update-Speicherort gewählt; Modell nicht gespeichert.")

if __name__ == "__main__":
    main()
