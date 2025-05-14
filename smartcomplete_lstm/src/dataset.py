"""
dataset.py
----------
Enthält Klassen und Funktionen zum Laden von Textdaten (tokenisierte Sequenzen)
und deren Verarbeitung in PyTorch DataLoadern, insbesondere für das SmartComplete-LSTM-Projekt.
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    """
    Repräsentiert ein Dataset für LSTM-Textverarbeitung.
    
    Erwartet eine CSV-Datei (train.csv, val.csv, etc.) mit einer Spalte 
    (default: 'token_ids'), in der jede Zeile eine Liste von Token-IDs enthält.
    Beispiel:
       token_ids
       [2, 45, 56, 7, 13]
       [2, 19, 21, 11, 5, 5]
       ...
       
    Dabei wird davon ausgegangen, dass alle Vorverarbeitungsschritte 
    (Säuberung, Tokenisierung, ID-Mapping) bereits erfolgt sind.
    """
    def __init__(self, csv_file, seq_col="token_ids"):
        """
        Args:
            csv_file (str): Pfad zur CSV-Datei mit tokenisierten Daten
            seq_col (str): Name der Spalte, in der die Token-IDs (Listen) gespeichert sind
        """
        self.df = pd.read_csv(csv_file)
        self.seq_col = seq_col
        
        # Konvertiert den String wie "[2, 45, 7]" in eine echte Python-Liste
        self.df[self.seq_col] = self.df[self.seq_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    def __len__(self):
        """Anzahl der Zeilen im Dataset."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Gibt eine Eingabesequenz (alle Tokens außer dem letzten)
        und eine Zielsequenz (alle Tokens außer dem ersten) zurück.
        So kann das LSTM lernen, das nächste Wort zu predicten.
        
        Beispiel:
         Tokenliste: [2, 45, 56, 7, 13]
         Input:  [2, 45, 56, 7]
         Target: [45, 56, 7, 13]
        """
        tokens_list = self.df.iloc[idx][self.seq_col]
        
        # Evtl. Fehlerabsicherung: Falls Zeile leer oder so
        if not tokens_list or len(tokens_list) < 2:
            tokens_list = [0, 0]  # Minimalfall
        
        input_seq = tokens_list[:-1]
        target_seq = tokens_list[1:]
        
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        target_seq = torch.tensor(target_seq, dtype=torch.long)
        
        return input_seq, target_seq


def collate_fn(batch, pad_id=0):
    """
    Collate-Funktion für den DataLoader, um variable Sequenzlängen
    in einen Batch zu bringen.
    
    batch: Liste von (input_seq, target_seq)-Tupeln
    pad_id: Integer-Wert, mit dem Sequenzen aufgefüllt werden (Default: 0).
    
    Rückgabe:
      input_padded:  [Batch, MaxLen] (LongTensor)
      target_padded: [Batch, MaxLen] (LongTensor)
    """
    inputs, targets = zip(*batch)  # Aufteilen in Eingabesequenzen und Zielsequenzen
    
    # pad_sequence erstellt einen Tensor, bei dem kürzere Sequenzen mit pad_id aufgefüllt werden
    input_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    target_padded = pad_sequence(targets, batch_first=True, padding_value=pad_id)
    
    return input_padded, target_padded


def create_dataloader(csv_file, batch_size, shuffle=True, pad_id=0):
    """
    Erzeugt einen PyTorch DataLoader aus einer CSV-Datei.
    
    Args:
        csv_file (str): Pfad zur CSV-Datei (z. B. train.csv oder val.csv)
        batch_size (int): Größe der Batches
        shuffle (bool): Ob die Daten durchmischt werden sollen
        pad_id (int): Padding-Index, der zum Auffüllen der Sequenzen verwendet wird

    Returns:
        DataLoader: PyTorch DataLoader-Objekt mit dem definierten collate_fn
    """
    dataset = TextDataset(csv_file=csv_file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id)
    )
    return loader


# Optionale Funktion: Laden des Vokabulars
def load_vocab(vocab_path):
    """
    Lädt ein Vokabular (Token->ID) aus einer JSON-Datei.
    
    Args:
        vocab_path (str): Pfad zur vocab.json
        
    Returns:
        dict: Mapping von token (str) zu ID (int)
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        word2id = json.load(f)
    return word2id


if __name__ == "__main__":
    # Beispiel: Kleiner Test, wenn du dataset.py direkt ausführst
    csv_example = "../data/processed/train.csv"
    loader = create_dataloader(csv_example, batch_size=4, shuffle=False, pad_id=0)
    
    for i, (inp, tgt) in enumerate(loader):
        print(f"Batch {i+1} - Eingabeform: {inp.shape}, Zielform: {tgt.shape}")
        if i == 1:
            break
    print("Testlauf erfolgreich!")
