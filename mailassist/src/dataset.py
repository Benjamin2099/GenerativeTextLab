"""
dataset.py
----------
Dieses Modul implementiert ein PyTorch Dataset für MailAssist. 
Es erwartet eine CSV-Datei mit zwei Spalten:
  - "subject_ids": Tokenisierte Betreffzeilen (als Liste im String-Format)
  - "body_ids": Tokenisierte E-Mail-Inhalte (als Liste im String-Format)
Beim Zugriff liefert __getitem__ ein Tupel:
  (subject_tensor, body_tensor)
Diese können als Input (z. B. Betreff) und Target (z. B. kompletter E-Mail-Body)
für Trainings- und Inferenz-Skripte verwendet werden.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class MailDataset(Dataset):
    """
    Dataset für E-Mail-Daten im MailAssist-Projekt.
    
    Erwartet eine CSV-Datei mit den Spalten "subject_ids" und "body_ids",
    wobei die Inhalte als String-repräsentierte Listen vorliegen, z. B. "[2, 45, 78, 3]".
    
    __getitem__ gibt ein Tupel (subject_tensor, body_tensor) zurück.
    """
    def __init__(self, csv_file, subject_col="subject_ids", body_col="body_ids"):
        self.df = pd.read_csv(csv_file)
        # Konvertiere String-Repräsentationen in echte Listen, falls nötig
        self.df[subject_col] = self.df[subject_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        self.df[body_col] = self.df[body_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        self.subjects = self.df[subject_col].tolist()
        self.bodies = self.df[body_col].tolist()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        subject_ids = self.subjects[idx]
        body_ids = self.bodies[idx]
        subject_tensor = torch.tensor(subject_ids, dtype=torch.long)
        body_tensor = torch.tensor(body_ids, dtype=torch.long)
        return subject_tensor, body_tensor

def collate_fn(batch, pad_id=0):
    """
    Collate-Funktion zum Padding variabler Sequenzen im Batch.
    
    Args:
        batch: Liste von (subject_tensor, body_tensor)-Tupeln.
        pad_id: Wert, der zum Auffüllen der Sequenzen verwendet wird.
    
    Returns:
        Tuple: (subjects_padded, bodies_padded) mit einheitlicher Sequenzlänge.
    """
    subjects, bodies = zip(*batch)
    subjects_padded = pad_sequence(subjects, batch_first=True, padding_value=pad_id)
    bodies_padded = pad_sequence(bodies, batch_first=True, padding_value=pad_id)
    return subjects_padded, bodies_padded

def create_dataloader(csv_file, batch_size=32, shuffle=True, pad_id=0, subject_col="subject_ids", body_col="body_ids"):
    """
    Erzeugt einen DataLoader aus der angegebenen CSV-Datei.
    
    Args:
        csv_file (str): Pfad zur CSV (z. B. train.csv).
        batch_size (int): Batch-Größe.
        shuffle (bool): Ob die Daten gemischt werden sollen.
        pad_id (int): Padding-Wert.
        subject_col (str): Name der Spalte mit tokenisierten Betreffzeilen.
        body_col (str): Name der Spalte mit tokenisierten E-Mail-Inhalten.
        
    Returns:
        DataLoader, der Tupel (subjects_padded, bodies_padded) liefert.
    """
    dataset = MailDataset(csv_file, subject_col=subject_col, body_col=body_col)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id)
    )
    return loader

def load_vocab(vocab_path):
    """
    Lädt das Vokabular (Mapping Token -> ID) aus einer JSON-Datei.
    
    Args:
        vocab_path (str): Pfad zur Vokabular-Datei (z. B. vocab.json)
    
    Returns:
        dict: Mapping von Token (str) zu ID (int)
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        word2id = json.load(f)
    return word2id

if __name__ == "__main__":
    # Optionaler Testcode, um das Dataset zu überprüfen, wenn das Skript direkt ausgeführt wird.
    test_csv = "../data/processed/train.csv"  # Passe den Pfad bei Bedarf an.
    if os.path.exists(test_csv):
        print("Teste MailDataset mit:", test_csv)
        loader = create_dataloader(test_csv, batch_size=4, shuffle=False, pad_id=0)
        for i, (subjects, bodies) in enumerate(loader):
            print(f"Batch {i+1}:")
            print("Subjects-Shape:", subjects.shape)
            print("Bodies-Shape:", bodies.shape)
            if i == 1:
                break
        print("MailDataset-Test erfolgreich.")
    else:
        print("Testdatei nicht gefunden. Bitte den Pfad anpassen.")
