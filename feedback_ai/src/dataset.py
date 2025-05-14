"""
dataset.py
----------
Dieses Modul implementiert Dataset-Klassen und Hilfsfunktionen für das Feedback AI-Projekt.
Es werden E-Mail- oder generierte Texte zusammen mit dem zugehörigen Nutzerfeedback (z. B. "thumbs_up" oder "thumbs_down")
aus einer CSV-Datei geladen. Dabei werden die Texte, die als tokenisierte Listen (z. B. "[2, 45, 78, 3]") vorliegen,
in Python-Listen konvertiert und als Tensoren ausgegeben. Zusätzlich wird das Feedback als Label (z. B. 1 oder 0) bereitgestellt.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class FeedbackDataset(Dataset):
    """
    Dataset für Feedback AI.

    Erwartet eine CSV-Datei mit den folgenden Spalten:
      - "token_ids": Enthält tokenisierte Texte als String-Repräsentationen von Listen, z. B. "[2, 45, 78, 3]".
      - "feedback_label": Enthält das Nutzerfeedback, z. B. "thumbs_up" oder "thumbs_down".
        Dieses Feedback wird in numerische Labels (z. B. 1 für positive, 0 für negative Bewertungen) umgewandelt.

    Bei jedem Zugriff liefert __getitem__ ein Tupel:
       (text_tensor, text_tensor, feedback_label)
    Hierbei kann der Text sowohl als Input als auch als Target genutzt werden (z. B. bei Vervollständigungsaufgaben).
    """
    def __init__(self, csv_file, text_col="token_ids", label_col="feedback_label"):
        self.df = pd.read_csv(csv_file)
        # Konvertiere die String-Repräsentationen in echte Python-Listen
        self.df[text_col] = self.df[text_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        # Konvertiere das Feedback in numerische Labels (z. B. 1 für "thumbs_up", 0 für "thumbs_down")
        self.df[label_col] = self.df[label_col].apply(lambda x: 1 if str(x).strip().lower() == "thumbs_up" else 0)
        self.texts = self.df[text_col].tolist()
        self.labels = self.df[label_col].tolist()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text_ids = self.texts[idx]
        label = self.labels[idx]
        text_tensor = torch.tensor(text_ids, dtype=torch.long)
        return text_tensor, text_tensor, label

def collate_fn(batch, pad_id=0):
    """
    Collate-Funktion zum Padding variabler Sequenzen im Batch.

    Args:
        batch: Eine Liste von Tupeln (text_tensor, text_tensor, feedback_label)
        pad_id: Der Wert, mit dem aufgefüllt wird (z. B. der <PAD>-Token)

    Returns:
        Tuple:
          - texts_padded: LongTensor [Batch, max_seq_len]
          - targets_padded: LongTensor [Batch, max_seq_len]
          - labels: LongTensor [Batch]
    """
    texts, targets, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=pad_id)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_id)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return texts_padded, targets_padded, labels_tensor

def create_dataloader(csv_file, batch_size=32, shuffle=True, pad_id=0, text_col="token_ids", label_col="feedback_label"):
    """
    Erzeugt einen DataLoader aus der angegebenen CSV-Datei.

    Args:
        csv_file (str): Pfad zur CSV-Datei (z. B. 'train.csv')
        batch_size (int): Batch-Größe
        shuffle (bool): Ob die Daten gemischt werden sollen
        pad_id (int): Padding-Wert (z. B. der ID für <PAD>)
        text_col (str): Name der Spalte, die die tokenisierten Texte enthält
        label_col (str): Name der Spalte, die das Feedback enthält

    Returns:
        DataLoader: Liefert Tupel (texts_padded, targets_padded, labels)
    """
    dataset = FeedbackDataset(csv_file, text_col=text_col, label_col=label_col)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id)
    )
    return loader

if __name__ == "__main__":
    # Testcode, um das FeedbackDataset zu überprüfen
    test_csv = "../data/processed/train.csv"  # Passe den Pfad bei Bedarf an
    if os.path.exists(test_csv):
        print("Teste FeedbackDataset mit:", test_csv)
        loader = create_dataloader(test_csv, batch_size=4, shuffle=False, pad_id=0)
        for i, (texts, targets, labels) in enumerate(loader):
            print(f"Batch {i+1}:")
            print("Texts-Shape:", texts.shape)
            print("Labels:", labels)
            if i == 1:
                break
        print("FeedbackDataset-Test erfolgreich.")
    else:
        print("Testdatei nicht gefunden. Bitte den Pfad anpassen.")
