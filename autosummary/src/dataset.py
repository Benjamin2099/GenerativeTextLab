"""
dataset.py
----------
Dieses Skript implementiert ein PyTorch Dataset für AutoSummary. 
Es wird erwartet, dass die vorverarbeiteten CSV-Dateien zwei Spalten enthalten:
    - "article_ids": Eine Liste von Token-IDs, die den Artikel repräsentieren.
    - "summary_ids": Eine Liste von Token-IDs, die die zugehörige Zusammenfassung repräsentieren.
Die Listen werden aus ihrem String-Format in echte Python-Listen konvertiert.
Zusätzlich gibt es Funktionen zum Erzeugen eines DataLoaders mit Padding sowie
zum Laden des Vokabulars aus einer JSON-Datei.
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class SummarizationDataset(Dataset):
    """
    PyTorch Dataset für die AutoSummary-Aufgabe.
    
    Erwartet eine CSV-Datei, in der die Spalten "article_ids" und "summary_ids"
    enthalten sind. Diese Spalten speichern tokenisierte Texte als Strings,
    z.B. "[2, 45, 78, 3]".
    
    Bei jedem Zugriff gibt __getitem__ ein Tupel zurück:
      (article_tensor, summary_tensor)
    """
    def __init__(self, csv_file, article_col="article_ids", summary_col="summary_ids"):
        self.df = pd.read_csv(csv_file)
        # Konvertiere String-Repräsentationen in echte Listen
        self.df[article_col] = self.df[article_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        self.df[summary_col] = self.df[summary_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        self.articles = self.df[article_col].tolist()
        self.summaries = self.df[summary_col].tolist()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        article_ids = self.articles[idx]
        summary_ids = self.summaries[idx]
        article_tensor = torch.tensor(article_ids, dtype=torch.long)
        summary_tensor = torch.tensor(summary_ids, dtype=torch.long)
        return article_tensor, summary_tensor

def collate_fn(batch, pad_id=0):
    """
    Sorgt dafür, dass alle Sequenzen im Batch auf die gleiche Länge aufgefüllt werden.
    
    Args:
        batch: Eine Liste von Tupeln (article_tensor, summary_tensor)
        pad_id: Der Wert, mit dem aufgefüllt wird (Standard: 0)
    
    Returns:
        article_padded: LongTensor [Batch, max_article_len]
        summary_padded: LongTensor [Batch, max_summary_len]
    """
    articles, summaries = zip(*batch)
    article_padded = pad_sequence(articles, batch_first=True, padding_value=pad_id)
    summary_padded = pad_sequence(summaries, batch_first=True, padding_value=pad_id)
    return article_padded, summary_padded

def create_dataloader(csv_file, batch_size=32, shuffle=True, pad_id=0, article_col="article_ids", summary_col="summary_ids"):
    """
    Erzeugt einen DataLoader aus einer CSV-Datei.
    
    Args:
        csv_file (str): Pfad zur CSV-Datei (z. B. train.csv)
        batch_size (int): Batch-Größe
        shuffle (bool): Ob die Daten gemischt werden sollen
        pad_id (int): Padding-Wert
        article_col (str): Name der Spalte mit den Artikeldaten
        summary_col (str): Name der Spalte mit den Zusammenfassungsdaten
        
    Returns:
        DataLoader, der Tupel (article_padded, summary_padded) liefert.
    """
    dataset = SummarizationDataset(csv_file, article_col=article_col, summary_col=summary_col)
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
    test_csv = "../data/processed/train.csv"
    if os.path.exists(test_csv):
        print("Teste SummarizationDataset mit:", test_csv)
        loader = create_dataloader(test_csv, batch_size=4, shuffle=False, pad_id=0)
        for i, (articles, summaries) in enumerate(loader):
            print(f"Batch {i+1}:")
            print("Article-Shape:", articles.shape)
            print("Summary-Shape:", summaries.shape)
            if i == 1:
                break
        print("Dataset-Test erfolgreich.")
    else:
        print("Testdatei nicht gefunden. Bitte den Pfad anpassen.")
