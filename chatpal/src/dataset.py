"""
dataset.py
----------
Dieses Skript kümmert sich um das Laden und Batching 
von Dialog-Daten für ChatPal. 

Angenommen, wir haben pro Zeile:
 - user_ids: Liste von Token-IDs für den User-Eingang
 - bot_ids:  Liste von Token-IDs für die Bot-Antwort
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class ChatDataset(Dataset):
    """
    PyTorch Dataset für Chat-Daten.
    
    Erwartet eine CSV mit zwei Spalten (z. B. 'user_ids' und 'bot_ids') 
    die bereits tokenisiert/aufbereitet sind (Listen von IDs).
    
    Beispiel:
      user_ids: "[2, 15, 16, 20]"
      bot_ids:  "[3, 47, 33, 8]"
      
    Nutzung:
    --------
      ds = ChatDataset(csv_file='train.csv')
      user_seq, bot_seq = ds[0]
      -> user_seq (Tensor), bot_seq (Tensor)
    """
    def __init__(self, csv_file, user_col="user_ids", bot_col="bot_ids"):
        """
        Args:
            csv_file (str): Pfad zur CSV-Datei (train.csv, val.csv ...)
            user_col (str): Name der Spalte mit User-Tokenlisten
            bot_col  (str): Name der Spalte mit Bot-Tokenlisten
        """
        self.df = pd.read_csv(csv_file)
        
        # Konvertiere die gespeicherten String-Listen in echte Python-Listen
        self.df[user_col] = self.df[user_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        self.df[bot_col]  = self.df[bot_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
        self.user_samples = self.df[user_col].tolist()
        self.bot_samples  = self.df[bot_col].tolist()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Gibt ein Tupel (user_seq, bot_seq) zurück, 
        wobei beide Listen von Token-IDs in einen Tensor konvertiert werden.
        """
        user_ids = self.user_samples[idx]
        bot_ids  = self.bot_samples[idx]
        
        user_tensor = torch.tensor(user_ids, dtype=torch.long)
        bot_tensor  = torch.tensor(bot_ids,  dtype=torch.long)
        
        return user_tensor, bot_tensor

def collate_fn(batch, pad_id=0):
    """
    Collate-Funktion für den DataLoader.
    
    batch: Liste von (user_tensor, bot_tensor)
    
    1) Wir packen alle user_tensors in user_padded
    2) Entsprechend alle bot_tensors in bot_padded
    3) Padding mit 'pad_id'
    
    Rückgabe: (user_padded, bot_padded)
    """
    user_seqs, bot_seqs = zip(*batch)
    user_padded = pad_sequence(user_seqs, batch_first=True, padding_value=pad_id)
    bot_padded  = pad_sequence(bot_seqs,  batch_first=True, padding_value=pad_id)
    return user_padded, bot_padded

def create_dataloader(csv_file, batch_size=32, shuffle=True, pad_id=0, user_col="user_ids", bot_col="bot_ids"):
    """
    Erzeugt einen PyTorch DataLoader, 
    indem ChatDataset + collate_fn kombiniert wird.
    
    Args:
        csv_file (str): Pfad zur CSV (z. B. 'train.csv')
        batch_size (int): Batch-Größe
        shuffle (bool): Ob die Daten durchmischt werden sollen
        pad_id (int): ID für Padding-Tokens
        user_col (str): Name der User-Spalte
        bot_col  (str): Name der Bot-Spalte
    
    Returns:
        DataLoader: Iterator über Batches (user_padded, bot_padded)
    """
    dataset = ChatDataset(csv_file=csv_file, user_col=user_col, bot_col=bot_col)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id)
    )
    return loader

if __name__ == "__main__":
    # Optional: kleiner Test, wenn du dataset.py alleine ausführst
    import os
    
    test_csv = "../data/processed/train.csv"
    if os.path.exists(test_csv):
        print("Teste ChatDataset mit:", test_csv)
        loader = create_dataloader(test_csv, batch_size=4, shuffle=False, pad_id=0)
        for i, (u, b) in enumerate(loader):
            print(f"Batch {i+1}:")
            print("User shape:", u.shape)
            print("Bot shape:", b.shape)
            if i == 1:
                break
        print("Dataset-Test abgeschlossen.")
    else:
        print("Keine Testdatei gefunden. Bitte pfad anpassen.")
