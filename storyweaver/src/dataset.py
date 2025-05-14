"""
dataset.py
----------
Implementiert PyTorch Datasets und DataLoader-Hilfsfunktionen 
für das StoryWeaver-Projekt. Hiermit kannst du deine tokenisierten 
Märchen-/Fantasy-Texte in Training und Inferenz einspeisen.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class StoryDataset(Dataset):
    """
    Repräsentiert einen Datensatz für das StoryWeaver-Projekt.
    
    Erwartet, dass in einer CSV-Datei eine Spalte 'token_ids' existiert,
    in der jeweils eine Liste von Token-IDs (z. B. [2, 45, 7, 13]) 
    abgespeichert ist.
    
    Nutzung:
    --------
      ds = StoryDataset(csv_file='train.csv')
      inp, tgt = ds[0]  # Hol dir das (input_seq, target_seq) 
    """
    def __init__(self, csv_file, seq_col="token_ids", bos_id=2, eos_id=3):
        """
        Args:
            csv_file (str): Pfad zur CSV-Datei (z. B. train.csv, val.csv)
            seq_col (str): Spalte, in der die tokenisierten IDs stehen
            bos_id (int): ID für <BOS> (optional, falls du sie im Code benötigst)
            eos_id (int): ID für <EOS> (optional)
        """
        self.df = pd.read_csv(csv_file)
        
        # Konvertiere Strings wie "[2, 45, 7]" in echte Python-Listen
        # (Wenn du in data_preprocessing.ipynb 'token_ids' als Liste gespeichert hast)
        self.df[seq_col] = self.df[seq_col].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )
        self.samples = self.df[seq_col].tolist()
        
        # Speichere Sondertokens, falls du sie brauchst
        self.bos_id = bos_id
        self.eos_id = eos_id
    
    def __len__(self):
        """Anzahl der Zeilen im Datensatz."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Gibt (input_seq, target_seq) als Tensors zurück.
        
        Konvention:
          input_seq = Tokens[:-1]
          target_seq= Tokens[1:]
          
        Falls die Sequenz zu kurz ist (1 Token),
        erstellen wir einen Minimal-Fallback: [<BOS>, <EOS>]
        """
        tokens = self.samples[idx]
        if len(tokens) < 2:
            # Minimal fallback
            tokens = [self.bos_id, self.eos_id]
        
        # input_seq = alle Tokens außer dem letzten
        inp = torch.tensor(tokens[:-1], dtype=torch.long)
        
        # target_seq = alle Tokens außer dem ersten
        tgt = torch.tensor(tokens[1:], dtype=torch.long)
        
        return inp, tgt

def collate_fn(batch, pad_id=0):
    """
    Collate-Funktion, um variable Eingabesequenzen im Batch 
    auf eine einheitliche Länge zu bringen (Padding).
    
    batch: Liste von (input_seq, target_seq) - Tupeln
    
    Rückgabe:
      input_padded:  LongTensor [Batch, MaxLen_in_Batch]
      target_padded: LongTensor [Batch, MaxLen_in_Batch]
    """
    input_seqs, target_seqs = zip(*batch)
    
    # Mit pad_sequence aus PyTorch füllen wir kürzere Sequenzen auf
    inp_padded = pad_sequence(
        input_seqs, 
        batch_first=True, 
        padding_value=pad_id
    )
    tgt_padded = pad_sequence(
        target_seqs, 
        batch_first=True, 
        padding_value=pad_id
    )
    
    return inp_padded, tgt_padded

def create_dataloader(csv_file, batch_size=32, shuffle=True, pad_id=0):
    """
    Erzeugt einen PyTorch DataLoader aus einer CSV-Datei,
    indem es StoryDataset nutzt.
    
    Args:
        csv_file (str): Pfad zur CSV (z. B. 'train.csv')
        batch_size (int): Größe der Batches
        shuffle (bool): Ob die Daten durchmischt werden sollen
        pad_id (int): Index des <PAD>-Tokens
    
    Returns:
        DataLoader: PyTorch-Objekt, das in einer Trainings- oder 
                    Evaluierungsschleife iteriert werden kann.
    """
    dataset = StoryDataset(csv_file=csv_file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id)
    )
    return loader

def load_vocab(vocab_path):
    """
    Lädt ein Vokabular (Token->ID) aus einer JSON-Datei.
    
    Args:
        vocab_path (str): Pfad zur vocab.json
        
    Returns:
        dict: Mapping token->int
    """
    import json
    with open(vocab_path, "r", encoding="utf-8") as f:
        word2id = json.load(f)
    return word2id

# Optionaler Testcode
if __name__ == "__main__":
    # Kleiner Test, wenn man dataset.py direkt ausführt
    test_csv = "../data/processed/train.csv"
    if os.path.exists(test_csv):
        loader = create_dataloader(test_csv, batch_size=4, shuffle=False, pad_id=0)
        for i, (inp, tgt) in enumerate(loader):
            print(f"Batch {i+1}:")
            print("Eingabe-Shape:", inp.shape)
            print("Ziel-Shape:", tgt.shape)
            if i == 1:
                break
        print("Dataset-Test erfolgreich.")
    else:
        print("Keine Testdatei gefunden. Bitte pfad anpassen.")
