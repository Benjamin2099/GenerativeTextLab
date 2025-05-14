"""
dataset.py
----------
Dieses Modul implementiert PyTorch Dataset-Klassen und Hilfsfunktionen 
für das AskMeNow-Projekt, bei dem es um FAQ-/Wissens-Q&A mit Retrieval-Augmented Generation (RAG) geht.
Die CSV-Dateien sollten mindestens zwei Spalten enthalten:
    - "question_ids": Tokenisierte Fragen (als String, z. B. "[2, 45, 78, 3]")
    - "answer_ids": Tokenisierte Antworten (als String)
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class FAQDataset(Dataset):
    """
    Dataset für FAQ-/Wissens-Q&A-Daten.
    
    Erwartet eine CSV-Datei mit den Spalten "question_ids" und "answer_ids".
    Diese Spalten enthalten tokenisierte Listen (als String repräsentiert, z. B. "[2, 45, 78, 3]").
    
    Beim Zugriff liefert __getitem__ ein Tupel (question_tensor, answer_tensor).
    """
    def __init__(self, csv_file, question_col="question_ids", answer_col="answer_ids"):
        self.df = pd.read_csv(csv_file)
        # Konvertiere die String-Repräsentationen in echte Listen
        self.df[question_col] = self.df[question_col].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )
        self.df[answer_col] = self.df[answer_col].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )
        self.questions = self.df[question_col].tolist()
        self.answers = self.df[answer_col].tolist()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        question_ids = self.questions[idx]
        answer_ids = self.answers[idx]
        question_tensor = torch.tensor(question_ids, dtype=torch.long)
        answer_tensor = torch.tensor(answer_ids, dtype=torch.long)
        return question_tensor, answer_tensor

def collate_fn(batch, pad_id=0):
    """
    Collate-Funktion zum Padding variabler Sequenzen im Batch.
    
    Args:
        batch: Eine Liste von Tupeln (question_tensor, answer_tensor).
        pad_id: Der Wert, der zum Auffüllen der Sequenzen verwendet wird.
    
    Returns:
        Tuple (questions_padded, answers_padded) mit den Formen [Batch, max_seq_len].
    """
    questions, answers = zip(*batch)
    questions_padded = pad_sequence(questions, batch_first=True, padding_value=pad_id)
    answers_padded = pad_sequence(answers, batch_first=True, padding_value=pad_id)
    return questions_padded, answers_padded

def create_dataloader(csv_file, batch_size=32, shuffle=True, pad_id=0,
                      question_col="question_ids", answer_col="answer_ids"):
    """
    Erzeugt einen PyTorch DataLoader aus der gegebenen CSV-Datei.
    
    Args:
        csv_file (str): Pfad zur CSV-Datei (z. B. train.csv).
        batch_size (int): Batch-Größe.
        shuffle (bool): Ob die Daten gemischt werden sollen.
        pad_id (int): Padding-Wert.
        question_col (str): Name der Spalte mit tokenisierten Fragen.
        answer_col (str): Name der Spalte mit tokenisierten Antworten.
        
    Returns:
        DataLoader: Ein Iterator, der Tupel (questions_padded, answers_padded) liefert.
    """
    dataset = FAQDataset(csv_file, question_col, answer_col)
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
        vocab_path (str): Pfad zur Vokabular-Datei (z. B. vocab.json).
    
    Returns:
        dict: Mapping von Token (str) zu ID (int).
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        word2id = json.load(f)
    return word2id

if __name__ == "__main__":
    # Optional: Testcode, um das FAQDataset zu überprüfen.
    test_csv = "../data/processed/train.csv"  # Passe den Pfad an, wenn nötig.
    if os.path.exists(test_csv):
        print("Teste FAQDataset mit:", test_csv)
        loader = create_dataloader(test_csv, batch_size=4, shuffle=False, pad_id=0)
        for i, (questions, answers) in enumerate(loader):
            print(f"Batch {i+1}:")
            print("Questions-Shape:", questions.shape)
            print("Answers-Shape:", answers.shape)
            if i == 1:
                break
        print("FAQDataset-Test erfolgreich.")
    else:
        print("Testdatei nicht gefunden. Bitte den Pfad anpassen.")
