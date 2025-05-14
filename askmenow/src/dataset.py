"""
dataset.py
----------
Dieses Modul implementiert ein PyTorch Dataset für Q&A-Daten im AskMeNow-Projekt.
Die CSV-Dateien sollten mindestens zwei Spalten enthalten:
  - "question_ids": Tokenisierte Frage als String, z. B. "[2, 45, 78, 3]"
  - "answer_ids":   Tokenisierte Antwort als String, z. B. "[2, 56, 90, 3]"

Die Funktionen in diesem Modul ermöglichen es, die Daten einzulesen, die String-Repräsentationen
in Python-Listen zu konvertieren und diese in Tensoren umzuwandeln, sodass sie für das Training
und die Inferenz eines Q&A-Modells genutzt werden können.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class QADataset(Dataset):
    def __init__(self, csv_file, question_col="question_ids", answer_col="answer_ids"):
        self.df = pd.read_csv(csv_file)
        # Konvertiere String-Repräsentationen in Python-Listen, falls nötig.
        self.df[question_col] = self.df[question_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        self.df[answer_col] = self.df[answer_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
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
    Sorgt dafür, dass alle Sequenzen im Batch auf die gleiche Länge aufgefüllt werden.
    
    Args:
        batch: Eine Liste von Tupeln (question_tensor, answer_tensor)
        pad_id: Wert, mit dem aufgefüllt wird (z. B. der <PAD>-Token)
    
    Returns:
        Tuple: (questions_padded, answers_padded) mit einheitlicher Sequenzlänge
    """
    questions, answers = zip(*batch)
    questions_padded = pad_sequence(questions, batch_first=True, padding_value=pad_id)
    answers_padded = pad_sequence(answers, batch_first=True, padding_value=pad_id)
    return questions_padded, answers_padded

def create_qa_dataloader(csv_file, batch_size=32, shuffle=True, pad_id=0, question_col="question_ids", answer_col="answer_ids"):
    """
    Erzeugt einen DataLoader aus der angegebenen CSV-Datei.
    
    Args:
        csv_file (str): Pfad zur CSV-Datei (z. B. 'train.csv')
        batch_size (int): Batch-Größe
        shuffle (bool): Ob die Daten gemischt werden sollen
        pad_id (int): Padding-Wert (z. B. die ID für <PAD>)
        question_col (str): Name der Spalte mit den tokenisierten Fragen
        answer_col (str): Name der Spalte mit den tokenisierten Antworten
    
    Returns:
        DataLoader: Liefert Tupel (questions_padded, answers_padded)
    """
    dataset = QADataset(csv_file, question_col=question_col, answer_col=answer_col)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id)
    )
    return loader

# Beispielhafter Test (wird ausgeführt, wenn das Modul direkt gestartet wird)
if __name__ == "__main__":
    test_csv = "../data/processed/train.csv"  # Passe den Pfad ggf. an
    if os.path.exists(test_csv):
        print("Teste QADataset mit:", test_csv)
        loader = create_qa_dataloader(test_csv, batch_size=4, shuffle=False, pad_id=0)
        for i, (questions, answers) in enumerate(loader):
            print(f"Batch {i+1}:")
            print("Questions-Shape:", questions.shape)
            print("Answers-Shape:", answers.shape)
            if i == 1:
                break
        print("QADataset-Test erfolgreich.")
    else:
        print("Testdatei nicht gefunden. Bitte den Pfad anpassen.")
