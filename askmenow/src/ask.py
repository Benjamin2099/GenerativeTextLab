"""
ask.py
------
CLI-Skript zur Beantwortung von Nutzerfragen im AskMeNow-Projekt.
Das Skript führt folgende Schritte aus:
  1. Lädt eine Wissensbasis (z.B. als CSV, Spalte "document") aus dem Ordner data/processed/.
  2. Baut einen BM25-Retriever auf, um die Top-N relevanten Dokumente basierend auf der Eingabefrage zu finden.
  3. Kombiniert die Frage und die abgerufenen Dokumente zu einem Prompt.
  4. Nutzt einen vortrainierten T5-Reader (Transformer) zur Generierung einer Antwort.
  
Beispielaufruf:
    python ask.py --question "Wie funktionieren neuronale Netze?" \
                  --knowledge_csv data/processed/knowledge.csv \
                  --model_name t5-small \
                  --max_length 100
"""

import argparse
import os
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

# Stelle sicher, dass NLTK-Tokendaten vorhanden sind (einmalig)
nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser(description="Beantworte Nutzerfragen mit Retrieval-Augmented Generation (RAG) im AskMeNow-Projekt.")
    parser.add_argument("--question", type=str, required=True,
                        help="Die Frage, die beantwortet werden soll.")
    parser.add_argument("--knowledge_csv", type=str, required=True,
                        help="Pfad zur CSV-Datei mit der Wissensbasis (Spalte 'document').")
    parser.add_argument("--model_name", type=str, default="t5-small",
                        help="Name des vortrainierten T5-Modells (z.B. 't5-small').")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximale Länge der generierten Antwort.")
    parser.add_argument("--top_n", type=int, default=3,
                        help="Anzahl der top abgerufenen Dokumente, die in den Prompt integriert werden sollen.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Gerät: 'cuda' oder 'cpu'.")
    return parser.parse_args()

def load_knowledge_base(csv_path):
    """
    Lädt die Wissensbasis aus einer CSV-Datei.
    Erwartet eine Spalte "document", die die Textpassagen enthält.
    """
    df = pd.read_csv(csv_path)
    if "document" not in df.columns:
        raise ValueError("Die CSV-Datei muss eine Spalte 'document' enthalten.")
    documents = df["document"].tolist()
    return documents

def build_bm25_retriever(documents):
    """
    Baut einen BM25-Retriever mithilfe der rank_bm25-Bibliothek.
    """
    # Tokenisiere alle Dokumente
    tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, tokenized_docs

def generate_prompt(question, retrieved_docs):
    """
    Kombiniert die Frage und die abgerufenen Dokumente zu einem Prompt.
    """
    prompt = "Frage: " + question + "\nWissensbasis:\n"
    for i, doc in enumerate(retrieved_docs, 1):
        prompt += f"{i}. {doc}\n"
    prompt += "Antwort:"
    return prompt

def generate_answer_with_t5(prompt, model_name, max_length, device):
    """
    Nutzt ein vortrainiertes T5-Modell zur Generierung einer Antwort basierend auf einem Prompt.
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    output_ids = model.generate(
        **inputs,
        num_beams=4,
        max_length=max_length,
        early_stopping=True
    )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

def main():
    args = parse_args()
    device = args.device
    print(f"Antwortgenerierung läuft auf: {device}")
    
    # Schritt 1: Wissensbasis laden
    documents = load_knowledge_base(args.knowledge_csv)
    print(f"Anzahl Wissensdokumente: {len(documents)}")
    
    # Schritt 2: BM25-Retriever aufbauen
    bm25, tokenized_docs = build_bm25_retriever(documents)
    
    # Schritt 3: Frage tokenisieren und relevante Dokumente abrufen
    tokenized_question = word_tokenize(args.question.lower())
    retrieved_docs = bm25.get_top_n(tokenized_question, documents, n=args.top_n)
    
    print("\nAbgerufene Dokumente:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i}. {doc}")
    
    # Schritt 4: Prompt generieren
    prompt = generate_prompt(args.question, retrieved_docs)
    print("\nGenerierungsprompt:")
    print(prompt)
    
    # Schritt 5: Antwort generieren mit T5
    answer = generate_answer_with_t5(prompt, args.model_name, args.max_length, device)
    print("\nGenerierte Antwort:")
    print(answer)

if __name__ == "__main__":
    main()
