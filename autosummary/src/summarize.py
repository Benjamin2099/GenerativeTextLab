"""
summarize.py
------------
CLI-Inferenzskript für das AutoSummary-Projekt zur Generierung von Zusammenfassungen.
Verwendet ein Seq2Seq-LSTM-Modell (Encoder-Decoder), um aus einem Artikel eine Zusammenfassung zu erzeugen.
Beispielaufruf:
    python summarize.py --model_path ../models/autosummary_epoch3_valloss2.1234.pt \
                        --vocab_json ../data/processed/vocab.json \
                        --article "Your long article text here" \
                        --max_length 150
"""

import argparse
import os
import torch

# Importiere die Modellkomponenten und Vokabular-Ladefunktion aus dem src-Verzeichnis
from model import Encoder, Decoder, Seq2Seq
from dataset import load_vocab

def parse_args():
    parser = argparse.ArgumentParser(description="Generate summary using AutoSummary Seq2Seq model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Pfad zur gespeicherten Modellgewichtsdatei (.pt)")
    parser.add_argument("--vocab_json", type=str, required=True,
                        help="Pfad zur vocab.json-Datei (Mapping Token->ID)")
    parser.add_argument("--article", type=str, required=True,
                        help="Der Artikeltext, der zusammengefasst werden soll")
    parser.add_argument("--max_length", type=int, default=150,
                        help="Maximale Länge der generierten Zusammenfassung (in Tokens)")
    args = parser.parse_args()
    return args

def generate_summary(encoder, decoder, article_ids, word2id, id2word, max_length, device):
    """
    Generiert eine Zusammenfassung für einen Artikel.
    
    Args:
        encoder: Das Encoder-Modul des Seq2Seq-Modells.
        decoder: Das Decoder-Modul des Seq2Seq-Modells.
        article_ids: Liste von Token-IDs, die den Artikel repräsentieren.
        word2id: Mapping von Token zu ID.
        id2word: Inverses Mapping von ID zu Token.
        max_length: Maximale Länge der generierten Zusammenfassung.
        device: "cuda" oder "cpu".
        
    Returns:
        Eine Liste von Token-IDs der generierten Zusammenfassung.
    """
    encoder.eval()
    decoder.eval()
    
    # Konvertiere den Artikel in einen Tensor: [1, article_len]
    src = torch.tensor([article_ids], dtype=torch.long).to(device)
    
    # Encoder: Lese den Artikel und erhalte den Hidden-State
    _, hidden = encoder(src)
    
    # Initialer Decoder-Input: <BOS> Token für Zusammenfassungen
    bos_id = word2id.get("<BOS>", 2)
    input_dec = torch.tensor([[bos_id]], dtype=torch.long).to(device)
    
    generated_ids = [bos_id]
    
    # Generiere schrittweise Tokens (Greedy-Decoding)
    for _ in range(max_length):
        # Decoder: Vorhersage basierend auf dem aktuellen Input und hidden State
        output, hidden = decoder(input_dec, hidden)
        # output: [1, 1, vocab_size] -> Wähle das Token mit dem höchsten Score
        next_id = output.argmax(dim=2).item()
        # Abbruch: Wenn <EOS> generiert wird, beende die Schleife
        if next_id == word2id.get("<EOS>", 3):
            break
        generated_ids.append(next_id)
        # Aktualisiere input für den Decoder
        input_dec = torch.tensor([[next_id]], dtype=torch.long).to(device)
    
    return generated_ids

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Generierung läuft auf: {device}")
    
    # Vokabular laden
    word2id = load_vocab(args.vocab_json)
    id2word = {v: k for k, v in word2id.items()}
    vocab_size = len(word2id)
    pad_id = word2id.get("<PAD>", 0)
    
    # Modellarchitektur aufbauen
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    
    encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, pad_id)
    decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, pad_id)
    model = Seq2Seq(encoder, decoder, pad_id).to(device)
    
    # Lade trainierte Gewichte
    if not os.path.exists(args.model_path):
        print(f"Modellpfad nicht gefunden: {args.model_path}")
        return
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Modellgewichte erfolgreich geladen.")
    
    # Artikeltext: Verwende den Input-Text (Eingabe als String)
    article_text = args.article.strip().lower()
    # Hier wird eine einfache Tokenisierung per split() verwendet.
    # In der Praxis sollte die gleiche Tokenisierung wie bei der Vorverarbeitung genutzt werden.
    article_tokens = article_text.split()
    # Konvertiere die Tokens in IDs (nutze <UNK> falls Token nicht gefunden wird)
    article_ids = [word2id.get(token, word2id.get("<UNK>", 1)) for token in article_tokens]
    
    # Generiere die Zusammenfassung
    generated_ids = generate_summary(encoder, decoder, article_ids, word2id, id2word, args.max_length, device)
    # Entferne das initiale <BOS>-Token für eine sauberere Ausgabe
    if generated_ids and generated_ids[0] == word2id.get("<BOS>", 2):
        generated_ids = generated_ids[1:]
    summary = " ".join([id2word.get(token, "<UNK>") for token in generated_ids])
    
    print("\nGenerierte Zusammenfassung:")
    print(summary)

if __name__ == "__main__":
    main()
