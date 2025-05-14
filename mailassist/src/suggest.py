"""
suggest.py
----------
CLI-Inferenzskript für MailAssist – Generierung von E-Mail-/Textvorschlägen.
Das Skript lädt ein vortrainiertes Seq2Seq-LSTM-Modell und das Vokabular, 
tokenisiert den Input-Text und generiert mithilfe des Decoders einen vervollständigten Text.

Beispielaufruf:
    python suggest.py --model_path models/mailassist_epoch3_valloss2.1357.pt \
                      --vocab_json data/processed/vocab.json \
                      --input_text "sehr geehrte damen und herren, ich möchte sie darüber informieren" \
                      --max_length 30
"""

import argparse
import os
import torch
import torch.nn.functional as F

# Importiere notwendige Module aus src
from dataset import load_vocab
from model import Encoder, Decoder, Seq2Seq

def parse_args():
    parser = argparse.ArgumentParser(description="Generiere E-Mail-/Textvorschläge mit MailAssist")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Pfad zur gespeicherten Modellgewichtsdatei (.pt)")
    parser.add_argument("--vocab_json", type=str, required=True,
                        help="Pfad zur vocab.json-Datei (Mapping Token -> ID)")
    parser.add_argument("--input_text", type=str, required=True,
                        help="Der unvollständige E-Mail-Text, der vervollständigt werden soll")
    parser.add_argument("--max_length", type=int, default=30,
                        help="Maximale Anzahl generierter Tokens (ohne das initiale <BOS>)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Gerät: 'cuda' oder 'cpu'")
    args = parser.parse_args()
    return args

def tokenize_input(text):
    """
    Eine einfache Tokenisierung: Splitting des Textes per Whitespace.
    In einem echten Projekt sollte dieselbe Tokenisierung wie in der Vorverarbeitung verwendet werden.
    """
    return text.strip().lower().split()

def generate_suggestion(model, encoder, decoder, input_ids, word2id, id2word, max_length, device):
    """
    Generiert einen E-Mail-/Textvorschlag basierend auf einem gegebenen Input.
    
    Args:
        model: Das vollständige Seq2Seq-Modell (Encoder und Decoder).
        encoder: Der Encoder-Teil des Modells.
        decoder: Der Decoder-Teil des Modells.
        input_ids: Liste von Token-IDs, die den Input repräsentieren.
        word2id: Mapping von Token zu ID.
        id2word: Inverses Mapping.
        max_length: Maximale Anzahl an zu generierenden Tokens.
        device: "cuda" oder "cpu".
        
    Returns:
        Vorschlag als lesbarer Text (String).
    """
    model.eval()
    # Konvertiere den Input in einen Tensor: [1, seq_len]
    src_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Encoder: Verarbeite den Input
    _, hidden = encoder(src_tensor)
    
    # Initialer Decoder-Input: Das <BOS>-Token aus der Zielsequenz
    bos_id = word2id.get("<BOS>", 2)
    input_dec = torch.tensor([[bos_id]], dtype=torch.long).to(device)
    generated_ids = [bos_id]
    
    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = decoder(input_dec, hidden)  # output: [1, 1, vocab_size]
            next_id = output.argmax(dim=2).item()
            if next_id == word2id.get("<EOS>", 3):
                break
            generated_ids.append(next_id)
            input_dec = torch.tensor([[next_id]], dtype=torch.long).to(device)
    
    # Konvertiere die generierten IDs in lesbaren Text und entferne ggf. das <BOS>-Token
    generated_tokens = [id2word.get(i, "<UNK>") for i in generated_ids]
    if generated_tokens and generated_tokens[0] == "<BOS>":
        generated_tokens = generated_tokens[1:]
    return " ".join(generated_tokens)

def main():
    args = parse_args()
    device = args.device
    print(f"Generierung läuft auf: {device}")
    
    # Vokabular laden
    word2id = load_vocab(args.vocab_json)
    id2word = {v: k for k, v in word2id.items()}
    vocab_size = len(word2id)
    pad_id = word2id.get("<PAD>", 0)
    
    # Modellarchitektur: Seq2Seq-LSTM
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
    
    # Tokenisiere den Input-Text
    tokens = tokenize_input(args.input_text)
    input_ids = [word2id.get(token, word2id.get("<UNK>", 1)) for token in tokens]
    print("\nInput-Text:", args.input_text)
    print("Tokenisierte IDs:", input_ids)
    
    # Generiere den Vorschlag
    suggestion = generate_suggestion(model, encoder, decoder, input_ids, word2id, id2word, args.max_length, device)
    print("\nGenerierter E-Mail-/Textvorschlag:")
    print(suggestion)

if __name__ == "__main__":
    main()
