"""
transfer.py
-----------
CLI-Skript für den Stil-Transfer im StyleShift-Projekt.
Das Skript lädt ein vortrainiertes Seq2Seq-LSTM-Modell und das Vokabular,
konvertiert den eingegebenen modernen Text in Token-IDs, und generiert dann
eine stilisierte Version des Textes, indem es den Decoder autoregressiv abfragt.

Beispielaufruf:
    python transfer.py --model_path models/styleshift_epoch3_valloss2.1234.pt \
                       --vocab_json data/processed/vocab.json \
                       --input_text "ich freue mich auf den sommer" \
                       --max_length 20
"""

import argparse
import os
import torch
import torch.nn.functional as F
from dataset import load_vocab
from model import Encoder, Decoder, Seq2Seq

def parse_args():
    parser = argparse.ArgumentParser(description="Stil-Transfer mit dem StyleShift Seq2Seq-Modell")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Pfad zur gespeicherten Modellgewichtsdatei (.pt)")
    parser.add_argument("--vocab_json", type=str, required=True,
                        help="Pfad zur vocab.json (Mapping Token->ID)")
    parser.add_argument("--input_text", type=str, required=True,
                        help="Der moderne Text, der in den Zielstil übertragen werden soll")
    parser.add_argument("--max_length", type=int, default=20,
                        help="Maximale Anzahl an generierten Tokens (ohne <BOS>)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Gerät: 'cuda' oder 'cpu'")
    args = parser.parse_args()
    return args

def tokenize_input(text):
    """
    Eine einfache Tokenisierung des Input-Texts.
    Hier wird der Text per whitespace gesplittet. In einem realen Projekt
    sollte dieselbe Tokenisierung genutzt werden wie bei der Vorverarbeitung.
    """
    return text.strip().lower().split()

def generate_styled_text(model, encoder, decoder, input_ids, word2id, id2word, max_length, device):
    """
    Generiert eine stilisierte Version des Input-Texts.
    
    Args:
        model: Das komplette Seq2Seq-Modell (hier zur Vereinfachung nicht direkt genutzt).
        encoder: Der Encoder des Seq2Seq-Modells.
        decoder: Der Decoder des Seq2Seq-Modells.
        input_ids: Liste von Token-IDs des Input-Textes.
        word2id: Mapping von Token zu ID.
        id2word: Inverses Mapping.
        max_length: Maximale Anzahl an zu generierenden Tokens.
        device: "cuda" oder "cpu".
    
    Returns:
        Der generierte Text (String) im Zielstil.
    """
    model.eval()
    # Konvertiere den Input in einen Tensor: [1, seq_len]
    src_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Encoder: Verarbeite den Input
    _, hidden = encoder(src_tensor)
    
    # Initialer Decoder-Input: Beginne mit dem <BOS>-Token für den Zielstil
    bos_id = word2id.get("<BOS>", 2)
    input_dec = torch.tensor([[bos_id]], dtype=torch.long).to(device)
    generated_ids = [bos_id]
    
    # Autoregressive Generierung mittels Greedy-Decoding
    for _ in range(max_length):
        output, hidden = decoder(input_dec, hidden)  # output: [1, 1, vocab_size]
        next_id = output.argmax(dim=2).item()
        # Abbruch: Wenn <EOS> generiert wird, beenden wir die Schleife
        if next_id == word2id.get("<EOS>", 3):
            break
        generated_ids.append(next_id)
        input_dec = torch.tensor([[next_id]], dtype=torch.long).to(device)
    
    # Konvertiere die generierten Token-IDs in lesbaren Text
    generated_tokens = [id2word.get(i, "<UNK>") for i in generated_ids]
    # Entferne das initiale <BOS>-Token aus der Ausgabe
    if generated_tokens and generated_tokens[0] == "<BOS>":
        generated_tokens = generated_tokens[1:]
    return " ".join(generated_tokens)

def main():
    args = parse_args()
    device = args.device
    print(f"Stiltransfer läuft auf: {device}")
    
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
    
    print("\nInput-Text (modern):", args.input_text)
    print("Tokenisierte IDs:", input_ids)
    
    # Generiere den stilisierten Text
    styled_text = generate_styled_text(model, encoder, decoder, input_ids, word2id, id2word, args.max_length, device)
    
    print("\nStilisierte Version:")
    print(styled_text)

if __name__ == "__main__":
    main()
