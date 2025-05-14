"""
generate.py
-----------
CLI-Inferenzskript für Feedback AI – Textgenerierung mit Feedback.
Dieses Skript lädt ein vortrainiertes Seq2Seq-Modell und das Vokabular,
tokenisiert den eingegebenen unvollständigen Text und generiert einen 
Vorschlag. Die Feedback-Informationen flossen bereits in das Training ein.
Beispielaufruf:
    python generate.py --model_path models/feedback_ai_epoch3_valloss2.1357.pt \
                       --vocab_json data/processed/vocab.json \
                       --input_text "sehr geehrte damen und herren, ich möchte sie" \
                       --max_length 30
"""

import argparse
import os
import torch

# Importiere notwendige Module aus src (hier nutzen wir unser Seq2Seq-Modell)
from dataset import load_vocab
from model import Encoder, Decoder, Seq2Seq

def parse_args():
    parser = argparse.ArgumentParser(description="Generiere Textvorschläge mit dem Feedback AI-Modell.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Pfad zur gespeicherten Modellgewichtsdatei (.pt)")
    parser.add_argument("--vocab_json", type=str, required=True,
                        help="Pfad zur vocab.json-Datei (Mapping Token->ID)")
    parser.add_argument("--input_text", type=str, required=True,
                        help="Der unvollständige Text, der vervollständigt werden soll")
    parser.add_argument("--max_length", type=int, default=30,
                        help="Maximale Anzahl generierter Tokens (ohne das initiale <BOS>)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Gerät: 'cuda' oder 'cpu'")
    return parser.parse_args()

def tokenize_input(text):
    """
    Eine einfache Tokenisierung: Splitting per Whitespace.
    (In einem realen System sollte dieselbe Tokenisierung wie bei der Vorverarbeitung verwendet werden.)
    """
    return text.strip().lower().split()

def generate_text(model, encoder, decoder, input_ids, word2id, id2word, max_length, device):
    """
    Generiert einen Textvorschlag basierend auf dem gegebenen Input.
    
    Args:
        model: Das Seq2Seq-Modell.
        encoder: Der Encoder-Teil.
        decoder: Der Decoder-Teil.
        input_ids: Liste von Token-IDs des Input-Texts.
        word2id: Mapping von Token zu ID.
        id2word: Inverses Mapping.
        max_length: Maximale Länge der generierten Sequenz.
        device: "cuda" oder "cpu".
        
    Returns:
        Generierter Textvorschlag als String.
    """
    model.eval()
    # Konvertiere den Input in einen Tensor [1, seq_len]
    src_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Encoder: Verarbeite den Input
    _, hidden = encoder(src_tensor)
    
    # Initialer Decoder-Input: <BOS>-Token der Zielsequenz
    bos_id = word2id.get("<BOS>", 2)
    input_dec = torch.tensor([[bos_id]], dtype=torch.long).to(device)
    generated_ids = [bos_id]
    
    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = decoder(input_dec, hidden)
            # Greedy-Decoding: Wähle das Token mit der höchsten Wahrscheinlichkeit
            next_id = output.argmax(dim=2).item()
            # Abbruch, falls <EOS> generiert wird
            if next_id == word2id.get("<EOS>", 3):
                break
            generated_ids.append(next_id)
            input_dec = torch.tensor([[next_id]], dtype=torch.long).to(device)
    
    # Konvertiere generierte Token-IDs in lesbaren Text
    generated_tokens = [id2word.get(i, "<UNK>") for i in generated_ids]
    # Entferne ggf. das initiale <BOS>-Token
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
    
    # Modellarchitektur: Seq2Seq-LSTM-Modell
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
    
    # Generiere den Textvorschlag
    generated_text = generate_text(model, encoder, decoder, input_ids, word2id, id2word, args.max_length, device)
    print("\nGenerierter Textvorschlag:")
    print(generated_text)

if __name__ == "__main__":
    main()
