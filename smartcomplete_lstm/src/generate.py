"""
generate.py
-----------
CLI-Skript für die Textgenerierung (Inference) mit dem SmartComplete-LSTM-Modell.
"""

import argparse
import os
import torch
import torch.nn.functional as F
from model import LSTMTextModel
from dataset import load_vocab

def parse_args():
    """
    Kommandozeilenargumente:
      --model_path  -> Pfad zur .pt-Datei mit trainierten Gewichten
      --vocab_json  -> Pfad zum vocab.json
      --prompt      -> Start-Text, der vervollständigt werden soll
      --max_length  -> Maximale Anzahl zu generierender Tokens
      --top_k       -> Top-K-Sampling, um mehr Varianz reinzubringen
      --temperature -> Temperaturwert (>= 0.0) für Sampling
    """
    parser = argparse.ArgumentParser(description="Generate text using SmartComplete LSTM model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model weights (.pt).")
    parser.add_argument("--vocab_json", type=str, required=True,
                        help="Path to vocab.json (token->ID mapping).")
    parser.add_argument("--prompt", type=str, default="<BOS>",
                        help="Prompt text to start generation (default: <BOS>).")
    parser.add_argument("--max_length", type=int, default=20,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-k sampling size. (If <=1 => Greedy)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (>= 0.0).")
    args = parser.parse_args()
    return args

def sample_next_token(logits, top_k=5, temperature=1.0):
    """
    Bestimmt das nächste Token basierend auf den Logits.
    
    - Wenn top_k <= 1, wird 'greedy' gewählt (Argmax).
    - Ansonsten Top-k-Sampling mit Softmax und Temperatur.
    """
    if top_k <= 1:
        # Greedy
        next_id = torch.argmax(logits, dim=-1).item()
        return next_id
    else:
        # Top-k-Sampling
        # 1) Skaliere Logits mit Temperatur
        scaled_logits = logits / max(temperature, 1e-8)
        
        # 2) Finde top_k
        top_values, top_indices = torch.topk(scaled_logits, k=top_k)
        
        # 3) Konvertiere in Wahrscheinlichkeiten
        probs = F.softmax(top_values, dim=-1)
        
        # 4) Zufällige Auswahl basierend auf den top_k-Wahrscheinlichkeiten
        chosen_idx = torch.multinomial(probs, 1).item()
        
        next_id = top_indices[chosen_idx].item()
        return next_id

def generate_text(model, word2id, id2word, prompt="<BOS>", max_length=20, top_k=5, temperature=1.0, device="cpu"):
    """
    Generiert Text Token für Token basierend auf dem prompt.
    """
    model.eval()
    
    # Tokenisiere prompt
    # Hier: vereinfachter Ansatz -> wir splitten prompt per Whitespace
    #       bzw. ersetzen <BOS>, <EOS> bei Bedarf.
    # In einem richtigen System würdest du dieselbe Tokenisierung anwenden
    # wie beim Training. Hier nur als Beispiel:
    if prompt == "<BOS>":
        tokens = [word2id.get("<BOS>", 2)]
    else:
        prompt_tokens = prompt.lower().split()
        tokens = [word2id.get(t, word2id.get("<UNK>", 1)) for t in prompt_tokens]
    
    # input_seq shape: [1, prompt_length]
    input_seq = torch.tensor([tokens], dtype=torch.long).to(device)
    hidden = None
    
    for _ in range(max_length):
        # Forward
        logits, hidden = model(input_seq, hidden)  # logits: [1, seq_len, vocab_size]
        
        # Nimm das letzte Token
        last_logits = logits[0, -1, :]  # shape: [vocab_size]
        
        # Wähle das nächste Token (Greedy oder Top-k-Sampling)
        next_id = sample_next_token(last_logits, top_k=top_k, temperature=temperature)
        
        # Hänge das Token an
        tokens.append(next_id)
        
        # Aktualisiere input_seq
        input_seq = torch.tensor([tokens], dtype=torch.long).to(device)
        
        # Abbruchkriterium: falls <EOS> auftritt
        if next_id == word2id.get("<EOS>", 3):
            break
    
    # Konvertiere IDs zurück zu Wörtern
    generated_words = [id2word.get(t, "<UNK>") for t in tokens]
    return generated_words

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1) Vokabular laden
    word2id = load_vocab(args.vocab_json)
    id2word = {v:k for k,v in word2id.items()}
    vocab_size = len(word2id)
    
    # 2) Modell vorbereiten
    print("Lade Modellgewichte:", args.model_path)
    
    # Annahme: Du weißt aus dem Training, welche hidden_dim, embed_dim etc. du hattest
    # oder du liest das aus dem Dateinamen / einer Config. Hier machen wir Hardcode / Demo:
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    pad_idx = word2id.get("<PAD>", 0)
    dropout = 0.1
    
    model = LSTMTextModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pad_idx=pad_idx,
        dropout=dropout
    ).to(device)
    
    # State-Dict laden
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 3) Generieren
    print(f"\nGeneriere Text... Prompt: '{args.prompt}'\n")
    generated_ids = generate_text(
        model=model,
        word2id=word2id,
        id2word=id2word,
        prompt=args.prompt,
        max_length=args.max_length,
        top_k=args.top_k,
        temperature=args.temperature,
        device=device
    )
    
    # 4) Ausgabe
    # Entferne ggf. <BOS> oder <EOS>
    if generated_ids and generated_ids[0] == "<BOS>":
        generated_ids = generated_ids[1:]
    if "<EOS>" in generated_ids:
        idx_eos = generated_ids.index("<EOS>")
        generated_ids = generated_ids[:idx_eos]
    
    # Zusammensetzen
    output_text = " ".join(generated_ids)
    print("Generierter Text:\n", output_text)
    
if __name__ == "__main__":
    main()
