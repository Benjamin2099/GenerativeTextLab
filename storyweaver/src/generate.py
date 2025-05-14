"""
generate.py
-----------
CLI-Skript für Textgenerierung (Inferenz) mit dem StoryWeaver-LSTM-Modell.
"""

import argparse
import os
import torch
from model import LSTMTextModel
from dataset import load_vocab

def parse_args():
    """
    Kommandozeilenargumente für das Inferenzskript.
    """
    parser = argparse.ArgumentParser(description="Generate stories using StoryWeaver LSTM.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Pfad zur trainierten Modellgewichtsdatei (.pt).")
    parser.add_argument("--vocab_json", type=str, required=True,
                        help="Pfad zur vocab.json (Mapping Token->ID).")
    parser.add_argument("--prompt", type=str, default="<BOS>",
                        help="Start-Text (Prompt) für die Generierung (Standard: <BOS>).")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximale Anzahl Tokens, die generiert werden sollen.")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Top-k-Sampling (1 => greedy). Höhere Werte => mehr Varianz.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Skalierung der Logits. Höher => kreativ, niedriger => deterministischer.")
    args = parser.parse_args()
    return args

def sample_next_token(logits, top_k=1, temperature=1.0):
    """
    Wählt den nächsten Token basierend auf den vorhergesagten logits.
    - Greedy, wenn top_k=1
    - Ansonsten top-k-Sampling 
    """
    import torch
    import torch.nn.functional as F
    
    # Skalierung
    scaled_logits = logits / max(temperature, 1e-8)
    
    if top_k <= 1:
        # Greedy: Wähle argmax
        return torch.argmax(scaled_logits).item()
    else:
        # Top-k-Sampling
        top_values, top_indices = torch.topk(scaled_logits, k=top_k)
        probs = F.softmax(top_values, dim=-1)
        chosen_idx = torch.multinomial(probs, 1).item()
        return top_indices[chosen_idx].item()

def generate_text(model, word2id, id2word, prompt="<BOS>", max_length=50, top_k=1, temperature=1.0, device="cpu"):
    """
    Generiert Text mithilfe des trainierten LSTM-Modells. 
    Startet mit einem Prompt (z.B. Liste von Token-IDs).
    """
    model.eval()
    
    # Tokenisierung des Prompt-Strings in diesem einfachen Beispiel:
    # Wir erwarten, dass der Prompt ein oder mehrere "Tokens" enthält,
    # z. B. "<BOS>", "Es", "war", "einmal" etc., getrennt durch Leerzeichen.
    prompt_tokens = prompt.strip().split()
    
    # Konvertiere Prompt Tokens -> IDs
    tokens = [word2id.get(tok, word2id.get("<UNK>", 1)) for tok in prompt_tokens]
    if not tokens:
        # Falls prompt leer ist, nimm <BOS> 
        tokens = [word2id.get("<BOS>", 2)]
    
    # Input-Tensor für das Modell
    input_seq = torch.tensor([tokens], dtype=torch.long).to(device)
    hidden = None
    
    for _ in range(max_length):
        # Forward
        logits, hidden = model(input_seq, hidden)  # logits: [1, seq_len, vocab_size]
        
        # Nimm das letzte Token
        last_logits = logits[0, -1, :]  # shape: [vocab_size]
        # Wähle das nächste Token
        next_id = sample_next_token(last_logits, top_k=top_k, temperature=temperature)
        
        # Füge es der Sequenz an
        tokens.append(next_id)
        input_seq = torch.tensor([tokens], dtype=torch.long).to(device)
        
        # Abbruchkriterium: <EOS>?
        if next_id == word2id.get("<EOS>", 3):
            break
    
    # Konvertiere IDs -> Tokens
    generated_tokens = [id2word.get(t, "<UNK>") for t in tokens]
    
    return generated_tokens

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Generierung auf Gerät:", device)
    
    # 1) Vokabular laden
    word2id = load_vocab(args.vocab_json)
    id2word = {v: k for k, v in word2id.items()}
    
    vocab_size = len(word2id)
    pad_id = word2id.get("<PAD>", 0)
    
    # 2) Modell vorbereiten
    from model import LSTMTextModel
    model = LSTMTextModel(
        vocab_size=vocab_size,
        embed_dim=128,    # oder pass dir aus dem Dateinamen / Config an
        hidden_dim=256,
        num_layers=2,
        pad_idx=pad_id
    ).to(device)
    
    # Gewichte laden
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Modellgewichte geladen: {args.model_path}")
    
    # 3) Textgenerierung
    print(f"\nStarte Generierung… Prompt: '{args.prompt}'\n")
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
    
    # <BOS> / <EOS> ggf. entfernen
    if generated_ids[0] == "<BOS>":
        generated_ids = generated_ids[1:]
    if "<EOS>" in generated_ids:
        idx_eos = generated_ids.index("<EOS>")
        generated_ids = generated_ids[:idx_eos]
    
    result_text = " ".join(generated_ids)
    print("Generierter Text:\n")
    print(result_text)
    
if __name__ == "__main__":
    main()
