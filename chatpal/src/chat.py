"""
chat.py
--------
Interaktives Inferenzskript (CLI) für ChatPal – den einfachen Chatbot.
Dieses Skript lädt das trainierte LSTM-Modell und das Vokabular, 
und ermöglicht es, über die Konsole mit dem Bot zu "chatten".
"""

import argparse
import os
import torch
import torch.nn.functional as F
from model import ChatLSTMModel
from dataset import load_vocab

def parse_args():
    parser = argparse.ArgumentParser(description="Chat with ChatPal LSTM Model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Pfad zur trainierten Modellgewichtsdatei (.pt).")
    parser.add_argument("--vocab_json", type=str, required=True,
                        help="Pfad zur vocab.json (Mapping Token->ID).")
    parser.add_argument("--max_length", type=int, default=20,
                        help="Maximale Anzahl an zu generierenden Tokens pro Antwort.")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Top-k Sampling: 1 = Greedy (Standard).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling-Temperatur (>= 1.0 = kreativer).")
    return parser.parse_args()

def sample_next_token(logits, top_k=1, temperature=1.0):
    """
    Wählt das nächste Token basierend auf den Logits.
    - Greedy (top_k==1) oder
    - Top-k Sampling (wenn top_k > 1)
    """
    scaled_logits = logits / max(temperature, 1e-8)
    if top_k <= 1:
        return torch.argmax(scaled_logits).item()
    else:
        # Top-k Sampling
        top_values, top_indices = torch.topk(scaled_logits, k=top_k)
        probs = torch.softmax(top_values, dim=-1)
        chosen_index = torch.multinomial(probs, num_samples=1).item()
        return top_indices[chosen_index].item()

def generate_response(model, prompt_ids, word2id, id2word, max_length=20, top_k=1, temperature=1.0, device="cpu"):
    """
    Generiert eine Bot-Antwort, basierend auf einem gegebenen Prompt (Liste von Token-IDs).
    
    Das Modell wird tokenweise autoregressiv abgefragt. 
    Es wird mit einem Greedy- oder Top-k-Ansatz das nächste Token ausgewählt,
    bis entweder <EOS_BOT> erreicht oder max_length Tokens generiert wurden.
    """
    model.eval()
    # Wir starten mit einer Kopie des Prompts (z.B. [<BOS_BOT>])
    generated = prompt_ids[:]  
    input_seq = torch.tensor([generated], dtype=torch.long).to(device)
    hidden = None
    
    for _ in range(max_length):
        logits, hidden = model(input_seq, hidden)  # logits: [1, seq_len, vocab_size]
        # Greife auf das Logits des letzten Tokens zu
        last_logits = logits[0, -1, :]
        next_token = sample_next_token(last_logits, top_k=top_k, temperature=temperature)
        generated.append(next_token)
        input_seq = torch.tensor([generated], dtype=torch.long).to(device)
        # Abbruchkriterium: falls <EOS_BOT> generiert wurde
        if next_token == word2id.get("<EOS_BOT>", 3):
            break
    return generated

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ChatPal startet auf {device}. Tippe 'exit' zum Beenden.\n")
    
    # Lade Vokabular
    word2id = load_vocab(args.vocab_json)
    id2word = {v: k for k, v in word2id.items()}
    vocab_size = len(word2id)
    pad_id = word2id.get("<PAD>", 0)
    
    # Modell initialisieren
    model = ChatLSTMModel(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        pad_idx=pad_id,
        dropout=0.1
    ).to(device)
    
    # Lade trainierte Gewichte
    if not os.path.exists(args.model_path):
        print(f"Modellpfad {args.model_path} nicht gefunden!")
        return
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Modell geladen.\n")
    
    # Interaktive Chat-Schleife
    print("Starte Chat. Gib 'exit' ein, um zu beenden.\n")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Chat beendet.")
            break
        if not user_input:
            continue
        
        # Tokenisiere den User-Input (hier simpel: split by whitespace)
        # In einem realen Szenario sollte dieselbe Tokenisierung wie beim Training verwendet werden.
        user_tokens = user_input.lower().split()
        # Konvertiere in Token-IDs (verwende <UNK>, wenn nicht gefunden)
        user_ids = [word2id.get(token, word2id.get("<UNK>", 1)) for token in user_tokens]
        # Optional: Du kannst auch spezielle Tokens hinzufügen, z. B. <BOS_USER> und <EOS_USER>
        BOS_USER_ID = word2id.get("<BOS_USER>", 2)
        EOS_USER_ID = word2id.get("<EOS_USER>", 3)
        user_ids = [BOS_USER_ID] + user_ids + [EOS_USER_ID]
        
        # Für die Bot-Generierung starten wir mit einem speziellen Starttoken für Bot-Antworten
        BOS_BOT_ID = word2id.get("<BOS_BOT>", 4)
        prompt_ids = [BOS_BOT_ID]
        
        # Hier kannst du entscheiden, ob du den User-Input mit einbeziehen möchtest (z.B. als Kontext).
        # Für dieses Educational-Beispiel generieren wir eine Antwort rein auf Basis des Bot-Starttokens.
        
        generated_ids = generate_response(
            model=model,
            prompt_ids=prompt_ids,
            word2id=word2id,
            id2word=id2word,
            max_length=args.max_length,
            top_k=args.top_k,
            temperature=args.temperature,
            device=device
        )
        
        # Entferne ggf. den Starttoken <BOS_BOT> und <EOS_BOT> für eine sauberere Ausgabe
        if generated_ids and generated_ids[0] == BOS_BOT_ID:
            generated_ids = generated_ids[1:]
        if "<EOS_BOT>" in [id2word.get(t, "") for t in generated_ids]:
            idx = [id2word.get(t, "") for t in generated_ids].index("<EOS_BOT>")
            generated_ids = generated_ids[:idx]
        
        bot_response = " ".join([id2word.get(t, "<UNK>") for t in generated_ids])
        print("Bot :", bot_response)
        print("-" * 40)

if __name__ == "__main__":
    main()
