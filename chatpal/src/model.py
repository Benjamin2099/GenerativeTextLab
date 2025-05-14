"""
model.py
--------
Enthält:
1) ChatLSTMModel: Einfache LSTM-Architektur für Chat-Eingaben (User -> Bot).
2) GPT2ModelWrapper: Ein Wrapper um ein GPT-2 (Transformer) Modell aus Hugging Face
   (nur relevant, wenn du Transformers einsetzen möchtest).
   
Hinweis: Für GPT-2 brauchst du 'transformers' installiert:
   pip install transformers
"""

import torch
import torch.nn as nn

# =============== TEIL A: LSTM MODELL ===============
class ChatLSTMModel(nn.Module):
    """
    Ein simples LSTM-basiertes Modell für Chatbot-Antworten.
    
    Idee: Input = User-Sequenz (Tokens),
          Output = logit pro Zeitschritt (zur Vorhersage der Bot-Sequenz).

    In einem grundlegenden Setting:
      - Du kannst pro Batch user_seq => LSTM => Logits
      - Dann berechnest du den CrossEntropyLoss gegen bot_seq.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1,
                 pad_idx=0, dropout=0.0):
        """
        Args:
          vocab_size (int): Anzahl der Tokens (inkl. Sondertokens)
          embed_dim  (int): Dimension der Embeddings
          hidden_dim (int): Größe des Hidden-States im LSTM
          num_layers (int): Anzahl gestackter LSTM-Schichten
          pad_idx    (int): Token-ID für <PAD> (Embedding ignoriert diesen)
          dropout    (float): Dropout zwischen LSTM-Layern, ab num_layers>1
        """
        super(ChatLSTMModel, self).__init__()
        
        # 1) Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )
        
        # 2) LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0.0
        )
        
        # 3) FC-Layer (vom Hidden-State zum Vokabular)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, user_seq, hidden=None):
        """
        user_seq: [Batch, SeqLen] => Token-IDs
        hidden: optionales (h_0, c_0)
        Returns: 
          logits [Batch, SeqLen, vocab_size],
          hidden (h_n, c_n)
        """
        emb = self.embedding(user_seq)           # -> [B, SeqLen, embed_dim]
        out, hidden = self.lstm(emb, hidden)     # -> [B, SeqLen, hidden_dim]
        logits = self.fc(out)                    # -> [B, SeqLen, vocab_size]
        return logits, hidden
    
    def init_hidden(self, batch_size, device='cpu'):
        """
        Erzeugt Null-Inits für (h_0, c_0).
        """
        num_layers = self.lstm.num_layers
        hidden_dim = self.lstm.hidden_size
        h_0 = torch.zeros(num_layers, batch_size, hidden_dim, device=device)
        c_0 = torch.zeros(num_layers, batch_size, hidden_dim, device=device)
        return (h_0, c_0)


# =============== TEIL B: GPT-2 / TRANSFORMER MODELL ===============
# Dieser Teil erfordert: pip install transformers
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch.nn.functional as F
    
    class GPT2ModelWrapper:
        """
        Ein Wrapper für ein GPT-2 Modell via Hugging Face Transformers.
        Ermöglicht einfache Textgenerierung (Chat-Bot-like), 
        indem man prompt -> generate().
        """
        def __init__(self, model_name='gpt2', device='cpu'):
            """
            Args:
                model_name (str): z. B. 'gpt2', 'gpt2-medium', 'gpt2-large'
                device (str): 'cpu' oder 'cuda'
            """
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.to(device)
            self.device = device
            
            # Setze ein pad_token, wenn GPT-2 keins hat
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        def generate_text(self, prompt, max_length=50, top_k=0, top_p=0.9, temperature=1.0):
            """
            Generiert Text (Bot-Antwort) auf Basis eines Prompt.
            
            do_sample=True => stochastische Auswahl (random),
            top_k=0 => kein Top-k
            top_p=0.9 => Nucleus-Sampling
            temperature => Skaliert Kreativität
            """
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=True,
                    top_k=top_k if top_k>0 else None,
                    top_p=top_p,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        def generate_text_greedy(self, prompt, max_length=50):
            """
            Generiert Text mithilfe von Greedy-Decoding (kein Sampling).
            """
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

except ImportError:
    # Falls 'transformers' nicht installiert ist oder du es nicht brauchst,
    # wird GPT2ModelWrapper unbenutzbar.
    class GPT2ModelWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("transformers (GPT-2) ist nicht installiert. Bitte `pip install transformers` ausführen.")


if __name__ == "__main__":
    # Kurzer Test oder Demo
    print("Dies ist model.py - Enthält ChatLSTMModel und GPT2ModelWrapper.")
    print("Für echte Tests bitte train.py oder chat.py nutzen.")
