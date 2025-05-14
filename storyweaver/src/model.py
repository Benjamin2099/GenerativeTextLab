"""
model.py
--------
Enthält:
1) LSTMTextModel: Einfaches LSTM-Sprachmodell für StoryWeaver
2) GPT2ModelWrapper: Wrapper-Klasse für ein vortrainiertes GPT-2 / Transformer-Modell

Hinweis:
- Für GPT-2 musst du 'transformers' installiert haben:
  pip install transformers
"""

import torch
import torch.nn as nn

# =============== TEIL A: LSTM MODELL ===============
class LSTMTextModel(nn.Module):
    """
    Ein einfaches LSTM-Sprachmodell:
     - Embedding
     - LSTM (ein oder mehrere Schichten)
     - FC-Layer, der pro Zeitschritt die logits über das gesamte Vokabular ausgibt.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, pad_idx=0, dropout=0.0):
        """
        Args:
            vocab_size (int): Größe des Vokabulars (inkl. Sondertokens)
            embed_dim (int): Dimension der Embeddings
            hidden_dim (int): Größe des versteckten Zustands in LSTM
            num_layers (int): Anzahl gestackter LSTM-Schichten
            pad_idx (int): Token-ID für <PAD> (wird in Embeddings ignoriert)
            dropout (float): Dropout (wirkt erst ab num_layers > 1)
        """
        super(LSTMTextModel, self).__init__()
        
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
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 3) FC-Layer für die Projektion von hidden_dim auf vocab_size
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        """
        x: [Batch, SeqLen] mit Token-IDs
        hidden: (h_0, c_0) optional, wenn man manuell den hidden State steuern will.
        Returns:
            logits: [Batch, SeqLen, vocab_size]
            hidden: (h_n, c_n)
        """
        emb = self.embedding(x)         # -> [Batch, SeqLen, embed_dim]
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)           # -> [Batch, SeqLen, vocab_size]
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
# Hinweis: Für dieses Segment benötigst du: pip install transformers
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch.nn.functional as F

    class GPT2ModelWrapper:
        """
        Eine Wrapper-Klasse für ein vortrainiertes GPT-2 Modell 
        (z. B. 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl' oder 
        eigene Fine-Tuning-Checkpoints).
        
        Usage:
          model = GPT2ModelWrapper(model_name='gpt2')
          text = model.generate_text("Es war einmal", max_length=50)
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
            
            # Optional: Manche GPT-2 Tokenizer haben kein pad_token -> legen wir fest
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        def generate_text(self, prompt, max_length=50, top_k=0, top_p=0.9, temperature=1.0):
            """
            Generiert Text auf Basis eines Prompts mit Sampling.
            Args:
                prompt (str): Starttext (z. B. "Es war einmal")
                max_length (int): maximale Tokenlänge
                top_k (int): Top-k-Sampling; 0 = kein top-k
                top_p (float): Nucleus-Sampling-Schwelle
                temperature (float): Skalierung der Logits; 
                                     höher => kreativ, niedriger => deterministischer
            Returns:
                str: Generierter Text
            """
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # GPT-2 generate() Methode
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=True,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p,
                    temperature=temperature,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return generated_text
        
        def generate_text_greedy(self, prompt, max_length=50):
            """
            Generiert Text mit Greedy-Decoding (keine zufällige Auswahl).
            """
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=False,  # greedy
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

except ImportError:
    # Falls 'transformers' nicht installiert ist, 
    # oder du es nicht verwenden möchtest, gibt es eine simple Info aus.
    class GPT2ModelWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("transformers-Paket nicht installiert. Bitte 'pip install transformers' ausführen.")
    
    # Du könntest hier auch "pass" benutzen, falls du den Transformer-Teil nicht benötigst.
