"""
model.py
--------
Enthält zwei Modellansätze für die E-Mail-Vervollständigung im MailAssist-Projekt:

1) Seq2Seq-LSTM-Modell:
   - Encoder: Verarbeitet den E-Mail-Input (z. B. den unvollständigen E-Mail-Body).
   - Decoder: Generiert schrittweise den vervollständigten E-Mail-Text.
   
2) Transformer-Wrapper:
   - Nutzt ein vortrainiertes GPT-2- oder T5-Modell zur Generierung von E-Mail-Vorschlägen.
   - Ideal, um natürlich klingende, kontextreiche Textvorschläge zu erzeugen.

Hinweis: Für den Transformer-Wrapper muss das "transformers"-Paket installiert sein.
      (pip install transformers)
"""

import torch
import torch.nn as nn

# ============================
# Part 1: Seq2Seq-LSTM-Modell
# ============================

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_idx):
        """
        Encoder-Modul: Verarbeitet den E-Mail-Input.
        
        Args:
            vocab_size (int): Größe des Vokabulars.
            embed_dim (int): Dimension der Embeddings.
            hidden_dim (int): Größe des Hidden-States.
            num_layers (int): Anzahl der LSTM-Schichten.
            pad_idx (int): ID für das Padding-Token.
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, src):
        # src: [Batch, src_len]
        embedded = self.embedding(src)       # [Batch, src_len, embed_dim]
        outputs, hidden = self.lstm(embedded)  # outputs: [Batch, src_len, hidden_dim]
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_idx):
        """
        Decoder-Modul: Generiert den vervollständigten E-Mail-Text.
        
        Args:
            vocab_size (int): Größe des Vokabulars.
            embed_dim (int): Dimension der Embeddings.
            hidden_dim (int): Größe des Hidden-States.
            num_layers (int): Anzahl der LSTM-Schichten.
            pad_idx (int): ID für das Padding-Token.
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, trg, hidden):
        # trg: [Batch, trg_len]
        embedded = self.embedding(trg)       # [Batch, trg_len, embed_dim]
        outputs, hidden = self.lstm(embedded, hidden)  # [Batch, trg_len, hidden_dim]
        predictions = self.fc(outputs)         # [Batch, trg_len, vocab_size]
        return predictions, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx):
        """
        Kombiniert Encoder und Decoder zu einem Seq2Seq-Modell.
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: [Batch, src_len] – unvollständiger E-Mail-Text (Input)
            trg: [Batch, trg_len] – kompletter E-Mail-Text (Target, inkl. <BOS> am Anfang)
            teacher_forcing_ratio: Wahrscheinlichkeit, dass das wahre Token beim Decoder genutzt wird.
        Returns:
            outputs: [Batch, trg_len, vocab_size]
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        
        # Encoder: Lese den E-Mail-Input
        _, hidden = self.encoder(src)
        
        # Initialer Decoder-Input: <BOS>-Token (erster Token in der Zielsequenz)
        input_dec = trg[:, 0].unsqueeze(1)  # [Batch, 1]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input_dec, hidden)
            outputs[:, t] = output.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(dim=2)  # [Batch, 1]
            input_dec = trg[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs

# ================================
# Part 2: Transformer-Wrapper (GPT-2/T5)
# ================================
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer

    class GPT2ModelWrapper:
        """
        Ein Wrapper für ein vortrainiertes GPT-2 Modell zur E-Mail-Vervollständigung.
        """
        def __init__(self, model_name="gpt2", device="cpu"):
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.to(device)
            self.device = device
            # Falls kein pad_token vorhanden ist, setze ihn auf das eos_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        def generate_email(self, prompt, max_length=50, top_k=50, top_p=0.95, temperature=1.0):
            """
            Generiert eine E-Mail-Vervollständigung basierend auf einem Prompt.
            
            Args:
                prompt (str): Der beginnende E-Mail-Text.
                max_length (int): Maximale Länge der generierten Sequenz.
                top_k (int): Top-k Sampling.
                top_p (float): Nucleus (Top-p) Sampling.
                temperature (float): Temperatur für die Sampling-Stärke.
            Returns:
                str: Generierter E-Mail-Text.
            """
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    class T5ModelWrapper:
        """
        Ein Wrapper für ein vortrainiertes T5-Modell zur E-Mail-Vervollständigung.
        """
        def __init__(self, model_name="t5-small", device="cpu"):
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(device)
            self.device = device
        
        def generate_email(self, prompt, max_length=50, num_beams=4, early_stopping=True):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            summary_ids = self.model.generate(
                input_ids,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=early_stopping
            )
            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

except ImportError:
    class GPT2ModelWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("Bitte installiere das transformers-Paket (pip install transformers).")
    class T5ModelWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("Bitte installiere das transformers-Paket (pip install transformers).")

if __name__ == "__main__":
    # Test des Seq2Seq-LSTM-Modells mit Dummy-Daten
    print("Test des Seq2Seq-LSTM-Modells:")
    demo_vocab_size = 1000
    demo_pad_idx = 0
    encoder_demo = Encoder(demo_vocab_size, embed_dim=64, hidden_dim=128, num_layers=1, pad_idx=demo_pad_idx)
    decoder_demo = Decoder(demo_vocab_size, embed_dim=64, hidden_dim=128, num_layers=1, pad_idx=demo_pad_idx)
    seq2seq_demo = Seq2Seq(encoder_demo, decoder_demo, demo_pad_idx)
    
    # Dummy-Eingabe: [Batch, src_len] und Target: [Batch, trg_len]
    src_demo = torch.randint(0, demo_vocab_size, (4, 20))
    trg_demo = torch.randint(0, demo_vocab_size, (4, 10))
    outputs_demo = seq2seq_demo(src_demo, trg_demo)
    print("Output-Shape (Seq2Seq):", outputs_demo.shape)
    
    # Test des GPT2-Wrapper (falls transformers installiert sind)
    try:
        print("\nTest des GPT2ModelWrapper:")
        gpt2_wrapper = GPT2ModelWrapper(model_name="gpt2", device="cpu")
        email_demo = gpt2_wrapper.generate_email("Sehr geehrte Damen und Herren,", max_length=50)
        print("Generierte E-Mail (GPT-2):", email_demo)
    except ImportError as e:
        print(e)
