"""
model.py
--------
Enthält zwei Modellansätze für Feedback AI:
1) LSTM-Reader mit Feedback-Anpassung (Seq2Seq-LSTM)
2) Transformer-Wrapper (z. B. GPT-2/T5) für RLHF-basierte Textgenerierung

Hinweis: Für den Transformer-Wrapper muss das Paket "transformers" installiert sein.
      (pip install transformers)
"""

import torch
import torch.nn as nn

# ============================
# Part 1: LSTM-Reader mit Feedback-Anpassung
# ============================

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_idx):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, src):
        # src: [Batch, src_len]
        embedded = self.embedding(src)  # [Batch, src_len, embed_dim]
        outputs, hidden = self.lstm(embedded)  # outputs: [Batch, src_len, hidden_dim]
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_idx):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, trg, hidden):
        # trg: [Batch, trg_len]
        embedded = self.embedding(trg)  # [Batch, trg_len, embed_dim]
        outputs, hidden = self.lstm(embedded, hidden)  # [Batch, trg_len, hidden_dim]
        predictions = self.fc(outputs)  # [Batch, trg_len, vocab_size]
        return predictions, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Führt einen Vorwärtsdurchlauf durch.
        Args:
            src: [Batch, src_len] – Input-Text
            trg: [Batch, trg_len] – Ziel-Text (für Teacher Forcing)
            teacher_forcing_ratio: Wahrscheinlichkeit, das wahre Token als nächsten Input zu verwenden
        Returns:
            outputs: [Batch, trg_len, vocab_size]
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        _, hidden = self.encoder(src)
        
        # Initialer Decoder-Input: <BOS>-Token (in trg[0])
        input_dec = trg[:, 0].unsqueeze(1)
        for t in range(1, trg_len):
            output, hidden = self.decoder(input_dec, hidden)
            outputs[:, t] = output.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(dim=2)
            input_dec = trg[:, t].unsqueeze(1) if teacher_force else top1
        return outputs

# ============================
# Part 2: Transformer-Wrapper mit RLHF (vereinfachte Demo)
# ============================
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer

    class GPT2ModelWrapper:
        """
        Wrapper für ein vortrainiertes GPT-2 Modell, das als Basis für RLHF dienen kann.
        (Hier als Demonstration; ein echtes RLHF erfordert zusätzliche Komponenten wie Reward-Modellierung.)
        """
        def __init__(self, model_name="gpt2", device="cpu"):
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.to(device)
            self.device = device
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        def generate_text(self, prompt, max_length=50, top_k=50, top_p=0.95, temperature=1.0):
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
        Wrapper für ein vortrainiertes T5-Modell zur Q&A-/Textgenerierung mit RLHF.
        (Echter RLHF würde hier zusätzliche Reward-Berechnungen und Policy-Updates erfordern.)
        """
        def __init__(self, model_name="t5-small", device="cpu"):
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(device)
            self.device = device
        
        def generate_text(self, prompt, max_length=50, num_beams=4, early_stopping=True):
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
    # Testcode für das Seq2Seq-LSTM-Modell (Dummy-Test)
    print("Test des Seq2Seq-LSTM-Modells:")
    demo_vocab_size = 1000
    demo_pad_idx = 0
    enc_demo = Encoder(demo_vocab_size, embed_dim=64, hidden_dim=128, num_layers=1, pad_idx=demo_pad_idx)
    dec_demo = Decoder(demo_vocab_size, embed_dim=64, hidden_dim=128, num_layers=1, pad_idx=demo_pad_idx)
    seq2seq_demo = Seq2Seq(enc_demo, dec_demo, demo_pad_idx)
    src_demo = torch.randint(0, demo_vocab_size, (4, 20))
    trg_demo = torch.randint(0, demo_vocab_size, (4, 10))
    outputs_demo = seq2seq_demo(src_demo, trg_demo)
    print("Output-Shape (Seq2Seq):", outputs_demo.shape)
    
    # Testcode für den GPT2-Wrapper (falls transformers installiert sind)
    try:
        print("\nTest des GPT2ModelWrapper:")
        gpt2_wrapper = GPT2ModelWrapper(model_name="gpt2", device="cpu")
        generated_text = gpt2_wrapper.generate_text("Bitte geben Sie an, wie ich Ihnen helfen kann.", max_length=50)
        print("Generierter Text (GPT-2):", generated_text)
    except ImportError as e:
        print(e)
