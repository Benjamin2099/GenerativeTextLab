"""
model.py
--------
Enthält zwei Modellansätze für AskMeNow:

1) Seq2Seq-LSTM-Modell:
   - Encoder: Verarbeitet den Input (z.B. Frage oder Kontext).
   - Decoder: Generiert schrittweise die Antwort.
   - Seq2Seq: Kombiniert Encoder und Decoder und nutzt Teacher Forcing während des Trainings.

2) Transformer-Reader (RAG):
   - T5ModelWrapper: Ein Wrapper für ein vortrainiertes T5-Modell, der den Input (z.B. als kombinierter Prompt aus Frage und abgerufenen Dokumenten) verarbeitet
     und eine faktenbasierte Antwort generiert.
     
Hinweis: Für den Transformer-Reader muss das "transformers"-Paket installiert sein.
      (pip install transformers)
"""

import torch
import torch.nn as nn

# ============================
# Part 1: Seq2Seq-LSTM-Modell (Reader)
# ============================

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_idx):
        """
        Encoder-Modul.
        
        Args:
            vocab_size (int): Vokabulargröße.
            embed_dim (int): Dimension der Embeddings.
            hidden_dim (int): Größe des Hidden-States.
            num_layers (int): Anzahl der LSTM-Schichten.
            pad_idx (int): Padding-Token-ID.
        """
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
        """
        Decoder-Modul.
        
        Args:
            vocab_size (int): Vokabulargröße.
            embed_dim (int): Dimension der Embeddings.
            hidden_dim (int): Größe des Hidden-States.
            num_layers (int): Anzahl der LSTM-Schichten.
            pad_idx (int): Padding-Token-ID.
        """
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
        """
        Kombiniert Encoder und Decoder zu einem Seq2Seq-Modell.
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        _, hidden = self.encoder(src)
        
        # Der erste Decoder-Input ist das <BOS>-Token (angenommen, trg[:, 0] enthält <BOS>)
        input_dec = trg[:, 0].unsqueeze(1)  # [Batch, 1]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input_dec, hidden)
            outputs[:, t] = output.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(dim=2)  # [Batch, 1]
            input_dec = trg[:, t].unsqueeze(1) if teacher_force else top1
        return outputs

# ============================
# Part 2: Transformer-Reader (T5) für RAG
# ============================
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    class T5ModelWrapper:
        def __init__(self, model_name="t5-small", device="cpu"):
            """
            Wrapper für ein vortrainiertes T5-Modell.
            
            Args:
                model_name (str): Name des vortrainierten Modells (z. B. "t5-small").
                device (str): "cpu" oder "cuda".
            """
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(device)
            self.device = device
        
        def generate_answer(self, prompt, max_length=50, num_beams=4, early_stopping=True):
            """
            Generiert eine Antwort basierend auf einem Prompt.
            
            Args:
                prompt (str): Der Eingabe-Prompt, z. B. "Frage: ... Wissensbasis: ... Antwort:".
                max_length (int): Maximale Länge der generierten Antwort.
                num_beams (int): Anzahl der Strahlen im Beam Search.
                early_stopping (bool): Ob das Beam Search frühzeitig gestoppt wird.
            
            Returns:
                str: Die generierte Antwort.
            """
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            summary_ids = self.model.generate(
                input_ids,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=early_stopping
            )
            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
except ImportError:
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
    
    # Testcode für den T5-Wrapper (falls transformers installiert sind)
    try:
        print("\nTest des T5ModelWrapper:")
        t5_reader = T5ModelWrapper(model_name="t5-small", device="cpu")
        prompt = "Frage: Wie funktioniert ein neuronales Netz? Wissensbasis: Neuronale Netze sind von der Funktionsweise des Gehirns inspiriert. Antwort:"
        answer = t5_reader.generate_answer(prompt, max_length=50)
        print("Generierte Antwort (T5):", answer)
    except ImportError as e:
        print(e)
