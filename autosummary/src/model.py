"""
model.py
--------
Enthält:
1) Seq2Seq-LSTM-Modell: Ein Encoder-Decoder-Ansatz, der für abstraktive Zusammenfassungen trainiert werden kann.
2) T5ModelWrapper: Ein Wrapper für das vortrainierte T5-Modell, das speziell auf Zusammenfassungen optimiert ist.
"""

import torch
import torch.nn as nn

# ============================
# Part 1: Seq2Seq-LSTM-Modell
# ============================

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_idx):
        """
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
        embedded = self.embedding(src)           # [Batch, src_len, embed_dim]
        outputs, hidden = self.lstm(embedded)      # outputs: [Batch, src_len, hidden_dim]
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_idx):
        """
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
        embedded = self.embedding(trg)            # [Batch, trg_len, embed_dim]
        outputs, hidden = self.lstm(embedded, hidden)  # [Batch, trg_len, hidden_dim]
        predictions = self.fc(outputs)              # [Batch, trg_len, vocab_size]
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
            src: [Batch, src_len] – Eingabetext (Artikel)
            trg: [Batch, trg_len] – Zieltext (Zusammenfassung, inkl. <BOS> am Anfang)
            teacher_forcing_ratio: Wahrscheinlichkeit, das tatsächliche Token als nächsten Input zu verwenden.
        Returns:
            outputs: [Batch, trg_len, vocab_size]
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc.out_features
        
        # Ausgabe-Tensor initialisieren
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        
        # Encoder: Lese den Input
        encoder_outputs, hidden = self.encoder(src)
        
        # Initialer Input für den Decoder: erstes Token (<BOS>)
        input_dec = trg[:, 0].unsqueeze(1)  # [Batch, 1]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input_dec, hidden)
            outputs[:, t] = output.squeeze(1)
            # Teacher Forcing: Entscheide, ob das wahre Token als Input genutzt wird
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)  # [Batch, 1]
            input_dec = trg[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs

# ============================
# Part 2: Transformer-Wrapper (T5)
# ============================
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    class T5ModelWrapper:
        """
        Ein Wrapper für das T5-Modell (z.B. t5-small) zur automatischen Zusammenfassung.
        
        Beispiel:
            model = T5ModelWrapper(model_name="t5-small", device="cuda")
            summary = model.generate_summary("Your input text here", max_length=150)
        """
        def __init__(self, model_name="t5-small", device="cpu"):
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(device)
            self.device = device
        
        def generate_summary(self, text, max_length=150, num_beams=4, early_stopping=True):
            """
            Generiert eine Zusammenfassung des Input-Textes.
            
            Args:
                text (str): Der Eingabetext, der zusammengefasst werden soll.
                max_length (int): Maximale Länge der Zusammenfassung.
                num_beams (int): Anzahl der Strahlen im Beam Search.
                early_stopping (bool): Ob der Search frühzeitig gestoppt werden soll.
                
            Returns:
                str: Die generierte Zusammenfassung.
            """
            input_ids = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            summary_ids = self.model.generate(
                input_ids,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=early_stopping
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary

except ImportError:
    # Falls transformers nicht installiert ist
    class T5ModelWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("Bitte installiere das transformers-Paket (pip install transformers)")

if __name__ == "__main__":
    # Kurzer Test für das Seq2Seq-Modell
    print("Test des Seq2Seq-LSTM-Modells:")
    vocab_size_demo = 1000
    pad_idx_demo = 0
    encoder_demo = Encoder(vocab_size_demo, embed_dim=64, hidden_dim=128, num_layers=1, pad_idx=pad_idx_demo)
    decoder_demo = Decoder(vocab_size_demo, embed_dim=64, hidden_dim=128, num_layers=1, pad_idx=pad_idx_demo)
    seq2seq_demo = Seq2Seq(encoder_demo, decoder_demo, pad_idx_demo)
    
    # Dummy-Daten
    src_demo = torch.randint(0, vocab_size_demo, (4, 20))
    trg_demo = torch.randint(0, vocab_size_demo, (4, 10))
    outputs_demo = seq2seq_demo(src_demo, trg_demo)
    print("Output-Shape:", outputs_demo.shape)
    
    # Test für den T5-Wrapper (falls transformers installiert sind)
    try:
        print("\nTest des T5-ModelWrapper:")
        t5_wrapper = T5ModelWrapper(model_name="t5-small", device="cpu")
        summary_demo = t5_wrapper.generate_summary("AutoSummary is a project for automatic text summarization.", max_length=50)
        print("Generierte Zusammenfassung:", summary_demo)
    except ImportError as e:
        print(e)
