# __init__.py
# Dieses File kennzeichnet den Ordner als Python-Paket.
# Optional können hier zentrale Module importiert werden, um den Zugriff zu erleichtern.

from .dataset import MailDataset, create_dataloader, load_vocab
from .model import Encoder, Decoder, Seq2Seq, GPT2ModelWrapper, T5ModelWrapper
from .train import *
from .suggest import *

# Optional: Definiere __all__, um zu steuern, welche Symbole exportiert werden:
__all__ = [
    "MailDataset", "create_dataloader", "load_vocab",
    "Encoder", "Decoder", "Seq2Seq", "GPT2ModelWrapper", "T5ModelWrapper"
]
