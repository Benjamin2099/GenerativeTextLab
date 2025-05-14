# __init__.py
# Dieses File markiert das 'src'-Verzeichnis als Python-Paket.
# Optional können hier zentrale Klassen oder Funktionen importiert werden,
# um sie für den Rest des Projekts einfacher verfügbar zu machen.

from .dataset import ChatDataset, create_dataloader, load_vocab
from .model import ChatLSTMModel, Seq2Seq, Encoder, Decoder, GPT2ModelWrapper  # falls verwendet
from .train import *
from .summarize import *

# Du kannst hier auch nur die wichtigen Module exportieren,
# z. B.:
# __all__ = ["ChatDataset", "create_dataloader", "load_vocab", "ChatLSTMModel", "Seq2Seq"]
