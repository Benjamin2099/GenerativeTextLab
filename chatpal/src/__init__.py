# __init__.py
# Dieser Code markiert das Verzeichnis als Python-Paket.
# Optional: Wir exportieren hier zentrale Klassen/Funktionen, 
# sodass sie direkt importiert werden können.

from .dataset import ChatDataset, create_dataloader, load_vocab
from .model import ChatLSTMModel, GPT2ModelWrapper

# Falls weitere Module vorhanden sind, können sie hier ergänzt werden.
