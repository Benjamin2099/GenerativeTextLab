# __init__.py
# Dieses File markiert den Ordner als Python-Paket für AskMeNow und ermöglicht zentrale Importe.

from .dataset import QADataset, create_qa_dataloader, load_vocab
from .retriever import BM25Retriever
from .model import Encoder, Decoder, Seq2Seq, T5ModelWrapper  # Alternativ: auch GPT2ModelWrapper falls benötigt
from .train import *
from .ask import *
from .build_index import *

# Optionale Definition von __all__, um zu steuern, welche Namen exportiert werden:
__all__ = [
    "QADataset", "create_qa_dataloader", "load_vocab",
    "BM25Retriever",
    "Encoder", "Decoder", "Seq2Seq", "T5ModelWrapper",
]
