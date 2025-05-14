"""
build_index.py
--------------
Skript zum Aufbau eines Retrieval-Indexes mit FAISS für das AskMeNow-Projekt.
Es lädt die Embeddings (z. B. von FAQ-Dokumenten oder Wissensbasis-Passagen) aus einer Datei,
baut einen FAISS-Index (IndexIVFFlat) auf und speichert diesen.
Beispielaufruf:
    python build_index.py --embeddings_file data/embeddings/faq_embeddings.npy \
                          --index_output data/embeddings/faiss_index.bin \
                          --nlist 100
"""

import argparse
import os
import numpy as np
import faiss

def parse_args():
    parser = argparse.ArgumentParser(description="Aufbau eines FAISS-Retrieval-Indexes für AskMeNow")
    parser.add_argument("--embeddings_file", type=str, required=True,
                        help="Pfad zur Datei mit den Embeddings (z. B. als .npy Datei)")
    parser.add_argument("--index_output", type=str, required=True,
                        help="Pfad, unter dem der FAISS-Index gespeichert werden soll")
    parser.add_argument("--nlist", type=int, default=100,
                        help="Anzahl der Cluster-Zellen im FAISS-Index (CLustering-Parameter)")
    return parser.parse_args()

def build_faiss_index(embeddings, nlist=100):
    """
    Baut einen FAISS-Index auf.

    Args:
        embeddings (np.ndarray): Array mit Shape (num_vectors, vector_dim)
        nlist (int): Anzahl der Cluster (z. B. 100)
    
    Returns:
        index: Trainierter FAISS-Index
    """
    d = embeddings.shape[1]  # Dimension der Embeddings
    quantizer = faiss.IndexFlatL2(d)  # Basisindex für L2-Distanz
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    print("Trainiere FAISS-Index...")
    index.train(embeddings)  # Index muss trainiert werden, bevor er Daten aufnehmen kann
    index.add(embeddings)    # Füge alle Embeddings hinzu
    return index

def main():
    args = parse_args()
    
    # Lade Embeddings aus der angegebenen Datei
    embeddings = np.load(args.embeddings_file)
    print(f"Embeddings geladen: {embeddings.shape}")
    
    # Baue den FAISS-Index
    index = build_faiss_index(embeddings, nlist=args.nlist)
    
    # Speichere den Index auf der Festplatte
    faiss.write_index(index, args.index_output)
    print(f"FAISS Index wurde unter '{args.index_output}' gespeichert.")

if __name__ == "__main__":
    main()
