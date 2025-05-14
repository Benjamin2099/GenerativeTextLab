"""
retriever.py
------------
Implementierung eines einfachen BM25-Retrievers für das AskMeNow-Projekt.
Dieses Modul nutzt die rank_bm25-Bibliothek, um aus einer Liste von Dokumenten
die Top-N relevanten Passagen basierend auf einer gestellten Frage abzurufen.

Installiere die rank_bm25-Bibliothek, falls noch nicht geschehen:
    pip install rank_bm25
"""

from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Stelle sicher, dass die NLTK-Tokendaten vorhanden sind (nur einmal ausführen)
nltk.download('punkt')

class BM25Retriever:
    def __init__(self, documents):
        """
        Initialisiert den BM25Retriever.
        
        Args:
            documents (list of str): Liste von Dokumenten (z. B. FAQ-Antworten oder Wissensbasis-Texten).
        """
        # Tokenisiere alle Dokumente
        self.documents = documents
        self.tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        # Erstelle den BM25-Index
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def get_top_n(self, query, n=5):
        """
        Gibt die Top-N Dokumente zurück, die für die gestellte Query am relevantesten sind.
        
        Args:
            query (str): Die Eingabefrage oder der Suchtext.
            n (int): Anzahl der zu returnierenden Dokumente.
        
        Returns:
            list of str: Die Top-N Dokumente.
        """
        # Tokenisiere die Query
        tokenized_query = word_tokenize(query.lower())
        # Hole die Top-N Dokumente basierend auf BM25-Scores
        top_n_docs = self.bm25.get_top_n(tokenized_query, self.documents, n=n)
        return top_n_docs

# Beispiel für die Nutzung des BM25Retrievers
if __name__ == "__main__":
    # Beispiel-Wissensbasis: Liste von FAQ-Antworten oder Wissensartikeln
    documents = [
        "Neuronale Netze sind Algorithmen, die von der Funktionsweise des menschlichen Gehirns inspiriert sind.",
        "Machine Learning ermöglicht es Computern, aus Daten zu lernen, ohne explizit programmiert zu sein.",
        "Deep Learning verwendet tiefe neuronale Netze, um komplexe Muster in großen Datenmengen zu erkennen.",
        "Die Relativitätstheorie beschreibt, wie Gravitation als Krümmung von Raum und Zeit verstanden wird.",
        "Künstliche Intelligenz findet in vielen Bereichen Anwendung, von der Medizin bis zur autonomen Fahrzeugsteuerung."
    ]
    
    # Initialisiere den Retriever mit der Wissensbasis
    retriever = BM25Retriever(documents)
    
    # Beispiel-Query
    query = "Wie funktionieren neuronale Netze?"
    top_docs = retriever.get_top_n(query, n=3)
    
    print("Top 3 relevante Dokumente für die Query:")
    for idx, doc in enumerate(top_docs, 1):
        print(f"{idx}. {doc}")
