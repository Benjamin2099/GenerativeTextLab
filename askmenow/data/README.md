# AskMeNow – Wissens-Q&A mit Retrieval-Augmented Generation (RAG)

Dieses Projekt entwickelt ein FAQ- und Wissens-Q&A-System, das mithilfe von Retrieval-Augmented Generation (RAG) relevante Textpassagen aus einer Wissensdatenbank extrahiert und in die Generierung von Antworten einbindet. Dieses Dokument bietet einen Überblick über die verwendeten Wissensquellen, die Retrieval-Methoden und die zu beachtenden Lizenzbedingungen.

---

## Wissensquellen

- **Öffentliche Daten:**  
  - **Wikipedia:**  
    Inhalte werden unter der [CC BY-SA](https://creativecommons.org/licenses/by-sa/3.0/) Lizenz bereitgestellt.
  - **Open-Data-Datenbanken:**  
    Daten, die frei verfügbar sind, z. B. aus Open-Data-Projekten oder Public-Domain-Materialien.
  
- **Fachspezifische Dokumentationen:**  
  - Firmeneigene FAQ-Dokumentationen oder technische Artikel, sofern sie öffentlich zugänglich sind.
  
- **Benutzergenerierte Inhalte:**  
  - Zusätzlich können eigene erstellte Wissensdaten (z. B. aus internen Dokumenten) integriert werden, sofern diese den Lizenzbedingungen entsprechen.

*Hinweis:* Alle verwendeten Datenquellen werden sorgfältig ausgewählt, um sicherzustellen, dass ihre Nutzung den jeweiligen Lizenzbedingungen entspricht.

---

## Retrieval-Methoden

Um relevante Passagen für die Antwortgenerierung zu finden, kommen folgende Methoden zum Einsatz:

- **BM25:**  
  Eine klassische, keyword-basierte Suchmethode, die gut geeignet ist, relevante Dokumentpassagen basierend auf der Term-Frequenz zu identifizieren.

- **FAISS:**  
  Eine leistungsfähige Vektor-basierte Suchbibliothek von Facebook, die schnelle Ähnlichkeitsvergleiche ermöglicht. FAISS wird eingesetzt, um aus einem großen Pool von Textvektoren diejenigen zu finden, die dem Anfragevektor am ähnlichsten sind.

- **Elasticsearch:**  
  Eine skalierbare Volltext-Suchmaschine, die sich besonders für umfangreiche Datensätze eignet und komplexe Suchabfragen unterstützt.

Diese Methoden können einzeln oder in Kombination genutzt werden, um die bestmöglichen Textpassagen als Grundlage für die Generierung von Antworten zu identifizieren.

---

## Lizenzinformationen

- **Wikipedia:**  
  Inhalte werden unter der [CC BY-SA](https://creativecommons.org/licenses/by-sa/3.0/) Lizenz bereitgestellt. Diese Lizenz erfordert die Namensnennung der Urheber und das Teilen unter gleichen Bedingungen.
  
- **Open Data & Public Domain:**  
  Inhalte, die unter Open-Data-Lizenzen oder als Public Domain verfügbar sind, können frei genutzt werden, sollten aber bei der Integration stets auf die ursprünglichen Lizenzbedingungen geachtet werden.

- **Interne oder proprietäre Daten:**  
  Falls firmeneigene oder benutzergenerierte Inhalte verwendet werden, müssen diese den internen Datenschutz- und Lizenzrichtlinien entsprechen.

---

## Hinweise

- **Bildungszweck:**  
  Dieses Projekt ist primär für Bildungszwecke konzipiert. Die hier dokumentierten Daten und Methoden dienen dazu, den Aufbau eines RAG-Systems verständlich zu machen.

- **Erweiterungsmöglichkeiten:**  
  - Integration weiterer Wissensquellen.
  - Experimentieren mit kombinierten Retrieval-Methoden.
  - Erweiterung des Systems um zusätzliche Kontextinformationen (z. B. Nutzerhistorie).

- **Setup:**  
  Für die Einrichtung des Retrieval-Systems sind zusätzliche Tools wie FAISS oder Elasticsearch erforderlich. Die Einrichtung und Konfiguration dieser Tools wird in den entsprechenden Abschnitten der Projekt-Dokumentation erläutert.

