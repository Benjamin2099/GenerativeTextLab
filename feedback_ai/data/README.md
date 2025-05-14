# Datenübersicht für Feedback AI

In diesem Ordner befinden sich alle Datensätze, die für das Feedback AI-Projekt verwendet werden. Das Ziel von Feedback AI ist es, ein generatives System zu entwickeln, das sich mithilfe von Nutzerfeedback (Daumen hoch / runter) kontinuierlich verbessert. Dabei werden zunächst generierte Texte samt Feedback in den Rohdaten (raw) gespeichert und anschließend in vorverarbeitete Datensätze (processed) überführt, bei denen positive und negative Beispiele entsprechend gewichtet werden.

## 1. Rohdaten (`data/raw/`)

- **Inhalt:**  
  Unbearbeitete generierte Texte zusammen mit Nutzerfeedback.  
  Jeder Datensatz enthält typischerweise:
  - **generated_text:** Der vom System generierte Text.
  - **feedback:** Die Bewertung des Nutzers (z. B. "thumbs_up" oder "thumbs_down").
  - Optional weitere Metadaten wie Zeitstempel oder Benutzerinformationen.

- **Beispiel eines Eintrags:**
  {
      "generated_text": "Dies ist ein Beispieltext, der von der KI generiert wurde.",
      "feedback": "thumbs_up"
  }
