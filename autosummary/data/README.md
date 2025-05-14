# Datenübersicht für AutoSummary

In diesem Ordner befinden sich alle Datensätze, die im Rahmen des AutoSummary-Projekts verwendet werden. Das Ziel des Projekts ist es, automatische Zusammenfassungen von längeren Texten zu generieren – beispielsweise Nachrichtenartikel, wissenschaftliche Berichte oder Blogbeiträge. 

## 1. Herkunft der Datensätze

Die Datensätze setzen sich aus verschiedenen Quellen zusammen:

- **Nachrichtenartikel**: 
  - Beispiel: Artikel von öffentlich zugänglichen Nachrichtenseiten (z. B. CNN, BBC).  
  - Hinweis: Es werden nur Artikel verwendet, die unter Public Domain oder passenden Creative Commons-Lizenzen verfügbar sind.

- **Wissenschaftliche Texte**: 
  - Beispiel: Zusammenfassungen von Forschungsarbeiten oder wissenschaftlichen Artikeln aus Open-Access-Datenbanken wie arXiv oder PubMed Central.
  - Lizenz: Diese Artikel stehen in der Regel unter Lizenzen, die die Nutzung für Bildungszwecke erlauben.

- **Blogs und Reportagen**: 
  - Beispiel: Beiträge von Blogs, die ihre Inhalte unter freien Lizenzen veröffentlichen.
  - Lizenz: Es wird darauf geachtet, dass ausschließlich Inhalte genutzt werden, die frei verwendbar sind oder vom Autor explizit zur Nutzung freigegeben wurden.

## 2. Lizenzinformationen

- **Public Domain**: Viele der verwendeten Texte (insbesondere ältere Nachrichtenartikel und wissenschaftliche Veröffentlichungen) befinden sich im Public Domain und können uneingeschränkt genutzt werden.
- **Creative Commons**: Einige Inhalte werden unter Creative Commons Lizenzen (z. B. CC-BY oder CC-BY-SA) bereitgestellt. Dabei sind die Bedingungen der jeweiligen Lizenz zu beachten – insbesondere die Namensnennung des Urhebers.
- **Eigene Datensätze**: Bei selbst erstellten oder aggregierten Datensätzen wurde darauf geachtet, dass keine urheberrechtlich geschützten Inhalte ohne Erlaubnis verwendet werden.

Bitte beachte, dass du – falls du dieses Projekt erweiterst oder in einem kommerziellen Kontext nutzen möchtest – die Lizenzbedingungen der verwendeten Quellen nochmals sorgfältig prüfen solltest.

## 3. Datenaufbereitung

Die Rohdaten (im Ordner `raw/`) werden im nächsten Schritt verarbeitet:
- **Bereinigung**: Entfernen von HTML-Tags, Sonderzeichen und Duplikaten.
- **Tokenisierung**: Zerlegung der Texte in einzelne Wörter oder Subwörter.
- **Splitting**: Aufteilung in Trainings-, Validierungs- und Testdatensätze.

Die aufbereiteten Daten werden im Ordner `processed/` abgelegt und stehen dann für das Training und die Evaluierung des AutoSummary-Modells zur Verfügung.

## 4. Hinweise zur Nutzung

- Alle Datensätze in `data/raw/` und `data/processed/` sind für **Educational**-Zwecke vorgesehen.
- Bitte stelle sicher, dass du die Lizenzbedingungen der einzelnen Quellen einhältst, falls du die Daten öffentlich oder kommerziell nutzen möchtest.
- Die Dokumentation zu den Datenaufbereitungsschritten findest du in den entsprechenden Notebooks im Ordner `notebooks/`.

