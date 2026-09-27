# Prüfung der überarbeiteten Oberfläche

Geprüft am 27.09.2026 mit lokalem Python/Dash auf Port 8058. Beobachtungen und Modellversionen lagen ausschließlich unter `tmp/design-preview/`. Die gespeicherten Nutzerdaten und das aktive Modell der Docker-Anwendung wurden nicht verändert.

| Prüfung | Ergebnis |
| --- | --- |
| Plus- und Minus-Schaltflächen | Schnabellänge wurde von 44,5 auf 44,6 erhöht und wieder auf 44,5 reduziert. |
| Klassifikation | Art, Klassenwahrscheinlichkeiten und aktuelle Beobachtung im Diagramm angezeigt. |
| Fachliche Bestätigung und Speicherung | Eine ausdrücklich synthetische Beobachtung mit unabhängig gewähltem Testlabel gespeichert; erneutes Speichern anschließend gesperrt. |
| Re-Training | Kandidat erstellt und Modellvergleich angezeigt. Der gespeicherte Kandidat enthält 500 Bäume mit höchstens Tiefe 5. |
| Vergleich mit bisherigem Modell | Baseline verwendet die bisherigen Parameter ohne Tiefenbegrenzung, Kandidat die neue Begrenzung. |
| Ausdrückliche Übernahme | Modellversion aktiviert; alte Vorhersage und Bestätigung zurückgesetzt. |
| Wiederherstellung | Vorherige Modellversion wieder aktiviert; erneute Klassifikation möglich. |
| Desktop-Darstellung | Bei 1366 × 768 Pixeln passen die vier Hauptbereiche mit Vorhersage in das Fenster. |
| Schmale Darstellung | Bei 390 Pixeln Fensterbreite werden die Bereiche untereinander angezeigt; keine horizontale Überbreite im geprüften Zustand. |

Die Kennzahlen dieses synthetischen Funktionstests belegen ausschließlich den Ablauf. Für die methodische Bewertung gilt der [Modellvergleich auf den ursprünglichen Trainingsdaten](../modellpruefung/einfaches_modell.md).

Zusätzlich bestanden nach der Überarbeitung 164 automatisierte Tests. NumPy-/Joblib-Abhängigkeitswarnungen beim Laden vorhandener Modelle sind bekannt; sie führten zu keinem Testfehler. Der erneute Test des aktualisierten Docker-Images und ein tatsächlicher Test ohne Netzwerk stehen noch aus.

Ansicht: [Oberfläche am Desktop](../ansichten/oberflaeche-desktop.png).
