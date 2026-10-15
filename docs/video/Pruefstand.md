# Prüfstand der Video-Vorbereitung

Stand der Unterlagen: 15.10.2026; Funktionsprüfungen vom 14.10.2026. Die Prüfungen beziehen sich auf die lokale Video-Vorbereitung. Es wurde kein Video aufgezeichnet und kein Container neu gebaut.

## Ausgeführte Prüfungen

| Prüfung | Ergebnis |
| --- | --- |
| Bestehende automatisierte Tests | 153 Tests im ersten Lauf bestanden; 11 POSIX-Starttests konnten wegen der Sandbox Git Bash nicht starten. Diese 11 wurden anschließend außerhalb der Sandbox wiederholt und bestanden. Damit sind alle 164 bestehenden Tests in diesen beiden Läufen abgedeckt. |
| Neue Tests für die Demo-Hilfe | 3 bestanden: getrennte neue Demo-Ordner, Basismodellkopien ohne aktive Modellverweise, belegter Port wird abgewiesen. |
| Vollständiger technischer Demo-Probelauf | Dash-Endpunkte, unbestätigter Ausschluss, synthetisches Zusatzlabel, Re-Training mit 500 Bäumen, ausdrückliche Übernahme und Wiederherstellung bestanden. Referenzdaten und Basismodellartefakte blieben nach SHA-256-Vergleich unverändert. |
| Tatsächlicher Browserdurchlauf | Vorhersage, Rücksetzung von Ergebnis/Stern/Bestätigung nach Eingabeänderung, Sperren des Speicherns, unbestätigte und bestätigte Speicherung, Trainingsfreigabe, Vergleich, Übernahme und Rückwechsel im Codex-Browser auf Port 8058 erfolgreich vorgeführt. |
| PCA und Dublettenprüfung | Datenhash, 256/86-Split und fünf CV-Aufteilungen stimmen mit den vorhandenen Modellstudien überein. PCA nur auf Trainingsdaten berechnet; Ergebnis zusätzlich in der Projektumgebung nachgerechnet. Keine exakten Dubletten oder Merkmalsüberschneidungen über die geprüften Grenzen. |
| Aufnahmeheft und Grafiken | PDF mit sechs A4-Seiten gerendert und jede Seite visuell geprüft. PCA-Grafik und Callback-Grafik auf Lesbarkeit geprüft. |

Die verwendete Projektumgebung ist `penguin-ml`: Python 3.12.13, scikit-learn 1.9.0, Dash 4.4.1 und pandas 3.0.5. Der PCA-Grafikexport nutzt die bereits vorhandene Anaconda-Basisumgebung mit Matplotlib; Versionen und Reproduktionsdaten stehen in [pca_ergebnisse.json](../modellpruefung/pca_ergebnisse.json). Die bestehenden Joblib-/NumPy-Abkündigungswarnungen beim Modellladen wurden weiterhin ausgegeben; die genannten Funktionsprüfungen bestanden.

## Reproduzieren

Aus dem Projektverzeichnis in einer passenden Python-Umgebung:

```powershell
python -m pytest -q
.\video-demo.bat check --no-pause
```

Der Probelauf legt unter `tmp/video-demo/` einen neuen Ordner mit `check-report.json` an. Für das Video danach `video-demo.bat` ohne `check` starten. Der Browserdurchlauf verwendete eine weitere, frisch angelegte Demo-Instanz; die Prüfungen verändern keine regulären Beobachtungen oder aktiven Modellversionen.

## Grenzen und verbleibende Abgabeschritte

- Die Demo läuft lokal mit Python. Sie bestätigt weder den aktuellen Docker-Image-Stand noch Linux-/macOS-Kompatibilität.
- Ein neuer Container-Funktionstest nach den bisherigen Programmänderungen und ein Test bei tatsächlich getrenntem Netzwerk stehen laut Projektprotokoll weiterhin aus.
- Die künstlich bestätigte Demo-Beobachtung dient nur dem Softwareablauf. Ihre angezeigten Trainingskennzahlen sind kein neuer Nachweis der Modellgüte.
- Video und Ton müssen vom Nutzer aufgenommen und abschließend kontrolliert werden. Abschlusszusammenfassung, aktuelle Softwaredokumentation und Softwarecontainer gehören zusätzlich in das finale Abgabepaket.

Die bestehenden Docker-Prüfungen sind separat im [Docker-Prüfprotokoll](../Docker_Pruefprotokoll.md) dokumentiert.
