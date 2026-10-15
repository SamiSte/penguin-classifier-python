# Video-Demo starten

Die Demo verwendet die normale Anwendung mit einer eigenen Beobachtungsdatei
und eigenen Modellversionen. Jeder Start erzeugt unter `tmp/video-demo/` einen
neuen Ordner und kopiert ausschließlich die drei mitgelieferten Basismodellartefakte.
Vorhandene Beobachtungen, aktive Modellversionen und frühere Demo-Läufe bleiben
erhalten. Die Referenzdatei `data/penguins.csv` wird nur gelesen.

## Start unter Windows

1. Im Projektordner **`video-demo.bat` doppelklicken**.
2. Das Konsolenfenster geöffnet lassen; der Browser öffnet
   **<http://127.0.0.1:8058>**. Bei Bedarf diese Adresse von Hand öffnen.
3. Im Video erklären: **„Für den Funktionsablauf nutze ich isolierte Demo-Daten.
   Die Artbestätigung in diesem Beispiel ist simuliert.“**
4. Nach der Aufnahme im Konsolenfenster **Strg+C** drücken.
   Falls Windows anschließend nach Abbrechen der Batchdatei fragt, mit **J** bestätigen.
5. Für eine weitere Aufnahme die Startdatei erneut öffnen. Sie beginnt wieder
   beim Basismodell `initial`, ohne Zusatzbeobachtungen und ohne Modellkandidaten.

Die Demo benötigt eine vorbereitete Python-Umgebung mit den Projektpaketen.
Der Starter sucht zuerst `.venv`, dann `%USERPROFILE%\anaconda3\envs\penguin-ml`,
danach `python` im Suchpfad. Er installiert nichts und benötigt kein Docker.
Optional wählt die Umgebungsvariable `PENGUIN_VIDEO_PYTHON` ausdrücklich eine
bestimmte `python.exe`. Die lokal geprüfte Umgebung ist `penguin-ml` (Python 3.12).

Ist Port 8058 belegt, zuerst eine noch laufende Video-Demo in ihrem Konsolenfenster
beenden. Der Starter beendet keine fremden Prozesse. Eine reguläre App auf Port
8050 kann parallel laufen; im Video ausschließlich die Adresse mit **8058** verwenden.

## Ablauf mit zwei Demo-Beobachtungen

| Feld | Erster Eintrag | Zweiter Eintrag |
| --- | --- | --- |
| Schnabellänge | 46,1 mm | 46,2 mm |
| Schnabeltiefe | 17,3 mm | 17,3 mm |
| Flossenlänge | 197 mm | 197 mm |
| Körpergewicht | 4050 g | 4050 g |
| Geschlecht | Männlich | Männlich |
| Fundort / Insel | Dream | Dream |
| Fachlich bestätigte Art | Nicht bestätigt | Adelie – ausschließlich simuliertes Demo-Label |

Je nach Browser werden Dezimalzahlen mit Punkt oder Komma eingegeben.

1. Ersten Eintrag mit 46,1 mm klassifizieren und vorübergehend **Adelie** als
   simulierte Bestätigung auswählen, aber noch nicht speichern.
2. Schnabellänge über **+** von 46,1 auf 46,2 ändern. Das bisherige Ergebnis und
   der Stern verschwinden, die Bestätigung wird zurückgesetzt und Speichern
   gesperrt. Genau hier die zusammenhängenden Callback-Zustände erklären.
3. Mit **−** zurück auf 46,1 mm wechseln, erneut klassifizieren und mit
   **Nicht bestätigt** speichern. Diese erste gespeicherte Beobachtung wird
   nicht zum Trainingslabel; das Re-Training bleibt gesperrt.
4. Wieder auf 46,2 mm ändern, erneut **Pinguinart bestimmen**, dann **Adelie** als simulierte Bestätigung
   auswählen und speichern. **Nicht** aus der Vorhersage auf eine echte
   fachliche Bestätigung schließen. In echten Erhebungen müsste die Art
   unabhängig bekannt sein.
5. Nach der nächsten Statusaktualisierung, üblicherweise nach etwa zwei Sekunden, zeigt der Trainingsbereich eine neue
   bestätigte Beobachtung. **Re-Training starten** und den Abschluss abwarten.
   Währenddessen kann die bestehende Modellversion weiter vorhersagen.
6. Vergleich von Accuracy, Macro-F1 und Cohen’s κ zeigen. Unter **Prüfdetails und
   vorherige Version** sind Kreuzvalidierung, Konfusionsmatrizen und die aktive
   Version zu sehen. Der Kandidat wurde noch nicht aktiviert.
7. **Neue Version übernehmen**. Die aktive Versionskennung ändert sich;
   eine bestehende Vorhersage muss neu berechnet werden.
8. Unter **Prüfdetails und vorherige Version** auf **Vorherige Version
   wiederherstellen** klicken. Die aktive Version ist wieder `initial`.

Die Kennzahlen dieses Demo-Trainings illustrieren den technischen Ablauf.
Ein künstliches Sample belegt keine bessere Klassifikationsgüte. Nach Übernahme
oder Wiederherstellung verschwindet die Vergleichstabelle; deshalb **vorher** filmen.
Nach dem Zurückwechseln kann dieselbe Zusatzbeobachtung erneut als „neu“ erscheinen,
weil sie im wieder aktiven Basismodell noch nicht berücksichtigt ist.

## Technischer Probelauf ohne Browser

In PowerShell im Projektordner:

```powershell
.\video-demo.bat check --no-pause
```

Der Probelauf legt einen eigenen neuen Demo-Ordner an, prüft Dash-Endpunkte,
speichert die zwei synthetischen Beispiele, trainiert mit der normalen
Modellkonfiguration und prüft Übernahme sowie Wiederherstellung. Er startet
keinen Browser und keinen dauerhaften Server. `check-report.json` enthält das
Ergebnis; der genaue Pfad steht in der Konsole. Er ersetzt keinen realen
Browserdurchlauf, Docker-Test oder Offlinebetrieb mit Netzwerkabschaltung.

Direkter Python-Start, falls eine passende Umgebung bereits aktiv ist:

```powershell
python scripts/video_demo.py run --open-browser
python scripts/video_demo.py check
```

Für die Aufnahme den normalen `run`-Start verwenden: Der geprüfte `check`-Ordner
enthält bereits Demo-Beobachtungen und wird nicht als frische Aufnahme wiederverwendet.
