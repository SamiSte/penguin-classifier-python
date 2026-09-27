# Docker-Bereitstellung: Prüfstand und Abnahme

Stand: 27.09.2026. Dieses Protokoll trennt ausgeführte Prüfungen von noch offenen
Container-Tests. Der erste Start wurde vom Nutzer bestätigt; anschließend wurde
der echte Container-Funktionstest erfolgreich ausgeführt und sein Protokoll geprüft.

## Bereits geprüft

- 22 automatisierte Starttests mit einer simulierten Docker-CLI: Windows-Batch
  unter `cmd.exe` und POSIX-Shellskripte unter Git Bash.
- Start aus einem anderen Arbeitsverzeichnis und Projektpfade mit Leerzeichen.
- Erstellen vor dem ersten Start; späterer Start mit `--no-build --pull never`.
- Bereitschaftsprüfung mit `--wait`; kein Erfolgshinweis bei Startfehlern.
- Fehlerbehandlung für unerreichbare Docker-Engine, fehlendes Compose/Image,
  fehlgeschlagenen Aufbau und ungültige Portangaben.
- Stoppen ohne Befehle zum Löschen von Containern oder Volumes.

Die echte Docker-CLI konnte aus der verwendeten Arbeitsumgebung nicht gestartet
werden (`Zugriff verweigert`). Docker Desktop war installiert und lief. Daher
wurden dort weder ein Image gebaut noch Container- oder Volume-Tests ausgeführt.
Linux und macOS wurden nicht auf eigenen Betriebssystemen getestet.

Der Nutzer hat `setup.bat` anschließend selbst unter Windows ausgeführt und den
erfolgreichen Start bestätigt. Ein technisches Ablaufprotokoll dieses ersten
Starts liegt noch nicht vor.

## Ergebnisse des Containerlaufs

Prüflauf: `penguin-pruefung-2a71f1fa6e594b82b3b1c00e61899847`, Windows mit Docker
Engine 29.8.0 und Compose 5.5.1, Linux-Container auf amd64, Python 3.12.13.
Image: `sha256:bf685f987e53d7f4328dbcb2c759f13c57a0c9325be28b64404e976c8d5776a8`.
Das [gesicherte Ablaufprotokoll](pruefungen/docker-2026-09-27.log) enthält alle
Prüfschritte; lediglich der persönliche Projektpfad wurde gekürzt.
SHA-256 des unveränderten Originalprotokolls:
`39211c3de0862540d437e915a1adf87dd4d017990fb15bc4372ca41113775295`.

Die Meldungen „no such volume“ vor Schritt 1 sind erwartete Prüfungen, dass die
neu benannten Testvolumes noch nicht existieren. Der Ablauf endet mit BESTANDEN.

| Prüfung | Erwartetes Ergebnis | Status |
| --- | --- | --- |
| Einmalige Einrichtung | Aufbau und Start über `setup.bat` erfolgreich | Vom Nutzer bestätigt; kein eigenes Ausführungsprotokoll |
| Klassifikation und Speichern | Vorhersage durch Modellservice, genau eine CSV-Zeile ergänzt | Bestanden im Container |
| Re-Training und Übernahme | Kandidat bewertet, neue Version erst nach ausdrücklichem Aufruf aktiv | Bestanden im Container |
| Container entfernen und neu starten | CSV und sämtliche Modellartefakte unverändert erhalten | Bestanden, SHA-256-Vergleich |
| Vorherige Version wiederherstellen | Gesicherte Version verwendbar, Vorhersage unverändert | Bestanden im Container |
| Start und Nutzung ohne Netzwerk | Oberfläche, Vorhersage, Speicherung und Re-Training funktionieren | Offen |

Der Lauf belegt den oben genannten Image-Stand. Nach der anschließenden
Überarbeitung von Oberfläche und Modellkonfiguration muss `setup.bat` das neue
Image bauen und `docker-pruefen.bat` erneut ausgeführt werden. Die lokalen
Regressionstests bestehen mit 164 Tests; sie ersetzen diese erneute Containerprüfung
nicht. Linux- und macOS-Hostsysteme sowie ein tatsächlich getrenntes Netzwerk
wurden weiterhin nicht geprüft.

## Automatischer Funktionstest unter Windows

Nach der einmaligen Einrichtung genügt ein Doppelklick auf
**`docker-pruefen.bat`** im Projektordner. Docker muss bereit sein; ein erneuter
Image-Aufbau ist nicht nötig. Port 8057 muss frei sein. Der Prüfstart legt ein
eigenes Compose-Projekt und eigene Testvolumes an. Die normale Anwendung auf
Port 8050 kann weiterlaufen.

Der Test prüft die Erreichbarkeit von Dash sowie die Anwendungsdienste für
Vorhersage, CSV-Speicherung, Re-Training, getrennte Modellübernahme und Rückkehr
zur vorherigen Version. Er entfernt den Testcontainer, erstellt ihn mit
denselben Volumes neu und vergleicht alle Daten- und Modellartefakte per SHA-256.
Die verwendeten Beobachtungen und Artbestätigungen sind synthetische Testfälle.
Die normale Modellkonfiguration bleibt dabei unverändert.

Am Ende erscheint **BESTANDEN** oder **FEHLGESCHLAGEN** mit dem Pfad zum
vollständigen Protokoll unter `tmp/docker-pruefung-<Laufkennung>.log`. Der
Testcontainer wird beendet; die Testvolumes bleiben für eine Nachprüfung
erhalten. Der Test öffnet keinen zusätzlichen Browser. Ein erfolgreicher Lauf
prüft die Dienste im Container, jedoch weder alle Browserinteraktionen noch
den Betrieb bei tatsächlich getrennter Netzwerkverbindung.

Der Ablauf wurde vorab mit 15 simulierten Docker-CLI-Tests und fünf lokalen
Prüfungen des Testskripts kontrolliert. Diese simulierten beziehungsweise lokalen
Tests sind getrennt vom oben protokollierten echten Containerlauf zu betrachten.

## Manuelle Ergänzung: Oberfläche und Offlinebetrieb

Die folgenden Schritte sind eine ausführliche Alternative zur automatischen
Prüfung sowie eine Ergänzung für Oberflächenbedienung und Offlinebetrieb.

Die folgenden Befehle werden **im Projektordner in einer neuen PowerShell**
ausgeführt. Sie verwenden einen eigenen Compose-Projektnamen, Port 8057 und
eigene Volumes. Synthetische Testbeobachtungen gelangen dadurch nicht in die
regulären Datenbestände. Die gleiche PowerShell bis zum Ende geöffnet lassen,
damit die Testeinstellungen erhalten bleiben.

### 1. Einrichten und starten

Docker Desktop mit laufender Linux-Engine vorbereiten; Internetzugang für den
Image-Aufbau herstellen. Dann:

```powershell
$qaSuffix = [guid]::NewGuid().ToString('N').Substring(0, 8)
$env:COMPOSE_PROJECT_NAME = "penguin-pruefung-$qaSuffix"
$env:PENGUIN_DATA_VOLUME = "$($env:COMPOSE_PROJECT_NAME)-data"
$env:PENGUIN_MODELS_VOLUME = "$($env:COMPOSE_PROJECT_NAME)-models"
$env:PENGUIN_PORT = '8057'

$qaDockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if ($qaDockerCommand) {
    $qaDocker = $qaDockerCommand.Source
} else {
    $qaDocker = Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
    if (-not (Test-Path -LiteralPath $qaDocker)) {
        $qaDocker = Join-Path $env:ProgramFiles 'Docker/Docker/resources/bin/docker.exe'
    }
}
if (-not (Test-Path -LiteralPath $qaDocker)) { throw 'Docker-CLI nicht gefunden.' }

& .\setup.bat --no-pause
if ($LASTEXITCODE -ne 0) { throw 'Einrichtung fehlgeschlagen; zuerst Fehlermeldung prüfen.' }
& $qaDocker compose ps
```

Erwartet: Containerstatus `healthy` und Browser unter `http://127.0.0.1:8057`.
Ein bereits belegter Port muss vor dem Test freigegeben oder anders gewählt werden.

### 2. Anwendung und Re-Training prüfen

1. Für den Funktionstest beispielsweise 46,1 mm Schnabellänge, 17,3 mm
   Schnabeltiefe, 197 mm Flossenlänge, 4050 g, Geschlecht „Unbekannt“ und Insel
   Biscoe eingeben. Plus-/Minus-Schaltflächen und Achsenwahl ausprobieren.
2. Klassifizieren und die Ergebnisanzeige prüfen. Für **diesen synthetischen
   Testfall ausschließlich in der Testinstanz** „Adelie“ als bestätigte Art
   auswählen und einmal speichern. Dies ist keine reale fachliche Bestätigung.
3. Prüfen, dass erneutes Speichern derselben Vorhersage gesperrt ist und eine
   neue bestätigte Beobachtung für das Re-Training angezeigt wird.
4. Re-Training starten, bis zum Abschluss warten und den Modellvergleich
   öffnen. Die neue Version ausdrücklich übernehmen und ihre Kennung notieren.

### 3. Dateistand vor dem Neustart festhalten

Das folgende Kommando bildet Prüfsummen aller Dateien in den beiden
Containerverzeichnissen. Während dieser Prüfung keine Daten speichern und
keinen weiteren Trainingsauftrag starten.

```powershell
$qaHashCode = "import hashlib,json; from pathlib import Path; print(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for root in ('/app/data','/app/models') for p in sorted(Path(root).rglob('*')) if p.is_file()}, sort_keys=True))"
$qaBefore = & $qaDocker compose exec -T app python -c $qaHashCode
if ($LASTEXITCODE -ne 0) { throw 'Dateiprüfung fehlgeschlagen.' }
$qaBefore
```

Die Ausgabe muss unter anderem `new_observations.csv`, `active_model.json` und
Dateien der neu erzeugten Modellversion enthalten.

### 4. Container entfernen und wieder starten

Nur in derselben Test-PowerShell ausführen. `down` entfernt hier den Testcontainer,
behält aber die Volumes. **Kein `-v` und keine Volume-Löschung verwenden.**

```powershell
& .\stop.bat --no-pause
if ($LASTEXITCODE -ne 0) { throw 'Stoppen fehlgeschlagen.' }
& $qaDocker compose down
if ($LASTEXITCODE -ne 0) { throw 'Entfernen des Testcontainers fehlgeschlagen.' }
& .\start.bat --no-pause
if ($LASTEXITCODE -ne 0) { throw 'Neustart fehlgeschlagen.' }

$qaAfter = & $qaDocker compose exec -T app python -c $qaHashCode
if ($LASTEXITCODE -ne 0) { throw 'Dateiprüfung nach Neustart fehlgeschlagen.' }
if (($qaBefore -join "`n") -ceq ($qaAfter -join "`n")) {
    'Persistenzprüfung bestanden: alle Dateien unverändert.'
} else {
    throw 'Dateistände unterscheiden sich; Persistenzprüfung nicht bestanden.'
}
```

Im Browser dieselbe aktive Modellkennung kontrollieren, eine neue Vorhersage
erstellen und anschließend die vorherige Modellversion wiederherstellen.
Auch damit muss eine Vorhersage möglich sein.

### 5. Offlinebetrieb prüfen und beenden

Nach erfolgreicher Einrichtung den Testcontainer stoppen. Die Netzwerkverbindung
des Rechners vorübergehend trennen, Docker weiterlaufen lassen und erneut
`start.bat` aus **derselben PowerShell** aufrufen. Die Seite vollständig neu laden.
Vorhersage, Speicherung und ein weiteres Re-Training mit einer neuen
synthetischen Testbeobachtung prüfen. Anschließend mit `stop.bat` beenden und
die Netzwerkverbindung wiederherstellen.

Datum, Betriebssystem, Docker-/Compose-Version und tatsächliche Ergebnisse in
der Tabelle ergänzen. Fehlermeldungen dokumentieren; offene Prüfungen nicht als
bestanden kennzeichnen. Testvolumes bleiben bis zur bewussten Bereinigung
erhalten. Nach Schließen dieser PowerShell gelten bei einem normalen Doppelklick
wieder die regulären Volumes und Port 8050.
