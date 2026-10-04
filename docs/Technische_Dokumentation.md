# Technische Dokumentation

Diese Dokumentation beschreibt Bedienungsdetails, Datenaufbereitung, Modellierung,
Speicherung, Bereitstellung und Prüfungen des Penguin Classifier. Eine kurze
Einführung mit Einrichtung und Start steht in der [README](../README.md).

**Alle Befehle und Codepfade in diesem Dokument beziehen sich auf das
Projektverzeichnis**, nicht auf den Ordner `docs/`. Relative Markdown-Links
führen dagegen vom Speicherort dieses Dokuments zum jeweiligen Ziel.

- [Bedienung](#bedienung)
- [Datengrundlage](#datengrundlage) und [Merkmalsauswahl](#merkmalsauswahl)
- [Modell](#machine-learning-modell) und [Bewertung](#modellbewertung)
- [Projektstruktur](#projektstruktur)
- [Docker](#start-mit-docker) und [Speicherung](#persistente-speicherung)
- [Python-Start](#lokaler-start-mit-python) und [Tests](#automatisierte-tests)
- [Modellstudie](#separate-prüfung-auf-überanpassung)
- [Basistraining](#modelltraining)
- [Re-Training und Modellversionen](#manuelles-re-training-und-modellvergleich)

## Bedienung

1. Die vorbelegten Beispielwerte durch die eigenen Messungen ersetzen. Plus und
   Minus ändern Schnabelmaße um 0,1 mm, Flossenlänge um 1 mm und Körpergewicht
   um 50 g. Die Werte können auch direkt eingegeben werden.
2. **Pinguinart bestimmen** anklicken, Art und Modellwahrscheinlichkeiten prüfen.
   Die Vorhersage wird rechts als schwarzer Stern eingezeichnet; beide Achsen
   können über die Auswahlfelder geändert werden.
3. Ist die tatsächliche Art unabhängig von der Modellvorhersage fachlich bekannt,
   unter **Fachlich bestätigte Art (optional)** die entsprechende Art auswählen.
   Andernfalls **Nicht bestätigt** belassen. Eine abweichende Bestätigung ändert
   die Modellvorhersage nicht.
4. **Beobachtung speichern** schreibt Messung, Vorhersage und eine gegebenenfalls
   bestätigte Art in die CSV-Datei.
   Eine Statusmeldung bestätigt das Speichern. Dieselbe Vorhersage kann nicht
   durch erneutes Klicken ein zweites Mal gespeichert werden.

Wird ein Messwert, das Geschlecht oder die Insel geändert, wird das bisherige
Ergebnis verworfen. Vor dem Speichern ist dann erneut zu klassifizieren.
Die Artbestätigung wird bei geänderten Eingaben und bei jeder neuen Klassifikation
auf **Nicht bestätigt** zurückgesetzt. Nach dem Speichern ist die Auswahl für
diese Beobachtung gesperrt. Die CSV-Spalte `validated_species` enthält nur die
ausdrücklich ausgewählte Art und bleibt andernfalls leer. Es wird nie automatisch
die Modellvorhersage als bestätigte Art übernommen. Dieses Feld bildet die
Grundlage für das manuelle Re-Training im Bereich **Modell aktualisieren**.
Die normale Desktopansicht ist bei 1366 × 768 Pixeln ohne Scrollen bedienbar;
auf schmalen Geräten werden Formular und Diagramm untereinander angeordnet.
Längere Warnmeldungen dürfen die Seite erweitern, damit alle Hinweise lesbar bleiben.

Die Dateien in `assets/` enthalten Gestaltung und Zahlenfeld-Steuerung und müssen
beim lokalen Start und im Docker-Image mitgeliefert werden.

## Datengrundlage

Als Datengrundlage wird der öffentlich verfügbare **Palmer-Penguins-Datensatz** verwendet.

Der ursprüngliche Datensatz enthält **344 Beobachtungen**.

Für die Modellierung werden Beobachtungen nur dann ausgeschlossen, wenn mindestens eines der vier benötigten morphologischen Merkmale fehlt. Dadurch werden zwei Beobachtungen entfernt und **342 Beobachtungen** für die Modellierung verwendet.

Fehlende Angaben zum Geschlecht führen dagegen nicht zum Ausschluss einer Beobachtung. Sie werden innerhalb der Modellpipeline als `unknown` behandelt.

Die Zielvariable ist:

```text
species
```

mit den drei Klassen:

```text
Adelie
Chinstrap
Gentoo
```

## Merkmalsauswahl

Für das finale Klassifikationsmodell werden folgende Merkmale verwendet:

```text
bill_length_mm
bill_depth_mm
flipper_length_mm
body_mass_g
sex
```

Die Variable `year` wird nicht für die Klassifikation verwendet.

Die Variable `island` wird in der Benutzeroberfläche weiterhin als Kontextinformation erfasst und bei neuen Beobachtungen gespeichert, ist jedoch **kein Eingangsmerkmal des finalen Klassifikators**.

Diese Entscheidung wurde zusätzlich durch einen Ablationstest überprüft. Obwohl `island` bei einem zunächst damit trainierten Modell eine hohe Permutationswichtigkeit aufwies, führte das erneute Training ohne dieses Merkmal praktisch zu keiner Verschlechterung der Modellleistung. Dadurch kann die Klassifikation stärker auf den biologischen Merkmalen des individuellen Pinguins basieren und ist weniger vom Fundort abhängig.

## Machine-Learning-Modell

Für die Mehrklassenklassifikation wird ein **Random Forest Classifier** aus
scikit-learn mit 500 Entscheidungsbäumen eingesetzt. Neue Trainingsläufe begrenzen
die Baumtiefe auf 5 (`max_depth=5`, `min_samples_leaf=1`). In der Modellprüfung
sank dadurch die mittlere Blattzahl je Baum von 14,75 auf 11,19, etwa 24 %.

Gespeicherte Modellversionen ändern sich durch ein Code-Update nicht. Das
mitgelieferte ursprüngliche Basismodell verwendet `max_depth=None`; bestehende
aktive Versionen behalten ihre bisherigen Einstellungen. Die Begrenzung wird beim nächsten Re-Training
angewendet; erst **Neue Version übernehmen** aktiviert den Kandidaten. Wie bisher
benötigt dieses Re-Training neue oder geänderte fachlich bestätigte Beobachtungen.

Die Vorverarbeitung und der Klassifikator sind in einer gemeinsamen scikit-learn-Pipeline gekapselt.

Numerische Merkmale werden direkt an den Klassifikator übergeben. Das kategoriale Merkmal `sex` wird innerhalb der Pipeline verarbeitet. Fehlende Geschlechtsangaben werden als `unknown` behandelt und anschließend mittels One-Hot-Encoding kodiert.

Für reproduzierbare Ergebnisse wird ein fester Random State von `42` verwendet.

## Modellbewertung

Die ursprüngliche Modellbewertung erfolgt mit einem stratifizierten **75/25-Train-Test-Split**. Zusätzlich wird auf den Trainingsdaten eine **5-fache stratifizierte Kreuzvalidierung** durchgeführt. Die folgenden Kennzahlen dokumentieren das ursprüngliche Basismodell; aktuelle Vergleiche nach einem Re-Training werden separat mit der jeweiligen Modellversion gespeichert.

Ergebnisse der Kreuzvalidierung:

| Kennzahl      |        Ergebnis |
| ------------- | --------------: |
| Accuracy      | 0,9882 ± 0,0157 |
| Macro-F1      | 0,9854 ± 0,0194 |
| Cohen's Kappa | 0,9814 ± 0,0247 |

Auf dem unabhängigen Testdatensatz mit 86 Beobachtungen wurden in dem verwendeten Split alle Beobachtungen korrekt klassifiziert:

| Kennzahl      | Ergebnis |
| ------------- | -------: |
| Test-Accuracy |   1,0000 |
| Macro-F1      |   1,0000 |
| Cohen's Kappa |   1,0000 |

Die Test-Accuracy von 100 % bezieht sich ausschließlich auf diesen konkreten Testsplit und wird nicht als allgemeine Modellgenauigkeit interpretiert. Für die Einschätzung der Generalisierungsleistung wird insbesondere die Kreuzvalidierung berücksichtigt.

Die Permutationswichtigkeit auf dem Testdatensatz zeigt die höchste Bedeutung für die Schnabellänge.

Die beim Training erzeugten Ergebnisse werden unter `models/metrics.json` gespeichert.

## Projektstruktur

```text
penguin-classifier-python/
│
├── app.py
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── requirements-dev.txt
├── setup.bat / start.bat / stop.bat          # Windows
├── setup.sh / start.sh / stop.sh             # Linux
├── setup.command / start.command / stop.command  # macOS
├── docker-pruefen.bat                    # Getrennter Container-Funktionstest
├── pytest.ini
├── README.md
├── .gitignore
├── .dockerignore
│
├── data/
│   └── penguins.csv
│
├── models/
│   ├── metrics.json
│   ├── model_metadata.json
│   ├── penguin_pipeline.joblib
│   ├── active_model.json       # Nach der ersten Modellübernahme
│   ├── pending_candidate.json  # Noch nicht übernommene Modellvariante
│   └── versions/              # Gesicherte und neu trainierte Modellversionen
│
├── scripts/
│   ├── explore_data.py
│   ├── train_model.py
│   ├── diagnose_island_feature.py
│   ├── evaluate_regularization.py
│   ├── model_study_report.py
│   ├── compare_model_baseline.py
│   ├── docker-launcher.bat
│   ├── docker-launcher.sh
│   └── container_check.py
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── figures.py
│   ├── formatting.py
│   ├── modeling.py
│   ├── model_service.py
│   ├── model_registry.py
│   ├── retraining.py
│   ├── retraining_data.py
│   ├── ui.py
│   └── storage.py
│
├── tests/
│   ├── test_data_processing.py
│   ├── test_model_service.py
│   ├── test_storage.py
│   ├── test_app.py
│   ├── test_model_registry.py
│   ├── test_retraining.py
│   ├── test_retraining_data.py
│   ├── test_model_study.py
│   ├── test_simple_model_comparison.py
│   ├── test_docker_launchers.py
│   ├── test_container_check.py
│   └── test_container_check_launcher.py
│
└── docs/
```

## Start mit Docker

Docker bündelt Python, Anwendung, Modell, Referenzdaten und Paketabhängigkeiten.
Eine lokale Python-Installation ist für diesen Startweg nicht erforderlich.
Die Anwendung ist nur auf dem eigenen Rechner unter `http://127.0.0.1:8050`
erreichbar. Zusätzliches Hosting ist nicht vorgesehen.

### Einmalige Einrichtung vor dem Einsatz

1. Den vollständigen Projektordner herunterladen oder klonen.
2. Docker Desktop für Windows/macOS beziehungsweise Docker Engine mit Compose v2
   für Linux installieren und starten. Unter Windows/macOS muss die Laufzeit
   Linux-Container verwenden. Compose muss `up --wait` unterstützen.
3. Mit Internetverbindung die passende Datei ausführen:

| System | Einrichtung | Späterer Start | Beenden |
| --- | --- | --- | --- |
| Windows | Doppelklick auf `setup.bat` | Doppelklick auf `start.bat` | Doppelklick auf `stop.bat` |
| macOS | Doppelklick auf `setup.command` | Doppelklick auf `start.command` | Doppelklick auf `stop.command` |
| Linux | `sh setup.sh` | `sh start.sh` | `sh stop.sh` |

Unter macOS werden die Startdateien bei der Einrichtung einmalig im Terminal
ausführbar gemacht: `chmod +x setup.command start.command stop.command`.
Unter Linux kann bei der Einrichtung eine Desktop-Verknüpfung angelegt werden,
die `sh` mit dem vollständigen Pfad zur jeweiligen Startdatei aufruft.
Kommandozeileneingaben sind dann für die reguläre Bedienung nicht erforderlich.

`setup` erstellt das Image `penguin-classifier:local`, startet den Container und
öffnet nach erfolgreicher Bereitschaftsprüfung den Browser. Der erste Aufbau
kann mehrere Minuten dauern. Bei Fehlern bleibt eine erklärende Meldung sichtbar.

### Regulärer Betrieb

Docker muss betriebsbereit sein. Danach genügt `start`: Es wird ausschließlich
das bereits vorhandene Image verwendet, ohne Neubau oder Image-Download.
Vorhersagen, Speicherung und Re-Training sind für den lokalen Offlinebetrieb
ausgelegt. Wenn sich kein Browser öffnet, die oben genannte Adresse aufrufen.

Das Schließen des Browserfensters beendet den Container nicht; dazu `stop`
verwenden. Beobachtungen und Modellversionen bleiben erhalten. Ein erneuter
Start öffnet die Anwendung mit den vorhandenen Daten. Nach Änderungen am
Programm wird `setup` erneut ausgeführt.

Ist Port 8050 belegt, zunächst eine bereits laufende Python- oder ältere
Docker-Instanz der Anwendung beenden. Alternativ kann bei der Einrichtung ein
anderer Port über `PENGUIN_PORT` festgelegt werden. Alle Starter akzeptieren
`--no-browser` und `--no-pause` für einen Aufruf ohne Browserstart und Warteprompt.

**Prüfstand:** Die Startabläufe sind mit simulierter Docker-CLI unter Windows
und Git Bash getestet. Der Nutzer hat Aufbau und Start über `setup.bat` unter
Windows bestätigt. Der echte Container-Funktionstest vom 27.09.2026 hat
CSV-Speicherung, Re-Training, Modellwechsel und Persistenz nach dem Neuerstellen
des Containers bestanden. Sein Protokoll gehört zum damals gebauten Image;
nach den anschließenden Änderungen an Oberfläche und Modellkonfiguration ist
ein erneuter Aufbau und Container-Test erforderlich. Ein Test bei tatsächlich
getrennter Netzwerkverbindung steht ebenfalls noch aus.

Mit **`docker-pruefen.bat`** lässt sich auf Port 8057 ein getrennter Funktionstest
starten. Er prüft Speichern, Re-Training, Modellwechsel und Dateierhalt nach dem
Entfernen und Neuerstellen des Containers. Die normalen Daten bleiben unberührt;
das Ergebnis wird unter `tmp/` protokolliert. Ablauf, Grenzen und Status stehen
im [Docker-Prüfprotokoll](Docker_Pruefprotokoll.md).

## Persistente Speicherung

Neue Beobachtungen werden in:

```text
data/new_observations.csv
```

gespeichert.

Compose legt die benannten Volumes `penguin-data` für `/app/data` und
`penguin-models` für `/app/models` automatisch an. Bei der ersten Verwendung
leerer Volumes übernimmt Docker die Referenzdaten und das Basismodell aus dem
Image. Die Volumes speichern Beobachtungen **und Modellversionen** unabhängig
vom Container; sie sind nicht mit den gleichnamigen Ordnern auf dem Host identisch.
Für Sicherungskopien müssen beide Verzeichnisse berücksichtigt werden.
Die Volumes dürfen nicht gelöscht werden, solange ihre Inhalte noch benötigt
werden. Ein neu gebautes Image ersetzt keine bereits vorhandenen Volumedaten.
Es darf nur eine Anwendungsinstanz auf dieselben Daten- und Modellvolumes
zugreifen; paralleler Mehrbenutzerbetrieb ist nicht vorgesehen.

Wer direkt auf die Dateien im Projektordner zugreifen möchte, kann stattdessen
beide vorhandenen lokalen Ordner als Bind-Mount einbinden. Diese Variante wird
**anstelle** der Startskripte aus dem Projektverzeichnis verwendet; eine zuvor
mit Compose gestartete Instanz muss beendet sein.
Die Ordner müssen die mitgelieferten Referenzdaten und Modelldateien enthalten.

### Windows PowerShell

```powershell
docker run --rm -p 127.0.0.1:8050:8050 --name penguin-classifier-app --mount "type=bind,source=$($PWD.Path)\data,target=/app/data" --mount "type=bind,source=$($PWD.Path)\models,target=/app/models" penguin-classifier:local
```

### Linux / macOS

```bash
docker run --rm -p 127.0.0.1:8050:8050 --name penguin-classifier-app --mount "type=bind,source=$(pwd)/data,target=/app/data" --mount "type=bind,source=$(pwd)/models,target=/app/models" penguin-classifier:local
```

Bei Bind-Mounts bleiben Beobachtungen und Modellversionen direkt im lokalen
Projektordner verfügbar. Es darf nur eine Anwendungsinstanz auf dieselben
Daten- und Modellordner zugreifen; paralleler Mehrbenutzerbetrieb ist nicht
vorgesehen.

Eine neu gespeicherte Beobachtung enthält neben den Messwerten unter anderem:

```text
timestamp_utc
island
predicted_species
probability_adelie
probability_chinstrap
probability_gentoo
validated_species
```

`predicted_species` bezeichnet die vom Modell vorhergesagte Art.

`validated_species` enthält die vor dem Speichern ausdrücklich ausgewählte,
fachlich bestätigte Art. Ohne Bestätigung bleibt die Spalte leer. Modellvorhersage
und verifiziertes Label werden dadurch getrennt gespeichert.

## Lokaler Start mit Python

Alternativ kann die Anwendung direkt mit Python gestartet werden.

Die getestete Entwicklungsumgebung verwendet Python 3.12 sowie die in `requirements.txt` festgelegten Paketversionen.

Installation der Abhängigkeiten:

```bash
pip install -r requirements.txt
```

Start der Anwendung:

```bash
python app.py
```

Die Anwendung ist anschließend unter:

```text
http://127.0.0.1:8050
```

erreichbar.

## Automatisierte Tests

Für zentrale Komponenten wurden automatisierte Tests mit `pytest` implementiert.

Getestet werden insbesondere:

* Datenbereinigung
* Erhalt von Beobachtungen mit unbekanntem Geschlecht
* Entfernung von Beobachtungen mit fehlenden morphologischen Messwerten
* Modellvorhersagen und Klassenwahrscheinlichkeiten
* Validierung fehlerhafter Eingaben
* Warnungen bei Werten außerhalb des Trainingsbereichs
* Erstellung einer CSV-Datei
* Anhängen mehrerer Beobachtungen an eine bestehende CSV-Datei
* getrennte Speicherung von vorhergesagter und fachlich bestätigter Art
* Ablehnung ungültiger Bestätigungen ohne Änderung vorhandener Daten
* Zurücksetzen der Bestätigung und Sperren veralteter oder bereits gespeicherter Ergebnisse
* zyklusfreie Abhängigkeiten zwischen den Dash-Callbacks
* Ausschluss unbestätigter und doppelter Daten sowie Erkennung widersprüchlicher Labels
* gleicher, vom Training ausgeschlossener Prüfanteil für beide Vergleichsmodelle
* Re-Training, Neustart mit ausstehendem Kandidaten, ausdrückliche Übernahme und Wiederherstellung
* unveränderte Originalmodelle bei Trainingsfehlern und abgewiesenen Modellwechseln
* identische CV-Aufteilungen und Ausschluss der Testdaten bei der separaten Modellstudie
* verschachtelte Lernkurven-Teilmengen und die vorab festgelegte Auswahlregel
* Startskripte mit simulierter Docker-CLI: Voraussetzungen, Offline-Startparameter,
  Bereitschaftsprüfung, Fehlerbehandlung und Stoppen ohne Löschen der Volumes
* getrennter Container-Prüfstart mit simulierter CLI sowie lokale Prüfung seiner
  Testlogik, einschließlich Ablehnung vorhandener Daten und Erkennung von Datenverlust

Testabhängigkeiten installieren und Tests ausführen:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -v
```

Windows-Batchtests benötigen Windows; POSIX-Tests benötigen `sh` oder unter
Windows Git Bash. Fehlt die jeweilige Shell, werden diese Tests übersprungen.
Die simulierten Starttests ersetzen keinen echten Container-Test.

Zuletzt dokumentierter vollständiger Testlauf (Windows mit Git Bash):

```text
164 passed
```

Nach der Anpassung der Zahlenformatierung am 03.10.2026 bestanden die 52 Tests
aus `tests/test_app.py`, `tests/test_model_service.py` und `tests/test_storage.py`.
Dieser gezielte Prüflauf ersetzt keinen erneuten vollständigen Testlauf.

## Separate Prüfung auf Überanpassung

Die Modellstudie untersucht neun Kombinationen aus `max_depth = None, 5, 10`
und `min_samples_leaf = 1, 2, 4` sowie eine Lernkurve des bisherigen Modells:

```bash
python scripts/evaluate_regularization.py
```

Das Skript verwendet die 256 Trainingsbeobachtungen des ursprünglichen
stratifizierten 75/25-Splits und dieselben fünf CV-Aufteilungen für jede Variante.
Die 86 bereits bekannten Testbeobachtungen werden weder für die Auswahl noch
für die Lernkurve oder eine erneute Bewertung verwendet. Die Vorverarbeitung
wird je Trainingsfold neu gelernt. Die Lernkurve verwendet wachsende,
verschachtelte Teilmengen mit annähernd gleicher Klassenverteilung.

Ergebnisse unter [docs/modellpruefung](modellpruefung):

* [Interaktiver Bericht](modellpruefung/bericht.html), auch offline im Browser lesbar
* [Ergebnistext und Tabellen](modellpruefung/bericht.md)
* [Variantenvergleich als CSV](modellpruefung/Vergleich.csv)
* [Lernkurve als CSV](modellpruefung/lernkurve.csv)
* [Vollständiges Versuchsprotokoll](modellpruefung/ergebnisse.json) mit Einzelwerten,
  Datenaufteilungen, Teilmengen, Versionen und Datei-Prüfsummen

Vor dem Vergleich wurde eine Toleranz von 0,005 im mittleren CV-Macro-F1
festgelegt. Innerhalb dieses Abstands zum besten Ergebnis wird die Variante mit
den kleinsten Bäumen (mittlere Knotenzahl) empfohlen. Diese praktische Regel
belegt keine statistische Gleichwertigkeit. Die Fold-Streuung ist kein
Konfidenzintervall; die zur Auswahl verwendete CV ersetzt keinen unabhängigen Test.

Im dokumentierten Lauf erzielte das bisherige Modell den höchsten CV-Macro-F1
von 0,9854. Eine Begrenzung auf Tiefe 5 bei mindestens einem Fall je Blatt erreichte
0,9807 mit rund 25 % weniger Knoten je Baum und wird nach der Auswahlregel als
sparsamere Variante empfohlen. Diese Konfiguration wird inzwischen für neue
Trainingsläufe verwendet. Eine bessere Generalisierung auf neue Expeditionen
ist damit nicht nachgewiesen. Das Studienskript selbst verändert weder die
Modellkonfiguration noch gespeicherte App-Modelle oder Beobachtungen.

### Vergleich mit einer logistischen Regression

`python scripts/compare_model_baseline.py` ergänzt eine logistische Regression
mit L2-Regularisierung und pro Fold gelernten Standardisierungsparametern.
Sie erreicht auf denselben fünf Folds einen mittleren Macro-F1 von 0,9848,
gegenüber 0,9854 beim ursprünglichen und 0,9807 beim begrenzten Random Forest.
Für die kleine Datenbasis ist sie damit eine ernsthafte, wesentlich einfachere
Alternative. Eine statistische Gleichwertigkeit wird daraus nicht abgeleitet.

Der Random Forest bleibt im Projekt für die geplante Erweiterung mit fachlich
bestätigten Expeditionserhebungen erhalten. Mehr Daten allein begründen jedoch
keine Überlegenheit; die einfachere Alternative sollte bei späteren Datenständen
erneut verglichen werden. Einzelwerte und Grenzen stehen im
[Modellvergleich](modellpruefung/einfaches_modell.md).

## Modelltraining

Das bereits trainierte Modell befindet sich unter:

```text
models/penguin_pipeline.joblib
```

Ein Basismodell kann über:

```bash
python scripts/train_model.py
```

mit der aktuellen Modellkonfiguration aus `data/penguins.csv` neu trainiert und
bewertet werden. Neue Läufe verwenden `max_depth=5`; sie reproduzieren deshalb
nicht die oben dokumentierten Kennzahlen des ursprünglichen Basismodells mit
`max_depth=None`.
Das Skript überschreibt die ursprünglichen Dateien `penguin_pipeline.joblib`,
`model_metadata.json` und `metrics.json` direkt unter `models/`. Es dient der
Erzeugung des Basismodells. Für die Aktualisierung mit bestätigten Beobachtungen
wird der nachfolgende Dashboard-Ablauf verwendet.

Nach einer Modellübernahme bestimmt `models/active_model.json`, welche Version
die Anwendung verwendet. Ein erneuter Aufruf des Basistrainingsskripts ändert
diese Auswahl nicht.

## Manuelles Re-Training und Modellvergleich

Neu erfasste Beobachtungen führen **nicht automatisch zu einem erneuten Training des Modells**.

Ein automatisches Retraining mit den eigenen Modellvorhersagen als Zielvariable könnte bestehende Fehlklassifikationen verstärken.

Stattdessen wird der Prozess im Bereich **Modell aktualisieren** ausdrücklich
ausgelöst:

1. Neue Beobachtungen mit unabhängig fachlich bestätigter Art speichern.
   Der Bereich zeigt, wie viele neue, eindeutige und bestätigte Beobachtungen
   seit der letzten Modellübernahme zur Verfügung stehen.
2. **Re-Training starten** anklicken. Die Anwendung liest die Referenzdaten und
   bestätigten Beobachtungen, prüft sie und trainiert eine neue Modellvariante
   im Hintergrund. Es läuft höchstens ein Trainingsauftrag gleichzeitig.
   Das aktive Modell bleibt währenddessen unverändert verfügbar.
3. Den Vergleich von **Accuracy, Macro-F1 und Cohen's Kappa** prüfen.
   Ergänzende Details zeigen Kreuzvalidierung, Konfusionsmatrizen,
   Klassenverteilung und Hinweise zur Datengrundlage.
4. Nur bei einer fachlich begründeten Entscheidung **Neue Version übernehmen**
   anklicken. Erst dieser Schritt aktiviert die neue Modellversion für
   Vorhersagen. Ein abgeschlossenes Training allein ersetzt das aktive Modell
   nicht.
5. Bei Bedarf **Vorherige Version wiederherstellen** verwenden. Die vorherige
   Version bleibt für diesen manuellen Rückwechsel erhalten.

Es gibt keine zeit- oder mengenabhängige automatische Aktualisierung und keine
Nutzerrollen. Die Anwendung ist für den lokalen Einzelbetrieb vorgesehen;
die fachliche Verantwortung für Labels und Modellübernahme liegt bei der
bedienenden Person.

### Datenprüfung

Für das Re-Training werden ausschließlich die Referenzlabels sowie nicht leere
und gültige Werte aus `validated_species` verwendet. `predicted_species` wird
nie als Trainingslabel übernommen. Unbestätigte Beobachtungen werden nicht
für das Training verwendet.

Bestätigte Datensätze mit ungültigen Messwerten oder Kategorien blockieren den
Trainingsauftrag und müssen korrigiert werden. Fehlendes Geschlecht wird wie
beim Basistraining als `unknown` behandelt. Identische Merkmalskombinationen
mit gleichem Label werden nur einmal berücksichtigt; widersprüchliche Labels
für dieselbe Merkmalskombination werden abgelehnt. Der Fundort bleibt
Kontextinformation und ist kein Modellmerkmal.

Neue Modellvarianten verwenden 500 Bäume, maximal Tiefe 5 und einen festen Zufallsstartwert.
Ein Re-Training führt weder eine Hyperparametersuche noch eine automatische
Klassenumgewichtung durch. Klassenhäufigkeiten und Warnhinweise helfen,
unausgewogene Ergänzungen der kleinen Datengrundlage zu erkennen.

### Vergleichbare Bewertung

Der Modellvergleich verwendet einen festen, stratifizierten Testanteil von
25 % der Referenzdaten. Für die bisherige und die neue Datenbasis werden
**separate Evaluationsmodelle** trainiert. Für die bisherige Variante werden die
Einstellungen ihrer gespeicherten Pipeline übernommen, für die neue Variante
die aktuelle Modellkonfiguration. Gelernte Bäume und Vorverarbeitung werden dabei
verworfen und jeweils neu angepasst. Der Testanteil einschließlich
identischer Merkmalskombinationen wird aus beiden Trainingsmengen ausgeschlossen.
Damit vergleicht die Anwendung zwei Datenstände auf denselben zurückgehaltenen
Beobachtungen; sie bewertet nicht das bereits auf allen Daten trainierte
Auslieferungsmodell mit dessen eigenen Trainingsdaten.

Zusätzlich wird auf dem Trainingsanteil der neuen Variante eine fünffache
stratifizierte Kreuzvalidierung durchgeführt. Erst nach der Bewertung wird das neue
Auslieferungsmodell auf allen geeigneten Daten trainiert. Die Kennzahlen gehören
zu den separat bewerteten Evaluationsmodellen.

Bei wiederholter Nutzung desselben Testanteils können Entscheidungen zunehmend
an diesen Beobachtungen ausgerichtet werden. Er ist deshalb kein dauerhaft
unberührter Abschlusstest. Die Ergebnisse auf dem kleinen Datensatz garantieren
keine gleich hohe Genauigkeit für neue Expeditionserhebungen; neue,
unabhängig bestätigte Prüfdaten bleiben für eine spätere belastbare Bewertung
wichtig.

### Versionen und Wiederherstellung

Jede neu trainierte Variante erhält einen eigenen Ordner unter
`models/versions/<id>/` mit Pipeline, Metadaten, Bewertungskennzahlen,
Trainingsdatenstand und Manifest. Bestehende Versionsdateien werden nicht
überschrieben. Bei der ersten Übernahme wird das Basismodell zusätzlich unter
`models/versions/initial/` gesichert; die ursprünglichen Dateien bleiben erhalten.

Der aktive Verweis in `models/active_model.json` wird atomar ersetzt.
Neue Vorhersagen verwenden dadurch entweder die bisherige oder die vollständig
gespeicherte neue Version. Für Sicherung und Übertragung gehören die CSV-Daten,
der Versionsordner und die Verweisdateien zusammen. Eine fertig trainierte,
noch nicht übernommene Variante wird über `models/pending_candidate.json`
auch nach einem Neustart wieder angezeigt. Ein unterbrochener Trainingslauf
muss neu gestartet werden. Versionsordner und Verweisdateien werden nicht in
ein neu gebautes Basisimage aufgenommen, sondern über die persistenten Ordner
beziehungsweise Volumes erhalten.

## Verwendete Technologien

```text
Python 3.12
Dash
Plotly
pandas
scikit-learn
NumPy
joblib
pytest
Docker
```

## Datenquelle

[Palmer Penguins im UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/690/palmer+penguins-3).
