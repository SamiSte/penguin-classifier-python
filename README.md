# Penguin Classifier

Interaktive und plattformunabhängig bereitstellbare Webanwendung zur Klassifikation der Pinguinarten **Adelie**, **Chinstrap** und **Gentoo** anhand morphologischer Merkmale.

Das Projekt wurde im Rahmen des Kurses **Machine Learning Systems Design** entwickelt. Die Anwendung richtet sich an Wissenschaftler:innen, die neue Pinguinbeobachtungen ohne Kenntnisse in Python oder Machine Learning erfassen und klassifizieren möchten.

## Funktionsumfang

Die Anwendung bietet:

* Eingabe von vier morphologischen Merkmalen:

  * Schnabellänge
  * Schnabeltiefe
  * Flossenlänge
  * Körpergewicht
* Erfassung des Geschlechts, einschließlich unbekannter Angaben
* Erfassung der Insel als Kontextinformation
* Klassifikation in Adelie, Chinstrap oder Gentoo
* Ausgabe der Klassenwahrscheinlichkeiten
* heuristische Einschätzung der Eindeutigkeit der Modellzuordnung
* Warnungen bei Messwerten außerhalb des Trainingsbereichs
* interaktive Visualisierung der Trainingsdaten und der neuen Beobachtung
* persistente Speicherung neuer Beobachtungen als CSV-Datei
* plattformunabhängige Bereitstellung mit Docker
* automatisierte Tests für Datenaufbereitung, Modellservice und Speicherung

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

Für die Mehrklassenklassifikation wird ein **Random Forest Classifier** aus scikit-learn mit 500 Entscheidungsbäumen eingesetzt.

Die Vorverarbeitung und der Klassifikator sind in einer gemeinsamen scikit-learn-Pipeline gekapselt.

Numerische Merkmale werden direkt an den Klassifikator übergeben. Das kategoriale Merkmal `sex` wird innerhalb der Pipeline verarbeitet. Fehlende Geschlechtsangaben werden als `unknown` behandelt und anschließend mittels One-Hot-Encoding kodiert.

Für reproduzierbare Ergebnisse wird ein fester Random State von `42` verwendet.

## Modellbewertung

Die Modellbewertung erfolgt zunächst mit einem stratifizierten **75/25-Train-Test-Split**. Zusätzlich wird auf den Trainingsdaten eine **5-fache stratifizierte Kreuzvalidierung** durchgeführt.

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
├── requirements.txt
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
│   └── penguin_pipeline.joblib
│
├── scripts/
│   ├── explore_data.py
│   ├── train_model.py
│   └── diagnose_island_feature.py
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── figures.py
│   ├── modeling.py
│   ├── model_service.py
│   └── storage.py
│
├── tests/
│   ├── test_data_processing.py
│   ├── test_model_service.py
│   └── test_storage.py
│
└── docs/
```

## Start mit Docker

Docker ist die empfohlene Form der Bereitstellung. Dadurch werden Python und alle benötigten Paketabhängigkeiten innerhalb des Containers gekapselt.

### Image erstellen

Im Projektverzeichnis:

```bash
docker build -t penguin-classifier .
```

### Anwendung starten

```bash
docker run --rm -p 8050:8050 penguin-classifier
```

Anschließend kann die Anwendung im Browser geöffnet werden:

```text
http://localhost:8050
```

## Persistente Speicherung neuer Beobachtungen

Neue Beobachtungen werden in:

```text
data/new_observations.csv
```

gespeichert.

Damit diese Datei auch nach dem Beenden oder Löschen eines Containers erhalten bleibt, wird der lokale `data`-Ordner als Bind-Mount mit dem Container verbunden.

### Windows PowerShell

```powershell
docker run --rm -p 8050:8050 --name penguin-classifier-app --mount "type=bind,source=$($PWD.Path)\data,target=/app/data" penguin-classifier
```

Die gespeicherten Beobachtungen bleiben dadurch direkt im lokalen Projektordner verfügbar.

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

`validated_species` bleibt zunächst leer und ist für eine spätere fachliche Bestätigung der tatsächlichen Art vorgesehen. Dadurch werden Modellvorhersage und verifiziertes Label bewusst voneinander getrennt.

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

Ausführung:

```bash
pytest -v
```

Aktueller Stand:

```text
8 passed
```

## Modelltraining

Das bereits trainierte Modell befindet sich unter:

```text
models/penguin_pipeline.joblib
```

Das Modell kann über:

```bash
python scripts/train_model.py
```

reproduzierbar neu trainiert und bewertet werden.

Dabei werden das Modell sowie Metadaten und Bewertungskennzahlen erneut unter `models/` gespeichert.

## Umgang mit neuen Daten und Retraining

Neu erfasste Beobachtungen führen **nicht automatisch zu einem erneuten Training des Modells**.

Ein automatisches Retraining mit den eigenen Modellvorhersagen als Zielvariable könnte bestehende Fehlklassifikationen verstärken.

Stattdessen ist ein kontrollierter Prozess vorgesehen:

```text
Neue Beobachtung
       ↓
Klassifikation
       ↓
Persistente Speicherung
       ↓
Fachliche Bestätigung der tatsächlichen Art
       ↓
Sammlung validierter neuer Beobachtungen
       ↓
Kontrolliertes Retraining
       ↓
Erneute Modellbewertung
       ↓
Bereitstellung einer neuen Modellversion
```

Dieses Vorgehen ermöglicht eine spätere Erweiterung der Datengrundlage, ohne unbestätigte Modellvorhersagen unmittelbar als Trainingslabels zu verwenden.

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

Palmer Penguins, verfügbar über das UCI Machine Learning Repository:

```text
https://archive.ics.uci.edu/dataset/690/palmer+penguins-3
```
