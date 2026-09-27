# Random-Forest-Modellprüfung

Empfehlung nach der vorab festgelegten Auswahlregel: Baumtiefe 5, mindestens 1 je Blatt. Der mittlere Macro-F1 in der Kreuzvalidierung beträgt 98,07 %; die Bäume haben im Mittel 21,4 Knoten. Die Studie verändert das aktive App-Modell nicht.

Auswahlregel: Berücksichtigt werden Varianten, deren mittlerer Validierungs-Macro-F1 höchstens 0,50 Prozentpunkte unter dem besten Wert liegt. Dieser beträgt 98,54 % (Baumtiefe unbegrenzt, mindestens 1 je Blatt). Innerhalb dieser Gruppe wird die Variante mit der kleinsten mittleren Knotenzahl gewählt. Die Toleranz ist eine praktische Entscheidungsregel und kein Nachweis statistischer Gleichwertigkeit.

Beim bisherigen Modell (Baumtiefe unbegrenzt, mindestens 1 je Blatt) liegt der mittlere Trainings-Macro-F1 bei 100,00 %, der mittlere Validierungs-Macro-F1 bei 98,54 %. Die Differenz beträgt 1,46 Prozentpunkte. Die Trainingsleistung allein belegt keine Überanpassung; maßgeblich sind die Leistung auf zurückgehaltenen Daten und deren Streuung.

## Vergleich der Varianten

Mittelwert ± Standardabweichung über die Kreuzvalidierungsfolds. PP = Prozentpunkte.

| Max. Tiefe | Min. je Blatt | Training Accuracy | CV Accuracy | Training Macro-F1 | CV Macro-F1 | Knoten je Baum | Auswahl |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unbegrenzt | 1 | 100,00 % ± 0,00 PP | 98,82 % ± 1,57 PP | 100,00 % ± 0,00 PP | 98,54 % ± 1,94 PP | 28,5 |  |
| unbegrenzt | 2 | 99,61 % ± 0,20 PP | 98,43 % ± 1,92 PP | 99,52 % ± 0,24 PP | 98,07 % ± 2,36 PP | 24,2 |  |
| unbegrenzt | 4 | 99,51 % ± 0,31 PP | 97,65 % ± 2,88 PP | 99,40 % ± 0,38 PP | 97,00 % ± 3,68 PP | 18,8 |  |
| 5 | 1 | 99,61 % ± 0,20 PP | 98,43 % ± 1,92 PP | 99,52 % ± 0,24 PP | 98,07 % ± 2,36 PP | 21,4 | Empfehlung |
| 5 | 2 | 99,51 % ± 0,31 PP | 98,04 % ± 2,48 PP | 99,40 % ± 0,38 PP | 97,55 % ± 3,10 PP | 19,7 |  |
| 5 | 4 | 99,51 % ± 0,31 PP | 97,65 % ± 2,88 PP | 99,40 % ± 0,38 PP | 97,00 % ± 3,68 PP | 17,2 |  |
| 10 | 1 | 100,00 % ± 0,00 PP | 98,82 % ± 1,57 PP | 100,00 % ± 0,00 PP | 98,54 % ± 1,94 PP | 28,3 |  |
| 10 | 2 | 99,61 % ± 0,20 PP | 98,43 % ± 1,92 PP | 99,52 % ± 0,24 PP | 98,07 % ± 2,36 PP | 24,2 |  |
| 10 | 4 | 99,51 % ± 0,31 PP | 97,65 % ± 2,88 PP | 99,40 % ± 0,38 PP | 97,00 % ± 3,68 PP | 18,8 |  |

## Lernkurve des bisherigen Modells

Die interaktive Grafik befindet sich in [bericht.html](bericht.html). Die Trainingsgröße ist der Mittelwert der tatsächlich verwendeten Fold-Größen. Die Y-Achse zeigt den Bereich 90–102 %, damit kleine Unterschiede sichtbar werden.

| Training je Fold (Ø) | Training Accuracy | CV Accuracy | Training Macro-F1 | CV Macro-F1 |
| --- | --- | --- | --- | --- |
| 41,0 | 100,00 % ± 0,00 PP | 97,25 % ± 2,93 PP | 100,00 % ± 0,00 PP | 96,38 % ± 3,93 PP |
| 81,6 | 100,00 % ± 0,00 PP | 97,26 % ± 2,94 PP | 100,00 % ± 0,00 PP | 96,66 % ± 3,59 PP |
| 123,2 | 100,00 % ± 0,00 PP | 98,05 % ± 1,75 PP | 100,00 % ± 0,00 PP | 97,66 % ± 2,10 PP |
| 163,8 | 100,00 % ± 0,00 PP | 98,04 % ± 1,75 PP | 100,00 % ± 0,00 PP | 97,66 % ± 2,10 PP |
| 204,8 | 100,00 % ± 0,00 PP | 98,82 % ± 1,57 PP | 100,00 % ± 0,00 PP | 98,54 % ± 1,94 PP |

Bei durchschnittlich 41,0 Trainingsbeobachtungen je Fold beträgt der Validierungs-Macro-F1 96,38 %; bei 204,8 Beobachtungen sind es 98,54 %. Die Differenz zwischen Training und Validierung verändert sich dabei von 3,62 auf 1,46 Prozentpunkte. Diese Werte beschreiben nur den untersuchten Datenbereich; ein weiterer Leistungsgewinn durch neue Daten ist damit nicht garantiert.

## Vorgehen und Aussagekraft

Verglichen werden neun Kombinationen aus maximaler Baumtiefe (unbegrenzt, 5, 10) und Mindestanzahl je Blatt (1, 2, 4), jeweils mit 500 Bäumen und identischen 5 stratifizierten Folds. Alle Vorverarbeitungsschritte werden innerhalb des jeweiligen Trainingsfolds gelernt. Die Lernkurve verwendet das bisherige Modell und verschachtelte stratifizierte Teilmengen innerhalb dieser Trainingsfolds.

Von 342 vollständigen Beobachtungen werden nur die 256 Fälle des ursprünglichen Trainingsanteils verwendet (Zufallsstartwert 42). Die 86 bereits bekannten Testfälle werden in dieser Studie weder trainiert noch erneut bewertet und gehen nicht in die Modellauswahl ein.

Die Angaben ± Standardabweichung beschreiben die Streuung über die Folds und sind keine Konfidenzintervalle. Die Trainingsanteile überlappen sich. Die Auswahl anhand derselben Kreuzvalidierung kann die berichtete Leistung des ausgewählten Modells optimistisch erscheinen lassen. Dieser kleine, explorative Vergleich liefert keinen neuen unabhängigen Gütenachweis. Für neue Expeditionen oder andere Populationen fehlen unabhängig erhobene Prüfdaten.

## Nachvollziehbarkeit

Erzeugt (UTC): 2026-09-26T17:35:05.673224+00:00. Python 3.12.13; scikit-learn 1.9.0.

Dateiprüfsummen (SHA-256):

- data_sha256: `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93`
- script_sha256: `2ff825e53b3d12830e1432a8e8f547817c6f479313ee06726cfcb00b91bc07bd`
- modeling_sha256: `6629192d0d9f5ddaa8d17e2dd4dda2115d3afb321f0c75866526d06ea2a6d0ae`

Rohwerte: [Vergleich.csv](Vergleich.csv) und [lernkurve.csv](lernkurve.csv). CSV-Metriken sind Anteile zwischen 0 und 1; als Trennzeichen dient das Semikolon.

## Methodische Quellen

Methodischer Bezug: IU-Skript Model Engineering, Abschnitt 6.1, gedruckte Seiten 96–97 (Überanpassung und Regularisierung). Die konkrete Parameterwahl ist eine eigene Prüfung für diesen Datensatz.

- [scikit-learn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [scikit-learn: Lernkurven](https://scikit-learn.org/stable/modules/learning_curve.html)
- [scikit-learn: Kreuzvalidierung](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn: Verzerrung bei der Modellauswahl](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
