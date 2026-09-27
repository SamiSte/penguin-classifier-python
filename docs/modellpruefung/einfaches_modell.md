# Vergleich mit einem einfacheren Modell

Identische fünf stratifizierte CV-Folds auf 256 Trainingsbeobachtungen. Die 86 ursprünglichen Testbeobachtungen werden nicht verwendet. Die Standardisierung der logistischen Regression wird pro Trainingsfold gelernt. C = 1 und die L2-Regularisierung bleiben unverändert; es erfolgt keine Parametersuche.

| Modell | CV Accuracy | CV Macro-F1 | CV Kappa | Blätter je Baum |
| --- | ---: | ---: | ---: | ---: |
| Random Forest ohne Tiefenbegrenzung | 0,9882 ± 0,0157 | 0,9854 ± 0,0194 | 0,9814 ± 0,0247 | 14,8 |
| Random Forest mit maximaler Tiefe 5 | 0,9843 ± 0,0192 | 0,9807 ± 0,0236 | 0,9753 ± 0,0302 | 11,2 |
| Logistische Regression mit L2-Regularisierung | 0,9882 ± 0,0157 | 0,9848 ± 0,0203 | 0,9812 ± 0,0251 | – |

Die Streuung beschreibt die fünf Folds und ist kein Konfidenzintervall. Die gleiche Kreuzvalidierung wurde bereits für die Wahl der Baumtiefe verwendet. Der Vergleich ist daher explorativ und kein neuer unabhängiger Gütenachweis. Mehr künftige Beobachtungen allein begründen keine Überlegenheit des Random Forest. Die einfachere Alternative sollte bei späteren Datenständen erneut mitgeführt werden.

Vollständige Einzelwerte und Aufteilungen: [einfaches_modell.json](einfaches_modell.json).

Methodischer Bezug: [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) und [logistische Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
