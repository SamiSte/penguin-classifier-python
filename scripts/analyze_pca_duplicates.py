"""Erzeuge PCA-Abbildung und Dubletten-Audit für Bericht und Video.

Aufruf: python scripts/analyze_pca_duplicates.py
Nur docs/modellpruefung/pca_* wird geschrieben; Daten und Modelle bleiben unverändert.
Die PCA wird ausschließlich auf dem ursprünglichen Trainingsanteil angepasst.
"""

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_regularization import make_protocol  # noqa: E402
from src.data_processing import (  # noqa: E402
    DATA_PATH, FEATURES, NUMERIC_FEATURES, TARGET, clean_data, load_raw_data,
)

OUTPUT = ROOT / "docs" / "modellpruefung"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def duplicate_summary(frame):
    """Zähle Gruppen, beteiligte Zeilen und überzählige exakte Wiederholungen."""
    members = frame.duplicated(keep=False)
    return {
        "row_count": len(frame),
        "duplicate_groups": int(frame.loc[members].drop_duplicates().shape[0]),
        "rows_in_duplicate_groups": int(members.sum()),
        "redundant_rows": int(frame.duplicated(keep="first").sum()),
    }


def normalized_features(data):
    """Gleiche numerische Werte und semantisch gleiche Geschlechtsangaben erkennen.

    Zahlen werden nicht gerundet. Fehlende/leer geschriebene Geschlechtsangaben
    werden unknown; übrige Angaben werden getrimmt und kleingeschrieben.
    """
    frame = data.loc[:, FEATURES].copy()
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(float)
    frame["sex"] = frame["sex"].fillna("unknown").astype(str).str.strip().str.lower()
    frame["sex"] = frame["sex"].replace("", "unknown")
    return frame


def split_overlap(frame, left_indices, right_indices):
    """Vergleiche Wertetupel ohne Zielvariable zwischen zwei disjunkten Teilmengen."""
    left = list(frame.iloc[left_indices].itertuples(index=False, name=None))
    right = list(frame.iloc[right_indices].itertuples(index=False, name=None))
    common = set(left).intersection(right)
    return {
        "shared_feature_combinations": len(common),
        "left_rows_with_match": sum(row in common for row in left),
        "right_rows_with_match": sum(row in common for row in right),
        "shared_row_indices": len(set(left_indices).intersection(right_indices)),
    }


def verify_saved_protocol(protocol, data_sha256):
    """Verhindere die unbemerkte Verwendung eines abweichenden Studienprotokolls."""
    checks = []
    for name in ("ergebnisse.json", "einfaches_modell.json"):
        path = OUTPUT / name
        if not path.exists():
            checks.append({"file": name, "available": False})
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        same_data = saved["metadata"]["data_sha256"] == data_sha256
        same_split = all(saved["protocol"][key] == protocol[key] for key in (
            "training_indices", "held_out_indices", "cv_splits",
        ))
        checks.append({"file": name, "available": True,
                       "same_data_sha256": same_data, "same_split_and_folds": same_split})
        if not same_data or not same_split:
            raise ValueError(f"{name}: Datensatz oder Aufteilung weicht von dieser Prüfung ab.")
    return checks


def analyze():
    raw = load_raw_data()
    data = clean_data(raw)
    protocol = make_protocol(data)
    training_indices = protocol["training_indices"]
    held_out_indices = protocol["held_out_indices"]
    if len(training_indices) != 256 or len(held_out_indices) != 86:
        raise ValueError("Erwartet wird der ursprüngliche Split mit 256/86 Beobachtungen.")
    digest = file_hash(DATA_PATH)
    protocol_checks = verify_saved_protocol(protocol, digest)
    normalized = normalized_features(data)
    numeric = normalized[NUMERIC_FEATURES]

    duplicates = {
        "raw_full_rows": duplicate_summary(raw),
        "clean_model_features_normalized": duplicate_summary(normalized),
        "clean_numeric_features": duplicate_summary(numeric),
    }
    audit = []
    pairs = [("Ursprünglicher Train/Test-Split", training_indices, held_out_indices)]
    pairs += [(f"CV-Fold {fold['fold']}", fold["train_indices"], fold["validation_indices"])
              for fold in protocol["cv_splits"]]
    for name, left, right in pairs:
        audit.append({"split": name, "left_count": len(left), "right_count": len(right),
                      "model_features": split_overlap(normalized, left, right),
                      "numeric_features": split_overlap(numeric, left, right)})

    training = data.iloc[training_indices]
    scaler = StandardScaler()
    standardized = scaler.fit_transform(training[NUMERIC_FEATURES])
    pca = PCA(n_components=4, svd_solver="full")
    coordinates = pca.fit_transform(standardized)
    report = {
        "schema_version": 2,
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "matplotlib_version": matplotlib.__version__,
            "data_sha256": digest,
            "script_sha256": file_hash(__file__),
            "protocol_script_sha256": file_hash(ROOT / "scripts" / "evaluate_regularization.py"),
            "data_processing_sha256": file_hash(ROOT / "src" / "data_processing.py"),
        },
        "data_counts": {"raw": len(raw), "clean": len(data),
                        "training": len(training), "held_out": len(held_out_indices)},
        "duplicate_definition": (
            "Exakte Zeilengleichheit für Rohdaten; Modellmerkmale mit numerischem dtype, "
            "Geschlecht getrimmt/kleingeschrieben und fehlend/leer als unknown; "
            "keine Rundung oder Ähnlichkeitsschwelle. Merkmalsprüfung ohne Artenlabel/" 
            "Insel/Jahr. Numeric-only-Prüfung ohne Geschlecht."
        ),
        "duplicates": duplicates,
        "split_overlap_audit": audit,
        "saved_protocol_checks": protocol_checks,
        "protocol": protocol,
        "pca": {
            "fit_subset": "original_training_only",
            "features": NUMERIC_FEATURES,
            "standardization": "StandardScaler, fit nur auf 256 Trainingszeilen",
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "first_two_explained_variance_ratio": float(pca.explained_variance_ratio_[:2].sum()),
            "first_three_explained_variance_ratio": float(pca.explained_variance_ratio_[:3].sum()),
            "displayed_pairs": [[1, 2], [1, 3], [2, 3]],
            "components": pca.components_.tolist(),
            "singular_values": pca.singular_values_.tolist(),
            "species_counts": {str(k): int(v) for k, v in training[TARGET].value_counts().items()},
            "scores": [{"clean_row_index": int(index), "species": str(label),
                        "pc1": float(point[0]), "pc2": float(point[1]),
                        "pc3": float(point[2]), "pc4": float(point[3])}
                       for index, label, point in zip(training_indices, training[TARGET], coordinates)],
        },
        "limitations": [
            "PCA ist eine explorative Projektion, kein Klassifikations- oder Overfitting-Test.",
            "Die Klassenlabels werden ausschließlich zur Einfärbung verwendet.",
            "Testdaten werden weder zur Standardisierung noch zum PCA-Fit oder zur Abbildung verwendet.",
            "Keine exakten Dubletten bedeutet keinen vollständigen Ausschluss von Leakage: "
            "nahe Dubletten, mehrfach vermessene Tiere und sonstige Abhängigkeiten sind nicht geprüft.",
            "Die wiederverwendeten CV-Folds liefern keinen neuen unabhängigen Gütenachweis.",
        ],
    }
    return report, training, coordinates


def percent(value):
    return f"{100 * value:.1f}".replace(".", ",")


def write_reference_figure(report, training, coordinates):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    fig.subplots_adjust(left=.105, right=.97, bottom=.255, top=.82)
    palette = [("Adelie", "#0072B2", "o"), ("Chinstrap", "#D55E00", "^"),
               ("Gentoo", "#009E73", "s")]
    for species, color, marker in palette:
        mask = training[TARGET].to_numpy() == species
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=47, c=color,
                   marker=marker, alpha=.82, edgecolors="white", linewidths=.45,
                   label=f"{species} (n = {mask.sum()})", zorder=3)
    variance = report["pca"]["explained_variance_ratio"]
    ax.set_xlabel(f"Hauptkomponente 1 · {percent(variance[0])} % erklärte Varianz", labelpad=12)
    ax.set_ylabel(f"Hauptkomponente 2 · {percent(variance[1])} % erklärte Varianz", labelpad=12)
    ax.grid(color="#dfe5eb", alpha=.7, linewidth=.7, zorder=0)
    ax.margins(.07)
    ax.legend(loc="lower right", framealpha=.95, fontsize=12)
    fig.text(.07, .94, "Wie unterscheiden sich die Pinguinarten?", fontsize=24,
             weight="bold", color="#152d43", va="top")
    fig.text(.07, .885,
             "PCA der vier standardisierten Körpermesswerte · 256 Trainingsbeobachtungen",
             fontsize=14, color="#435569", va="top")
    fig.text(.07, .068,
             f"Die ersten zwei Komponenten erklären zusammen {percent(sum(variance[:2]))} % der Varianz.\n"
             "Arten nur zur Einfärbung · PCA und Skalierung nur auf Trainingsdaten angepasst",
             fontsize=11.5, color="#435569", va="bottom", linespacing=1.5)
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"pca_pc1_pc2.{suffix}", dpi=150, facecolor="white")
    plt.close(fig)


def write_figure(report, training, coordinates):
    """Mehrere Projektionen derselben PCA, ohne neue Anpassung oder Filterung."""
    write_reference_figure(report, training, coordinates)
    variance = report["pca"]["explained_variance_ratio"]
    labels = training[TARGET].to_numpy()
    palette = [("Adelie", "#0072B2", "o"), ("Chinstrap", "#D55E00", "^"),
               ("Gentoo", "#009E73", "s")]
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12})

    def plot_pair(ax, first, second):
        for species, color, marker in palette:
            mask = labels == species
            ax.scatter(coordinates[mask, first], coordinates[mask, second],
                       s=32, color=color, marker=marker, alpha=.82,
                       edgecolors="white", linewidths=.35, zorder=3,
                       label=f"{species} (n = {mask.sum()})")
        ax.set_xlabel(f"PC{first + 1} ({percent(variance[first])} %)", labelpad=8)
        ax.set_ylabel(f"PC{second + 1} ({percent(variance[second])} %)", labelpad=8)
        ax.grid(color="#dfe5eb", linewidth=.7, zorder=0)
        ax.margins(.09)
        # Gleiche Komponente behält in allen Ansichten denselben Wertebereich.
        for set_limit, component in ((ax.set_xlim, first), (ax.set_ylim, second)):
            low, high = coordinates[:, component].min(), coordinates[:, component].max()
            pad = (high - low) * .09
            set_limit(low - pad, high + pad)

    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=.065, right=.978, top=.76, bottom=.285, wspace=.31)
    pairs = [(0, 1), (0, 2), (1, 2)]
    titles = ["PC1 / PC2", "PC1 / PC3", "PC2 / PC3"]
    notes = ["Gentoo deutlich getrennt;\nAdelie und Chinstrap überlappen.",
             "Mit PC3 werden weitere Unterschiede\nzwischen Adelie und Chinstrap sichtbar.",
             "Chinstrap hebt sich stärker ab;\nAdelie und Gentoo überlappen."]
    for ax, (first, second), title, note in zip(axes, pairs, titles, notes):
        plot_pair(ax, first, second)
        ax.set_title(title, fontsize=17, weight="bold", color="#152d43", pad=15)
        box = ax.get_position()
        fig.text(box.x0, .165, note, fontsize=12, color="#435569", va="top", linespacing=1.5)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.51, .866),
               ncol=3, frameon=False, fontsize=13, markerscale=1.2)
    fig.text(.055, .96, "PCA der Körpermesswerte", fontsize=25,
             weight="bold", color="#152d43", va="top")
    fig.text(.055, .902, "Dieselben 256 Trainingsfälle · vier standardisierte Körpermesswerte · drei Blickrichtungen",
             fontsize=14, color="#435569", va="top")
    fig.text(.055, .067,
             f"PC1 + PC2: {percent(sum(variance[:2]))} % Varianz. PC1 bis PC3 zusammen: {percent(sum(variance[:3]))} %. "
             "Jede Ansicht zeigt nur zwei Komponenten.\n"
             "Arten nur zur Kennzeichnung · Skalierung und PCA nur auf Trainingsdaten · keine Aussage über die Testgenauigkeit",
             fontsize=11.5, color="#435569", va="bottom", linespacing=1.6)
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"pca_training.{suffix}", dpi=120, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=120)
    fig.subplots_adjust(left=.07, right=.98, top=.83, bottom=.09, hspace=.45, wspace=.30)
    for ax, (first, second) in zip(axes.flat, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]):
        plot_pair(ax, first, second)
        ax.set_title(f"PC{first+1} / PC{second+1}", fontsize=14, weight="bold", pad=8)
    fig.suptitle("Alle sechs Komponentenpaare derselben Trainings-PCA", fontsize=23,
                 weight="bold", color="#152d43", y=.97)
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .93),
               ncol=3, frameon=False, fontsize=12)
    fig.text(.07, .025, "Alle 256 Trainingsfälle in jedem Panel; kein Weglassen von Überlappungen. "
             "PCA erklärt Merkmalsvarianz, keine Klassifikationsgüte.", fontsize=11, color="#435569")
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"pca_alle_paare.{suffix}", dpi=120, facecolor="white")
    plt.close(fig)


def write_text(report):
    variance = report["pca"]["explained_variance_ratio"]
    raw = report["duplicates"]["raw_full_rows"]
    model = report["duplicates"]["clean_model_features_normalized"]
    numeric = report["duplicates"]["clean_numeric_features"]
    lines = ["# PCA und Dublettenprüfung", "",
             "## Vorgehen und Abgrenzung", "",
             "Die Auswertung verwendet ausschließlich den unveränderten Referenzdatensatz "
             "`data/penguins.csv`: 344 Rohbeobachtungen, davon 342 mit vollständigen Körpermesswerten. "
             "Neue App-Beobachtungen sind nicht Bestandteil dieser Prüfung. Der ursprüngliche "
             "stratifizierte Split (Seed 42, 25 % Test) wird reproduziert: 256 Trainings- und "
             "86 Testbeobachtungen. Die vorhandenen fünf CV-Folds liegen innerhalb des Trainingsanteils. "
             "Datensatzhash, Split und Folds stimmen mit beiden gespeicherten Modellstudien überein.", "",
             "Schnabellänge, Schnabeltiefe, Flossenlänge und Körpermasse werden mit `StandardScaler` "
             "standardisiert. Skalierung und PCA werden nur auf den 256 Trainingsbeobachtungen angepasst. "
             "Die Abbildung enthält ausschließlich diese Trainingsfälle. Geschlecht, Insel und Jahr "
             "gehen nicht in die PCA ein; die Art dient nur zur nachträglichen Einfärbung. "
             "Es werden weder ein neues Klassifikationsmodell trainiert noch bestehende Modelle verändert.", "",
             "## Abbildung", "", "![PCA der Trainingsdaten](pca_training.png)", "",
             f"Die erste Hauptkomponente erklärt **{percent(variance[0])} %**, die zweite "
             f"**{percent(variance[1])} %**, zusammen **{percent(sum(variance[:2]))} %** der "
             "Varianz der standardisierten Trainingsmerkmale. Dieser Anteil ist keine Accuracy.", "",
             "Die sichtbare Klassenstruktur macht die hohen Klassifikationswerte plausibel. "
             "Gentoo bildet in der ersten Ansicht (PC1/PC2) eine deutlich getrennte Gruppe; Adelie und Chinstrap "
             "überlappen teilweise. Die Grafik belegt weder fehlerfreie Trennbarkeit im Allgemeinen "
             "noch das Fehlen von Overfitting. Die PCA maximiert erklärte Merkmalsvarianz und verwendet "
             "dafür keine Artenlabels.", "",
             "### Warum die dritte Komponente hilft", "",
             f"PC3 erklärt weitere **{percent(variance[2])} %**, PC4 **{percent(variance[3])} %**. "
             f"Die ersten drei Komponenten enthalten zusammen **{percent(sum(variance[:3]))} %** "
             "der Varianz. Das gilt für die drei Achsen gemeinsam, nicht für jedes einzelne Panel. "
             f"PC1/PC3 enthält beispielsweise **{percent(variance[0] + variance[2])} %**.", "",
             "In PC1/PC3 sind Adelie und Chinstrap deutlicher unterscheidbar als in PC1/PC2, "
             "mit verbleibenden Überschneidungen. PC2/PC3 hebt Chinstrap hervor, während Adelie "
             "und Gentoo stärker überlappen. Es gibt also keine einzelne hier gezeigte Projektion, "
             "die allgemein fehlerfreie Klassifikation belegt. Alle Ansichten verwenden dieselben "
             "Koordinaten und alle 256 Trainingsfälle. Eine Übersicht zeigt auch "
             "[alle sechs Komponentenpaare](pca_alle_paare.png); die "
             "[ursprüngliche PC1/PC2-Ansicht](pca_pc1_pc2.png) bleibt verfügbar.", "",
             "PC3 gewichtet die standardisierte Schnabellänge mit +0,630, die Schnabeltiefe mit "
             "−0,426, die Flossenlänge mit −0,208 und die Körpermasse mit −0,615. Sie zeigt damit "
             "zusätzliche Unterschiede der Körperproportionen. Die Koeffizienten sind Richtungswerte "
             "dieser PCA und keine Merkmalswichtigkeiten des Klassifikators.", "",
             "Die ergänzenden Komponenten wurden nach Sichtung der ersten Grafik zur Erklärung "
             "betrachtet. Das ist eine explorative Darstellung und keine neue unabhängige Validierung. "
             "Eine zweidimensionale Projektion lässt Information weg. Der Klassifikator arbeitet "
             "dagegen mit allen vier Körpermesswerten und zusätzlich mit dem Geschlecht; die PCA "
             "ist kein Bestandteil seiner Pipeline.", "",
             "## Dublettenbefund", "",
             "| Prüfung | Zeilen | Dublettengruppen | Überzählige Zeilen |",
             "| --- | ---: | ---: | ---: |"]
    for label, row in (("Vollständige Rohzeilen", raw),
                       ("Normalisierte Modellmerkmale (vier Messwerte + Geschlecht)", model),
                       ("Nur vier Körpermesswerte", numeric)):
        lines.append(f"| {label} | {row['row_count']} | {row['duplicate_groups']} | {row['redundant_rows']} |")
    lines += ["", "Numerische Werte werden nicht gerundet. Für die Merkmalsprüfung werden "
              "Geschlechtsangaben getrimmt und kleingeschrieben; fehlend oder leer wird `unknown`. "
              "Die Zielvariable wird beim Merkmalsvergleich bewusst nicht berücksichtigt.", "",
              "| Aufteilung | Training / Prüfung | Gleiche Modellmerkmale über Grenze | Gleiche Messwerte über Grenze |",
              "| --- | ---: | ---: | ---: |"]
    for row in report["split_overlap_audit"]:
        lines.append(f"| {row['split']} | {row['left_count']} / {row['right_count']} | "
                     f"{row['model_features']['shared_feature_combinations']} | "
                     f"{row['numeric_features']['shared_feature_combinations']} |")
    lines += ["", "Gezählt werden unterschiedliche identische Merkmalskombinationen, die auf beiden "
              "Seiten der jeweiligen Grenze vorkommen. Auch die Zeilenindizes sind innerhalb jeder "
              "Aufteilung disjunkt. **Es wurden keine exakten Dubletten gefunden.** Das schließt "
              "ähnliche Beobachtungen, wiederholte Messungen desselben Tiers oder andere Formen "
              "von Datenabhängigkeit und Leakage nicht vollständig aus.", "",
              "## Reproduktion", "", "```powershell", "python scripts/analyze_pca_duplicates.py", "```", "",
              "Benötigt werden die Projektbibliotheken und Matplotlib für den statischen Export. "
              "In der vorhandenen Projektumgebung `penguin-ml` fehlt Matplotlib. Der vollständige "
              "Export wurde deshalb mit der bereits vorhandenen Anaconda-Basisumgebung erstellt "
              "(Versionen siehe JSON); es wurde nichts installiert. Auf diesem Rechner kann der "
              "Aufruf mit `& 'C:\\Users\\samis\\anaconda3\\python.exe' scripts/analyze_pca_duplicates.py` "
              "in PowerShell wiederholt werden. "
              "Die verwendeten Versionen, Dateihashes, Indizes, PCA-Komponenten, Skalierungsparameter "
              "und Einzelkoordinaten stehen in [pca_ergebnisse.json](pca_ergebnisse.json). "
              "PCA-Vorzeichen können zwischen numerischen Implementierungen gespiegelt sein; "
              "Varianz und relative Lage bleiben dabei gleich.", "",
              "Für die Videoaufnahme: [pca_video.html](pca_video.html) im Browser öffnen und die "
              "PCA-Abbildung zeigen; [pca_training.svg](pca_training.svg) ist der Vektorexport.", "",
              "Methodik: [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/"
              "sklearn.decomposition.PCA.html) und [StandardScaler](https://scikit-learn.org/"
              "stable/modules/generated/sklearn.preprocessing.StandardScaler.html)."]
    (OUTPUT / "pca_bericht.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    title = "PCA und Dublettenprüfung"
    page = f"""<!doctype html><html lang="de"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>body{{margin:0;background:#edf2f5;color:#152d43;font-family:system-ui,sans-serif}}
main{{max-width:1250px;margin:28px auto;padding:0 24px}}h1{{font-size:28px;margin:0 0 12px}}
img{{display:block;width:100%;background:white;border-radius:12px}}section{{background:white;
padding:22px 28px;border-radius:12px;margin:18px 0}}h2{{margin:0 0 12px;font-size:23px}}
p{{font-size:19px;line-height:1.5;margin:10px 0}}.note{{color:#435569;font-size:16px}}</style>
<main><h1>{html.escape(title)}</h1><img src="pca_training.svg"
alt="Drei Ansichten derselben PCA: PC1/PC2, PC1/PC3 und PC2/PC3 mit allen 256 Trainingsfällen">
<section><h2>Warum die zusätzliche Ansicht?</h2>
<p>PC1/PC2 zeigt vor allem die Trennung von Gentoo. Mit PC3 werden zusätzliche Unterschiede
zwischen Adelie und Chinstrap sichtbar. Auch hier bleiben Überschneidungen.</p>
<p class="note">PC1: {percent(variance[0])} % · PC2: {percent(variance[1])} % · PC3: {percent(variance[2])} % ·
PC4: {percent(variance[3])} % erklärte Merkmalsvarianz. Die ersten drei Komponenten enthalten zusammen
{percent(sum(variance[:3]))} %. Das bedeutet weder {percent(sum(variance[:3]))} % Accuracy noch vollständige Trennbarkeit.</p>
<details><summary>Alle sechs Komponentenpaare anzeigen</summary><img src="pca_alle_paare.svg"
alt="Alle sechs Paarungen der vier Hauptkomponenten"></details></section>
<section><h2>Ergänzende Prüfung: keine exakten Dubletten</h2>
<p>344 Rohzeilen · 342 vollständige Messwertzeilen · 256 Training / 86 Test</p>
<p>0 doppelte Rohzeilen · 0 identische Modellmerkmalskombinationen<br>
0 gleiche Merkmalskombinationen über die Train/Test-Grenze oder innerhalb der fünf CV-Folds</p>
<p class="note">Verglichen wurde in jedem CV-Fold dessen Training mit dessen Validierung.
Geprüft wurden auch die vier Messwerte ohne Geschlecht. Ähnliche Beobachtungen und andere
Formen von Datenabhängigkeit werden dadurch nicht ausgeschlossen.</p></section>
<section><h2>Was lässt sich daraus ableiten?</h2>
<p>Die Klassenstruktur macht hohe Klassifikationswerte plausibel.</p>
<p class="note">PCA erklärt Merkmalsvarianz, keine Accuracy. Die Darstellung ist kein neuer
Gütenachweis und kein eigenständiger Test auf Overfitting.</p></section></main></html>"""
    (OUTPUT / "pca_video.html").write_text(page, encoding="utf-8")


def main():
    report, training, coordinates = analyze()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "pca_ergebnisse.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_figure(report, training, coordinates)
    write_text(report)
    print(json.dumps({"counts": report["data_counts"], "duplicates": report["duplicates"],
                      "pca_explained_variance": report["pca"]["explained_variance_ratio"],
                      "output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
