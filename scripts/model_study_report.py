"""Offline-Berichte für den getrennten Random-Forest-Vergleich erstellen."""

from __future__ import annotations

import csv
from html import escape
from pathlib import Path
from statistics import mean, pstdev

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _percent(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}".replace(".", ",") + " %"


def _number(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _label(variant: dict) -> str:
    depth = "unbegrenzt" if variant["max_depth"] is None else variant["max_depth"]
    return f"Baumtiefe {depth}, mindestens {variant['min_samples_leaf']} je Blatt"


def _score(summary: dict, metric: str) -> str:
    return (
        f"{_percent(summary[metric + '_mean'])} "
        f"± {_number(100 * summary[metric + '_std'], 2)} PP"
    )


def _learning_summary(rows: list[dict]) -> list[dict]:
    summaries = []
    for fraction in sorted({row["fraction"] for row in rows}):
        group = [row for row in rows if row["fraction"] == fraction]
        summary = {
            "fraction": fraction,
            "n_train_mean": mean(row["n_train"] for row in group),
        }
        for metric in (
            "train_accuracy", "validation_accuracy", "train_macro_f1",
            "validation_macro_f1",
        ):
            values = [row[metric] for row in group]
            summary[metric + "_mean"] = mean(values)
            summary[metric + "_std"] = pstdev(values)
        summaries.append(summary)
    return summaries


def _learning_figure(summaries: list[dict]) -> go.Figure:
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Macro-F1"))
    for column, metric in enumerate(("accuracy", "macro_f1"), start=1):
        for phase, label, color in (
            ("train", "Training", "#355e7c"),
            ("validation", "Validierung", "#ad7029"),
        ):
            key = f"{phase}_{metric}"
            figure.add_trace(
                go.Scatter(
                    x=[row["n_train_mean"] for row in summaries],
                    y=[row[key + "_mean"] for row in summaries],
                    mode="lines+markers",
                    name=label,
                    legendgroup=phase,
                    showlegend=column == 1,
                    line={"color": color, "width": 2},
                    marker={"size": 6},
                    error_y={
                        "type": "data",
                        "array": [row[key + "_std"] for row in summaries],
                        "visible": True,
                        "thickness": 1,
                        "width": 3,
                    },
                    hovertemplate=(
                        "Mittlere Trainingsgröße: %{x:.1f}<br>"
                        + label + ": %{y:.2%}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
    figure.update_xaxes(title_text="Beobachtungen je Trainingsfold (Mittelwert)")
    figure.update_yaxes(range=[0.90, 1.02], tickformat=".0%", title_text="Leistung")
    figure.update_layout(
        template="plotly_white",
        height=420,
        margin={"l": 55, "r": 25, "t": 45, "b": 80},
        font={"family": "Arial, sans-serif", "size": 13, "color": "#26323b"},
        legend={"orientation": "h", "y": -0.23, "x": 0.35},
    )
    return figure


def _narrative(study: dict) -> list[str]:
    protocol = study["protocol"]
    selection = study["selection"]
    variants = {variant["name"]: variant for variant in study["variants"]}
    baseline = next(
        variant for variant in study["variants"]
        if variant["max_depth"] is None and variant["min_samples_leaf"] == 1
    )
    recommended = variants[selection["recommended_variant"]]
    best = variants[selection["best_score_variant"]]
    baseline_summary = baseline["summary"]
    gap = (
        baseline_summary["train_macro_f1_mean"]
        - baseline_summary["validation_macro_f1_mean"]
    )
    recommended_summary = recommended["summary"]
    return [
        (
            f"Empfehlung nach der vorab festgelegten Auswahlregel: {_label(recommended)}. "
            f"Der mittlere Macro-F1 in der Kreuzvalidierung beträgt "
            f"{_percent(recommended_summary['validation_macro_f1_mean'])}; "
            f"die Bäume haben im Mittel {_number(recommended_summary['tree_nodes_mean'])} "
            f"Knoten. Die Studie verändert das aktive App-Modell nicht."
        ),
        (
            f"Auswahlregel: Berücksichtigt werden Varianten, deren mittlerer "
            f"Validierungs-Macro-F1 höchstens "
            f"{_number(100 * selection['tolerance'], 2)} Prozentpunkte unter dem besten "
            f"Wert liegt. Dieser beträgt "
            f"{_percent(best['summary']['validation_macro_f1_mean'])} "
            f"({_label(best)}). Innerhalb dieser Gruppe wird die Variante mit der "
            f"kleinsten mittleren Knotenzahl gewählt. Die Toleranz ist eine praktische "
            f"Entscheidungsregel und kein Nachweis statistischer Gleichwertigkeit."
        ),
        (
            f"Beim bisherigen Modell ({_label(baseline)}) liegt der mittlere "
            f"Trainings-Macro-F1 bei {_percent(baseline_summary['train_macro_f1_mean'])}, "
            f"der mittlere Validierungs-Macro-F1 bei "
            f"{_percent(baseline_summary['validation_macro_f1_mean'])}. "
            f"Die Differenz beträgt {_number(100 * gap, 2)} Prozentpunkte. "
            f"Die Trainingsleistung allein belegt keine Überanpassung; maßgeblich "
            f"sind die Leistung auf zurückgehaltenen Daten und deren Streuung."
        ),
        (
            f"Verglichen werden neun Kombinationen aus maximaler Baumtiefe "
            f"(unbegrenzt, 5, 10) und Mindestanzahl je Blatt (1, 2, 4), jeweils mit "
            f"{protocol['n_estimators']} Bäumen und identischen "
            f"{protocol['cv_folds']} stratifizierten Folds. Alle Vorverarbeitungsschritte "
            f"werden innerhalb des jeweiligen Trainingsfolds gelernt. Die "
            f"Lernkurve verwendet das bisherige Modell und verschachtelte "
            f"stratifizierte Teilmengen innerhalb dieser Trainingsfolds."
        ),
        (
            f"Von {protocol['total_count']} vollständigen Beobachtungen werden nur "
            f"die {protocol['training_count']} Fälle des ursprünglichen Trainingsanteils "
            f"verwendet (Zufallsstartwert {protocol['seed']}). Die "
            f"{protocol['held_out_count']} bereits bekannten Testfälle werden in "
            f"dieser Studie weder trainiert noch erneut bewertet und gehen nicht "
            f"in die Modellauswahl ein."
        ),
        (
            "Die Angaben ± Standardabweichung beschreiben die Streuung über die Folds "
            "und sind keine Konfidenzintervalle. Die Trainingsanteile überlappen sich. "
            "Die Auswahl anhand derselben Kreuzvalidierung kann die berichtete "
            "Leistung des ausgewählten Modells optimistisch erscheinen lassen. "
            "Dieser kleine, explorative Vergleich liefert keinen neuen unabhängigen "
            "Gütenachweis. Für neue Expeditionen oder andere Populationen fehlen "
            "unabhängig erhobene Prüfdaten."
        ),
    ]


def _comparison_rows(study: dict) -> list[list[str]]:
    recommended = study["selection"]["recommended_variant"]
    rows = []
    for variant in study["variants"]:
        summary = variant["summary"]
        depth = "unbegrenzt" if variant["max_depth"] is None else str(variant["max_depth"])
        rows.append([
            depth,
            str(variant["min_samples_leaf"]),
            _score(summary, "train_accuracy"),
            _score(summary, "validation_accuracy"),
            _score(summary, "train_macro_f1"),
            _score(summary, "validation_macro_f1"),
            _number(summary["tree_nodes_mean"]),
            "Empfehlung" if variant["name"] == recommended else "",
        ])
    return rows


def write_report(study: dict, output_dir: Path) -> None:
    """Schreibe HTML, Markdown und zwei CSV-Tabellen ohne Modelländerungen.

    Der interaktive HTML-Bericht enthält Plotly-JavaScript vollständig inline.
    Zum Öffnen und Bedienen ist damit keine Internetverbindung erforderlich.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = study["variants"]
    comparison_fields = [
        "name", "max_depth", "min_samples_leaf", "recommended",
        *variants[0]["summary"].keys(),
    ]
    with (output_dir / "Vergleich.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields, delimiter=";")
        writer.writeheader()
        for variant in variants:
            writer.writerow({
                "name": variant["name"],
                "max_depth": variant["max_depth"] if variant["max_depth"] is not None else "None",
                "min_samples_leaf": variant["min_samples_leaf"],
                "recommended": variant["name"] == study["selection"]["recommended_variant"],
                **variant["summary"],
            })
    with (output_dir / "lernkurve.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "fraction", "fold", "n_train", "n_validation", "train_accuracy",
                "validation_accuracy", "train_macro_f1", "validation_macro_f1",
            ],
            delimiter=";",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(study["learning_curve"])

    paragraphs = _narrative(study)
    comparison_headers = [
        "Max. Tiefe", "Min. je Blatt", "Training Accuracy", "CV Accuracy",
        "Training Macro-F1", "CV Macro-F1", "Knoten je Baum", "Auswahl",
    ]
    comparison_rows = _comparison_rows(study)
    summaries = _learning_summary(study["learning_curve"])
    learning_headers = [
        "Training je Fold (Ø)", "Training Accuracy", "CV Accuracy",
        "Training Macro-F1", "CV Macro-F1",
    ]
    learning_rows = [
        [
            _number(row["n_train_mean"]),
            *[_score(row, metric) for metric in (
                "train_accuracy", "validation_accuracy", "train_macro_f1",
                "validation_macro_f1",
            )],
        ]
        for row in summaries
    ]
    first, last = summaries[0], summaries[-1]
    learning_interpretation = (
        f"Bei durchschnittlich {_number(first['n_train_mean'])} Trainingsbeobachtungen "
        f"je Fold beträgt der Validierungs-Macro-F1 "
        f"{_percent(first['validation_macro_f1_mean'])}; bei "
        f"{_number(last['n_train_mean'])} Beobachtungen sind es "
        f"{_percent(last['validation_macro_f1_mean'])}. "
        f"Die Differenz zwischen Training und Validierung verändert sich dabei von "
        f"{_number(100 * (first['train_macro_f1_mean'] - first['validation_macro_f1_mean']), 2)} "
        f"auf {_number(100 * (last['train_macro_f1_mean'] - last['validation_macro_f1_mean']), 2)} "
        "Prozentpunkte. Diese Werte beschreiben nur den untersuchten Datenbereich; "
        "ein weiterer Leistungsgewinn durch neue Daten ist damit nicht garantiert."
    )
    source_text = (
        "Methodischer Bezug: IU-Skript Model Engineering, Abschnitt 6.1, "
        "gedruckte Seiten 96–97 (Überanpassung und Regularisierung). "
        "Die konkrete Parameterwahl ist eine eigene Prüfung für diesen Datensatz."
    )
    sources = [
        ("scikit-learn: RandomForestClassifier", "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"),
        ("scikit-learn: Lernkurven", "https://scikit-learn.org/stable/modules/learning_curve.html"),
        ("scikit-learn: Kreuzvalidierung", "https://scikit-learn.org/stable/modules/cross_validation.html"),
        ("scikit-learn: Verzerrung bei der Modellauswahl", "https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html"),
    ]
    metadata = study["metadata"]

    def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
        return "\n".join([
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ])

    markdown = "\n\n".join([
        "# Random-Forest-Modellprüfung",
        *paragraphs[:3],
        "## Vergleich der Varianten",
        "Mittelwert ± Standardabweichung über die Kreuzvalidierungsfolds. PP = Prozentpunkte.",
        markdown_table(comparison_headers, comparison_rows),
        "## Lernkurve des bisherigen Modells",
        "Die interaktive Grafik befindet sich in [bericht.html](bericht.html). "
        "Die Trainingsgröße ist der Mittelwert der tatsächlich verwendeten Fold-Größen. "
        "Die Y-Achse zeigt den Bereich 90–102 %, damit kleine Unterschiede sichtbar werden.",
        markdown_table(learning_headers, learning_rows),
        learning_interpretation,
        "## Vorgehen und Aussagekraft",
        *paragraphs[3:],
        "## Nachvollziehbarkeit",
        f"Erzeugt (UTC): {metadata['created_at_utc']}. Python {metadata['python_version']}; "
        f"scikit-learn {metadata['sklearn_version']}.",
        "Dateiprüfsummen (SHA-256):\n\n" + "\n".join(
            f"- {key}: `{metadata[key]}`"
            for key in ("data_sha256", "script_sha256", "modeling_sha256")
        ),
        "Rohwerte: [Vergleich.csv](Vergleich.csv) und [lernkurve.csv](lernkurve.csv). "
        "CSV-Metriken sind Anteile zwischen 0 und 1; als Trennzeichen dient das Semikolon.",
        "## Methodische Quellen",
        source_text,
        "\n".join(f"- [{label}]({url})" for label, url in sources),
    ]) + "\n"
    (output_dir / "bericht.md").write_text(markdown, encoding="utf-8")

    def html_table(headers: list[str], rows: list[list[str]]) -> str:
        heading = "".join(f"<th scope='col'>{escape(value)}</th>" for value in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
            for row in rows
        )
        return f"<div class='table-wrap'><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>"

    chart_html = _learning_figure(summaries).to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id="lernkurve",
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {"format": "svg", "filename": "lernkurve", "width": 1100, "height": 420},
        },
    )
    intro = "".join(f"<p>{escape(text)}</p>" for text in paragraphs[:3])
    method = "".join(f"<p>{escape(text)}</p>" for text in paragraphs[3:])
    sources_html = "".join(
        f"<li><a href='{escape(url, quote=True)}'>{escape(label)}</a></li>"
        for label, url in sources
    )
    checksums = "".join(
        f"<dt>{key}</dt><dd><code>{escape(str(metadata[key]))}</code></dd>"
        for key in ("data_sha256", "script_sha256", "modeling_sha256")
    )
    html = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Random-Forest-Modellprüfung</title>
<style>
body {{ margin: 0; color: #25313a; background: white; font: 16px/1.55 Arial, sans-serif; }}
main {{ max-width: 1150px; margin: 36px auto; padding: 0 24px 32px; }}
h1 {{ font-size: 29px; line-height: 1.25; margin-bottom: 12px; }}
h2 {{ font-size: 21px; margin-top: 30px; padding-top: 12px; border-top: 1px solid #ccd3d8; }}
p {{ max-width: 100ch; }} a {{ color: #285473; }}
.note {{ font-size: 14px; color: #505d66; }}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; line-height: 1.4; }}
th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid #d8dfe3; }}
th {{ background: #edf1f4; vertical-align: bottom; }}
td {{ white-space: nowrap; }}
code {{ overflow-wrap: anywhere; font-size: 12px; }}
dt {{ font-weight: bold; font-size: 13px; }} dd {{ margin: 0 0 8px; }}
@media(max-width:600px) {{ main {{ margin-top: 20px; padding: 0 14px; }} h1 {{ font-size: 25px; }} }}
@media print {{ main {{ margin: 0; padding: 0; }} .table-wrap {{ overflow: visible; }} table {{ font-size: 10px; }} }}
</style></head><body><main>
<h1>Random-Forest-Modellprüfung</h1>
<p class="note">Separater Vergleich · kein Wechsel des aktiven App-Modells</p>
{intro}
<h2>Vergleich der Varianten</h2>
<p class="note">Mittelwert ± Standardabweichung über die Kreuzvalidierungsfolds. PP = Prozentpunkte.</p>
{html_table(comparison_headers, comparison_rows)}
<h2>Lernkurve des bisherigen Modells</h2>
<p class="note">Punkte: Mittelwerte. Fehlerbalken: ± eine Standardabweichung über die Folds, keine Konfidenzintervalle. Die Trainingsgröße ist der Mittelwert der tatsächlich verwendeten Fold-Größen. Die Y-Achse zeigt den Bereich 90–102 %, damit kleine Unterschiede sichtbar werden.</p>
{chart_html}
<p>{escape(learning_interpretation)}</p>
<details><summary>Werte der Lernkurve</summary>{html_table(learning_headers, learning_rows)}</details>
<h2>Vorgehen und Aussagekraft</h2>{method}
<h2>Nachvollziehbarkeit</h2>
<p class="note">Erzeugt (UTC): {escape(str(metadata['created_at_utc']))}. Python {escape(str(metadata['python_version']))}; scikit-learn {escape(str(metadata['sklearn_version']))}.</p>
<p><a href="Vergleich.csv">Varianten als CSV</a> · <a href="lernkurve.csv">Lernkurven-Folds als CSV</a> · <a href="bericht.md">Bericht als Markdown</a></p>
<p class="note">CSV-Metriken sind Anteile zwischen 0 und 1; als Trennzeichen dient das Semikolon. Dieser Bericht enthält alle Grafikbibliotheken und funktioniert ohne Internetverbindung. Externe Quellenlinks benötigen Internet.</p>
<details><summary>Dateiprüfsummen (SHA-256)</summary><dl>{checksums}</dl></details>
<h2>Methodische Quellen</h2><p>{escape(source_text)}</p><ul>{sources_html}</ul>
</main></body></html>"""
    (output_dir / "bericht.html").write_text(html, encoding="utf-8")
