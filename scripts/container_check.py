"""Funktionstest in frischen Docker-Testvolumes, in zwei getrennten Prozessen.

Aufruf durch docker-pruefen.bat via stdin: python - prepare / python - verify.
Die synthetische Artbestätigung dient ausschließlich diesem Softwaretest.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.request import urlopen

from src import model_registry as registry
from src import model_service
from src.retraining import RetrainingService
from src.storage import save_observation


STATE_FILE = ".docker-test-state.json"
OBSERVATION = {
    "bill_length_mm": 49.12345678901235,
    "bill_depth_mm": 18.12345678901235,
    "flipper_length_mm": 201,
    "body_mass_g": 4123,
    "sex": "unknown",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_context(root, test_id):
    require(re.fullmatch(r"penguin-pruefung-[a-z0-9]+(?:-[a-z0-9]+)*", test_id),
            "Ungueltige Kennung: nur getrennte Testinstanzen verwenden.")
    require(model_service.MODELS_DIR.resolve() == (root / "models").resolve(),
            "Modellpfad passt nicht zum Testverzeichnis.")
    require((root / "data" / "penguins.csv").is_file(), "Referenzdaten fehlen.")


def file_hashes(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in (root / "data", root / "models")
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != root / "data" / STATE_FILE
    }


def predict():
    result = model_service.predict_species(OBSERVATION)
    probabilities = result["probabilities"]
    require(set(probabilities) == {"Adelie", "Chinstrap", "Gentoo"},
            "Die Vorhersage enthaelt nicht alle drei Arten.")
    require(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
            and math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-10),
            "Ungueltige Modellwahrscheinlichkeiten.")
    require(result["predicted_species"] in probabilities, "Ungueltige vorhergesagte Art.")
    return result


def prepare(root, test_id):
    check_context(root, test_id)
    data = root / "data"
    models = root / "models"
    require({p.name for p in data.iterdir()} == {"penguins.csv"},
            "Testdatenverzeichnis ist nicht leer. Vorhandene Daten bleiben unveraendert.")
    require({p.name for p in models.iterdir()} == set(registry.ARTIFACT_NAMES[k]
            for k in ("model", "metadata", "metrics")),
            "Modellverzeichnis enthaelt bereits Zusatzdateien. Neue Testvolumes erforderlich.")
    initial_files = file_hashes(root)
    initial_prediction = predict()
    require(initial_prediction["model_version"] == "initial", "Erwartet wurde das Basismodell.")
    print("OK: Basismodell geladen und Vorhersage berechnet.", flush=True)

    observations = data / "new_observations.csv"
    save_observation(
        initial_prediction["validated_observation"], "Dream",
        initial_prediction["predicted_species"], initial_prediction["probabilities"],
        output_path=observations, validated_species="Chinstrap",
    )
    with observations.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1 and rows[0]["validated_species"] == "Chinstrap"
            and rows[0]["predicted_species"] == initial_prediction["predicted_species"],
            "Die CSV-Beobachtung wurde nicht korrekt gespeichert.")
    print("OK: Synthetische Beobachtung und separates Testlabel gespeichert.", flush=True)

    service = RetrainingService(models, data / "penguins.csv", observations)
    try:
        require(service.get_status()["new_confirmed_count"] == 1,
                "Die neue Testbeobachtung wurde nicht erkannt.")
        print("Re-Training mit der normalen Modellkonfiguration laeuft ...", flush=True)
        service.start_training()
        service.wait(timeout=300)
        status = service.get_status()
        require(status["phase"] == "ready" and status["can_adopt"]
                and bool(status["comparison"]), status["message"])
        require(registry.get_registry_state(models)["active_version"] == "initial",
                "Das Training hat das aktive Modell ohne Uebernahme ersetzt.")
        candidate = status["candidate_id"]
        state = service.adopt_candidate()
        require(state == {"active_version": candidate, "previous_version": "initial"},
                "Die Modelluebernahme hat nicht die erwarteten Verweise gespeichert.")
        candidate_prediction = predict()
        require(candidate_prediction["model_version"] == candidate,
                "Der Modellservice verwendet nach Uebernahme noch die alte Version.")
    finally:
        service.close()

    hashes = file_hashes(root)
    require(all(hashes.get(path) == value for path, value in initial_files.items()),
            "Referenzdaten oder urspruengliche Modellartefakte wurden veraendert.")
    snapshot = {
        "test_id": test_id,
        "candidate_id": candidate,
        "initial_prediction": initial_prediction,
        "candidate_prediction": candidate_prediction,
        "hashes": hashes,
    }
    (data / STATE_FILE).write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print("OK: Kandidat bewertet, separat uebernommen und Dateipruefsummen gespeichert.", flush=True)


def verify(root, test_id):
    check_context(root, test_id)
    data = root / "data"
    models = root / "models"
    snapshot = json.loads((data / STATE_FILE).read_text(encoding="utf-8"))
    require(snapshot["test_id"] == test_id, "Die Testdaten gehoeren zu einem anderen Lauf.")
    require(file_hashes(root) == snapshot["hashes"],
            "Dateien fehlen oder wurden nach dem Entfernen des Containers veraendert.")
    print("OK: CSV, Referenzdaten und alle Modellartefakte nach Neuerstellung unveraendert.", flush=True)
    require(predict() == snapshot["candidate_prediction"],
            "Modellversion oder Vorhersage unterscheiden sich nach dem Neustart.")

    service = RetrainingService(models, data / "penguins.csv", data / "new_observations.csv")
    try:
        require(service.get_status()["can_rollback"], "Vorherige Modellversion fehlt.")
        state = service.restore_previous()
        require(state == {"active_version": "initial", "previous_version": snapshot["candidate_id"]},
                "Die vorherige Version wurde nicht wiederhergestellt.")
        require(predict() == snapshot["initial_prediction"],
                "Die wiederhergestellte Version liefert eine abweichende Vorhersage.")
    finally:
        service.close()
    print("OK: Vorherige Version wiederhergestellt und Vorhersage geprueft.", flush=True)


def check_http():
    for endpoint in ("/", "/_dash-layout", "/_dash-dependencies"):
        with urlopen("http://127.0.0.1:8050" + endpoint, timeout=10) as response:
            body = response.read()
            require(response.status == 200 and bool(body), "Dash ist nicht erreichbar.")
            if endpoint != "/":
                json.loads(body)
    print("OK: Dash-Seite, Layout und Callback-Definitionen erreichbar.", flush=True)


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ("prepare", "verify"),
            "Aufruf ausschliesslich ueber docker-pruefen.bat.")
    root = Path("/app")
    require(Path("/.dockerenv").is_file() and Path.cwd().resolve() == root
            and os.path.ismount(root / "data") and os.path.ismount(root / "models"),
            "Dieser Test darf nur im vorbereiteten Docker-Testcontainer laufen.")
    test_id = os.environ.get("PENGUIN_CONTAINER_TEST_ID", "")
    check_context(root, test_id)
    check_http()
    if sys.argv[1] == "prepare":
        prepare(root, test_id)
    else:
        verify(root, test_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FEHLER: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
