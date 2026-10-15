"""Frische Video-Demo mit eigener CSV und eigenen Modellversionen.

Die Anwendung liest die unveränderten Referenzdaten aus dem Projekt. Sämtliche
neuen Beobachtungen und Modellversionen landen in einem neuen Demo-Verzeichnis.
Aufruf: python scripts/video_demo.py run [--open-browser] | check
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from threading import Thread
import time
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ("penguin_pipeline.joblib", "model_metadata.json", "metrics.json")
PORT = 8058
URL = f"http://127.0.0.1:{PORT}"
DEMO_OBSERVATION = {
    "bill_length_mm": 46.1,
    "bill_depth_mm": 17.3,
    "flipper_length_mm": 197,
    "body_mass_g": 4050,
    "sex": "male",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def source_hashes(root=PROJECT_ROOT):
    paths = [root / "data" / "penguins.csv"]
    paths.extend(root / "models" / name for name in MODEL_FILES)
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def prepare_demo(root=PROJECT_ROOT):
    """Kopiere nur die Basismodellartefakte; bestehende Läufe bleiben erhalten."""
    root = Path(root).resolve()
    hashes = source_hashes(root)  # Fehlende Quelldateien vor dem Anlegen erkennen.
    session = root / "tmp" / "video-demo" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:12]
    )
    models = session / "models"
    models.mkdir(parents=True, exist_ok=False)
    (session / "data").mkdir()
    for name in MODEL_FILES:
        shutil.copyfile(root / "models" / name, models / name)
    info = {
        "purpose": "Isolierte Video-Demo; synthetische Labels sind keine Felderhebung.",
        "reference_data": str(root / "data" / "penguins.csv"),
        "observations": str(session / "data" / "new_observations.csv"),
        "models": str(models),
        "source_sha256": hashes,
    }
    (session / "demo-info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return session


def ensure_port_available():
    """Keine laufende Instanz übernehmen oder beenden."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            listener.bind(("127.0.0.1", PORT))
        except OSError as error:
            raise RuntimeError(
                f"Port {PORT} ist belegt oder nicht verfügbar. Eine noch laufende "
                "Video-Demo zuerst in ihrem Konsolenfenster mit Strg+C beenden. "
                "Es wurde kein Prozess beendet."
            ) from error


def configure_demo(session):
    """Vor dem ersten App-Import sämtliche schreibbaren App-Pfade umstellen."""
    require(not any(name in sys.modules for name in (
        "app", "src.storage", "src.model_registry", "src.model_service", "src.retraining",
    )), "Die Demo muss in einem frischen Python-Prozess gestartet werden.")
    os.environ["PENGUIN_MODEL_DIR"] = str(session / "models")
    os.environ["PENGUIN_OBSERVATIONS_PATH"] = str(session / "data" / "new_observations.csv")
    sys.path.insert(0, str(PROJECT_ROOT))


def load_demo_app(session):
    configure_demo(session)
    import app as application
    from src import model_registry, model_service, storage

    expected_models = (session / "models").resolve()
    expected_csv = (session / "data" / "new_observations.csv").resolve()
    require(model_registry.MODELS_DIR.resolve() == expected_models
            and model_service.MODELS_DIR.resolve() == expected_models
            and application.RETRAINING.models_dir.resolve() == expected_models
            and storage.OBSERVATIONS_PATH.resolve() == expected_csv
            and application.RETRAINING.observations_path.resolve() == expected_csv,
            "Die Datenpfade sind nicht vollständig auf die Demo umgestellt.")
    require(model_registry.get_registry_state()["active_version"] == "initial",
            "Die Demo muss mit dem Basismodell beginnen.")
    return application


def check_demo(application, session):
    """Vollständiger Software-Probelauf; keinerlei echte Artbestätigung."""
    from src import model_registry, model_service, storage

    client = application.server.test_client()
    for endpoint in ("/", "/_dash-layout", "/_dash-dependencies"):
        response = client.get(endpoint)
        require(response.status_code == 200 and response.data,
                f"Dash-Endpunkt nicht erreichbar: {endpoint}")
    print("OK: Dash-Seite, Layout und Callback-Definitionen geladen.", flush=True)
    service = application.RETRAINING
    initial = model_service.predict_species(DEMO_OBSERVATION)
    storage.save_observation(initial["validated_observation"], "Dream",
                             initial["predicted_species"], initial["probabilities"])
    require(not service.get_status()["can_train"],
            "Eine unbestätigte Beobachtung darf kein Re-Training freigeben.")
    confirmed = {**DEMO_OBSERVATION, "bill_length_mm": 46.2}
    prediction = model_service.predict_species(confirmed)
    storage.save_observation(prediction["validated_observation"], "Dream",
                             prediction["predicted_species"], prediction["probabilities"],
                             validated_species="Adelie")
    require(service.get_status()["new_confirmed_count"] == 1,
            "Genau eine synthetisch bestätigte Demo-Beobachtung erwartet.")
    print("OK: Unbestätigte Zeile ausgeschlossen; separates synthetisches Demo-Label erkannt.",
          flush=True)
    print("Re-Training mit regulärer Konfiguration läuft ...", flush=True)
    started = time.monotonic()
    service.start_training()
    service.wait(timeout=300)
    status = service.get_status()
    require(status["phase"] == "ready" and status["can_adopt"], status["message"])
    require(model_registry.get_registry_state()["active_version"] == "initial",
            "Das aktive Modell darf sich erst bei Übernahme ändern.")
    candidate = status["candidate_id"]
    service.adopt_candidate()
    require(model_service.predict_species(confirmed)["model_version"] == candidate,
            "Die Übernahme wurde nicht im Vorhersagedienst wirksam.")
    service.restore_previous()
    require(model_service.predict_species(DEMO_OBSERVATION) == initial,
            "Die Wiederherstellung liefert nicht die ursprüngliche Vorhersage.")
    info = json.loads((session / "demo-info.json").read_text(encoding="utf-8"))
    require(source_hashes() == info["source_sha256"], "Eine Quelldatei wurde verändert.")
    report = {
        "status": "passed",
        "purpose": "Software-Funktionstest mit synthetischem Label, kein Gütenachweis",
        "training_and_switch_seconds": round(time.monotonic() - started, 2),
        "initial_prediction": initial,
        "candidate_id": candidate,
        "comparison": status["comparison"],
        "final_registry": model_registry.get_registry_state(),
        "source_sha256_unchanged": True,
    }
    output = session / "check-report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK: Training, ausdrückliche Übernahme und Wiederherstellung bestanden.", flush=True)
    print(f"Prüfbericht: {output}", flush=True)


def open_browser_when_ready():
    for _ in range(100):
        try:
            with urlopen(URL, timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(URL)
                    return
        except (OSError, URLError):
            time.sleep(0.2)
    print(f"Browser bitte manuell unter {URL} öffnen.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "check"), nargs="?", default="run")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "run":
        ensure_port_available()
    session = prepare_demo()
    print(f"Video-Demo: {session}", flush=True)
    print("Neue Beobachtungen und Modellversionen werden nur hier gespeichert.", flush=True)
    print("Jeder Start beginnt frisch; ältere Demo-Läufe bleiben erhalten.", flush=True)
    application = load_demo_app(session)
    try:
        if args.mode == "check":
            check_demo(application, session)
        else:
            print(f"Browser: {URL}", flush=True)
            print("Beenden: Strg+C in diesem Fenster. Demo-Labels sind nur Testdaten.", flush=True)
            if args.open_browser:
                Thread(target=open_browser_when_ready, daemon=True).start()
            application.app.run(host="127.0.0.1", port=PORT, debug=False,
                                use_reloader=False, jupyter_mode="external")
    finally:
        application.RETRAINING.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nVideo-Demo beendet. Die Demo-Dateien bleiben erhalten.")
    except Exception as error:
        print(f"FEHLER: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
