"""Manuelles Re-Training im lokalen Einzelbetrieb, getrennt von der Übernahme."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

import pandas as pd
import joblib

from src.data_processing import DATA_PATH
from src import model_registry as registry
from src.retraining_data import prepare_dataset, train_and_evaluate
from src.storage import OBSERVATIONS_PATH


class RetrainingService:
    """Ein Hintergrundauftrag; das aktive Modell bleibt währenddessen nutzbar.

    Trainiert wird immer mit einem vollständigen Daten-Snapshot. Erst eine
    ausdrückliche Übernahme ändert den Modellzeiger. Eine noch nicht übernommene
    Variante bleibt auch nach einem Neustart zur Prüfung verfügbar.
    """

    def __init__(self, models_dir=None, reference_path=None, observations_path=None):
        self.models_dir = Path(models_dir or registry.MODELS_DIR)
        self.reference_path = Path(reference_path or DATA_PATH)
        self.observations_path = Path(observations_path or OBSERVATIONS_PATH)
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="penguin-training")
        self._future = None
        self._state = {
            "phase": "idle", "message": "Bereit für bestätigte Beobachtungen.",
            "candidate_id": None, "comparison": None,
        }
        self._restore_pending()

    @property
    def pending_path(self):
        return self.models_dir / "pending_candidate.json"

    def _restore_pending(self):
        if not self.pending_path.exists():
            return
        try:
            candidate = json.loads(self.pending_path.read_text(encoding="utf-8"))["candidate_id"]
            manifest = registry.get_manifest(candidate, models_dir=self.models_dir)
            current = registry.get_registry_state(models_dir=self.models_dir)["active_version"]
            if manifest["base_version"] == current and candidate != current:
                metrics_path = registry.version_paths(candidate, models_dir=self.models_dir)["metrics"]
                self._state.update(
                    phase="ready", message="Neue Variante bereit. Bitte den Vergleich prüfen.",
                    candidate_id=candidate,
                    comparison=json.loads(metrics_path.read_text(encoding="utf-8")),
                )
        except (OSError, ValueError, KeyError, TypeError):
            self._state.update(phase="error", message="Die gespeicherte Trainingsvariante ist nicht lesbar. Bitte erneut trainieren.")

    def _remember_pending(self, candidate):
        temporary = self.pending_path.with_name(f".pending-{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps({"candidate_id": candidate}), encoding="utf-8")
            os.replace(temporary, self.pending_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _progress(self, message):
        with self._lock:
            self._state["message"] = message

    def get_status(self):
        """Liefere Zähler, Datenprüfung und Jobstatus, ohne ein Training auszulösen."""
        with self._lock:
            state = dict(self._state)
        try:
            current = registry.get_registry_state(models_dir=self.models_dir)
            state.update(current)
            _, report = prepare_dataset(self.reference_path, self.observations_path)
            active = registry.get_manifest(current["active_version"], models_dir=self.models_dir)
            previous_ids = set(active.get("confirmed_ids", []))
            current_ids = set(report["confirmed_ids"])
            reference_changed = (active.get("reference_hash") is not None
                                 and active["reference_hash"] != report["reference_hash"])
            if reference_changed:
                raise ValueError("Die Referenzdaten wurden seit der Modellübernahme verändert. Bitte die ursprüngliche Referenzdatei wiederherstellen.")
            state.update(
                data_report=report, new_confirmed_count=len(current_ids - previous_ids),
                can_train=current_ids != previous_ids and state["phase"] != "training",
                data_error=None,
            )
            # Ein zweiter Browser kann die aktive Version bereits geändert haben.
            if state["phase"] == "ready":
                manifest = registry.get_manifest(state["candidate_id"], models_dir=self.models_dir)
                if manifest["base_version"] != current["active_version"]:
                    state.update(phase="idle", candidate_id=None, comparison=None,
                                 message="Die aktive Version hat sich geändert. Bitte erneut trainieren.")
                elif not set(manifest["confirmed_ids"]).issubset(current_ids):
                    state.update(phase="idle", candidate_id=None, comparison=None,
                                 message="Bestätigungen wurden seit dem Training geändert oder entfernt. Bitte erneut trainieren.")
                elif manifest["reference_hash"] != report["reference_hash"]:
                    state.update(phase="idle", candidate_id=None, comparison=None,
                                 message="Die Referenzdaten wurden seit dem Training geändert. Bitte erneut trainieren.")
        except (OSError, ValueError, KeyError, TypeError) as error:
            state.update(data_error=str(error), can_train=False, new_confirmed_count=0,
                         data_report=None)
        state["can_adopt"] = state["phase"] == "ready" and not state.get("data_error")
        state["can_rollback"] = bool(state.get("previous_version")) and state["phase"] != "training"
        return state

    def start_training(self):
        with self._lock:
            if self._state["phase"] == "training":
                raise ValueError("Es läuft bereits ein Re-Training.")
            status = self.get_status()
            if status.get("data_error"):
                raise ValueError(status["data_error"])
            if not status["can_train"]:
                raise ValueError("Keine neuen oder geänderten bestätigten Beobachtungen vorhanden.")
            self.pending_path.unlink(missing_ok=True)
            self._state.update(phase="training", message="Bestätigte Beobachtungen werden geprüft …",
                               candidate_id=None, comparison=None)
            try:
                self._future = self._executor.submit(self._train)
            except RuntimeError as error:
                self._state.update(phase="error", message="Der Trainingsauftrag konnte nicht gestartet werden. Bitte die Anwendung neu starten.")
                raise ValueError(self._state["message"]) from error

    def _train(self):
        try:
            data, report = prepare_dataset(self.reference_path, self.observations_path)
            current = registry.get_registry_state(models_dir=self.models_dir)["active_version"]
            active_manifest = registry.get_manifest(current, models_dir=self.models_dir)
            if (active_manifest.get("reference_hash") is not None
                    and active_manifest["reference_hash"] != report["reference_hash"]):
                raise ValueError("Die Referenzdaten passen nicht zur aktiven Modellversion.")
            if set(report["confirmed_ids"]) == set(active_manifest.get("confirmed_ids", [])):
                raise ValueError("Keine neuen oder geänderten bestätigten Beobachtungen vorhanden.")
            baseline_path = registry.version_paths(current, models_dir=self.models_dir)["training_data"]
            if current == "initial":
                baseline = data.loc[data["source"] == "reference"].copy()
            else:
                baseline = pd.read_csv(baseline_path, keep_default_na=False, float_precision="round_trip")
            baseline_pipeline = joblib.load(
                registry.version_paths(current, models_dir=self.models_dir)["model"]
            )
            result = train_and_evaluate(
                data, baseline, progress=self._progress, baseline_pipeline=baseline_pipeline,
            )
            result["metrics"]["data_report"] = report
            manifest = {
                "base_version": current, "reference_hash": report["reference_hash"],
                "confirmed_ids": report["confirmed_ids"],
            }
            self._progress("Modellvariante und Trainingsdaten werden gesichert …")
            candidate = registry.save_candidate(
                result["model"], result["metadata"], result["metrics"], data, manifest,
                models_dir=self.models_dir,
            )
            self._remember_pending(candidate)
            with self._lock:
                self._state.update(phase="ready", message="Training abgeschlossen. Bitte den Modellvergleich prüfen.",
                                   candidate_id=candidate, comparison=result["metrics"])
        except Exception as error:
            # Auch ein Fehler in scikit-learn darf den bisherigen Betrieb nicht beenden.
            with self._lock:
                self._state.update(phase="error", message=f"Re-Training fehlgeschlagen: {error}",
                                   candidate_id=None, comparison=None)

    def adopt_candidate(self):
        with self._lock:
            status = self.get_status()
            if not status["can_adopt"]:
                raise ValueError("Es steht keine geprüfte neue Variante zur Übernahme bereit.")
            state = registry.activate_candidate(self._state["candidate_id"], models_dir=self.models_dir)
            self._state.update(phase="adopted", message="Neue Modellversion übernommen. Bitte eine neue Vorhersage erstellen.",
                               candidate_id=None, comparison=None)
            self._forget_pending()
            return state

    def restore_previous(self):
        with self._lock:
            if self._state["phase"] == "training":
                raise ValueError("Bitte das laufende Re-Training zuerst abwarten.")
            state = registry.rollback(models_dir=self.models_dir)
            self._state.update(phase="restored", message="Vorherige Modellversion wiederhergestellt. Bitte erneut klassifizieren.",
                               candidate_id=None, comparison=None)
            self._forget_pending()
            return state

    def _forget_pending(self):
        try:
            self.pending_path.unlink(missing_ok=True)
        except OSError:
            # Der bereits atomar aktivierte Modellzeiger ist weiterhin maßgeblich.
            pass

    def wait(self, timeout=60):
        """Auf einen gestarteten Auftrag warten; für CLI und isolierte Integrationstests."""
        future = self._future
        if future is not None:
            future.result(timeout=timeout)

    def close(self):
        self._executor.shutdown(wait=True)
