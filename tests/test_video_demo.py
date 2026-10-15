"""Die Video-Vorbereitung darf keine bestehenden Beobachtungen/Versionen übernehmen."""

import json
from pathlib import Path
import socket

import pytest

from scripts import video_demo


@pytest.fixture
def project(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "models" / "versions" / "existing").mkdir(parents=True)
    (tmp_path / "data" / "penguins.csv").write_text("reference", encoding="utf-8")
    (tmp_path / "data" / "new_observations.csv").write_text("real observations", encoding="utf-8")
    for name in video_demo.MODEL_FILES:
        (tmp_path / "models" / name).write_bytes(name.encode())
    (tmp_path / "models" / "active_model.json").write_text("existing pointer", encoding="utf-8")
    (tmp_path / "models" / "pending_candidate.json").write_text("pending", encoding="utf-8")
    (tmp_path / "models" / "versions" / "existing" / "real.txt").write_text("real", encoding="utf-8")
    return tmp_path


def test_each_demo_starts_fresh_and_preserves_every_existing_file(project):
    before = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
    first = video_demo.prepare_demo(project)
    (first / "data" / "new_observations.csv").write_text("previous demo", encoding="utf-8")
    second = video_demo.prepare_demo(project)
    assert first != second
    assert second.parent == project / "tmp" / "video-demo"
    assert set(path.name for path in (second / "models").iterdir()) == set(video_demo.MODEL_FILES)
    assert not list((second / "data").iterdir())
    assert all(path.read_bytes() == value for path, value in before.items())
    assert (first / "data" / "new_observations.csv").read_text() == "previous demo"
    info = json.loads((second / "demo-info.json").read_text(encoding="utf-8"))
    assert info["source_sha256"] == video_demo.source_hashes(project)
    assert Path(info["models"]) == second / "models"


def test_missing_base_artifact_leaves_project_unchanged(project):
    (project / "models" / video_demo.MODEL_FILES[0]).unlink()
    with pytest.raises(FileNotFoundError):
        video_demo.prepare_demo(project)
    assert not (project / "tmp").exists()


def test_occupied_port_is_reported_without_closing_other_server(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        monkeypatch.setattr(video_demo, "PORT", port)
        with pytest.raises(RuntimeError, match=f"Port {port}.*belegt"):
            video_demo.ensure_port_available()
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            connection, _ = listener.accept()
            connection.close()
