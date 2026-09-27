"""Windows-Docker-Prüfstart mit simuliertem Docker, ohne echte Container."""

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != "nt", reason="Die Batchdatei benötigt Windows.")


@pytest.fixture
def container_launcher(tmp_path_factory):
    # Batchdateien brauchen bei tiefen OneDrive-Pfaden kurze Testverzeichnisse.
    sandbox = tmp_path_factory.mktemp("c")
    project = sandbox / "Mit Platz"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(ROOT / "docker-pruefen.bat", project / "docker-pruefen.bat")
    shutil.copyfile(ROOT / "compose.yaml", project / "compose.yaml")
    (scripts / "container_check.py").write_text("# PROBE_SENTINEL\n", encoding="ascii")
    binary = sandbox / "bin"
    binary.mkdir()
    command_log = sandbox / "calls.log"
    state = sandbox / "state.txt"
    (binary / "docker.cmd").write_text(
        '@echo off\n'
        'setlocal EnableExtensions EnableDelayedExpansion\n'
        'echo ENV %PENGUIN_CONTAINER_TEST_ID%^|%COMPOSE_PROJECT_NAME%^|%PENGUIN_DATA_VOLUME%^|%PENGUIN_MODELS_VOLUME%^|%PENGUIN_PORT%>>"%MOCK_LOG%"\n'
        'echo CMD %*>>"%MOCK_LOG%"\n'
        'if "%~1"=="info" if "%MOCK_CASE%"=="engine-down" exit /b 1\n'
        'if "%~1"=="image" if "%MOCK_CASE%"=="missing-image" exit /b 1\n'
        'if "%~1"=="volume" goto volume\n'
        'if "%~8"=="version" if "%MOCK_CASE%"=="missing-compose" exit /b 1\n'
        'if "%~8"=="up" goto up\n'
        'if "%~8"=="exec" goto exec\n'
        'if "%~8"=="down" if "%MOCK_CASE%"=="down-failed" exit /b 1\n'
        'exit /b 0\n'
        ':volume\n'
        'set "NAME=%~3"\n'
        'if "!NAME:~-5!"=="-data" if "%MOCK_CASE%"=="data-collision" exit /b 0\n'
        'if "!NAME:~-7!"=="-models" if "%MOCK_CASE%"=="models-collision" exit /b 0\n'
        'exit /b 1\n'
        ':up\n'
        'set "COUNT=0"\n'
        'if exist "%MOCK_STATE%" set /p COUNT=<"%MOCK_STATE%"\n'
        'set /a COUNT+=1\n'
        '>"%MOCK_STATE%" echo !COUNT!\n'
        'if "%MOCK_CASE%"=="start-failed" exit /b 1\n'
        'if "!COUNT!"=="2" if "%MOCK_CASE%"=="restart-failed" exit /b 1\n'
        'exit /b 0\n'
        ':exec\n'
        'findstr /c:"PROBE_SENTINEL" >nul\n'
        'if errorlevel 1 exit /b 9\n'
        'echo Probe-Skript ueber stdin erhalten.\n'
        ':exec_args\n'
        'if "%~1"=="" exit /b 0\n'
        'if "%~1"=="prepare" if "%MOCK_CASE%"=="prepare-failed" exit /b 1\n'
        'if "%~1"=="verify" if "%MOCK_CASE%"=="verify-failed" exit /b 1\n'
        'shift\n'
        'goto exec_args\n',
        encoding="ascii",
    )

    def run(scenario="ok", missing_file=None):
        if command_log.exists():
            command_log.unlink()
        if state.exists():
            state.unlink()
        if missing_file:
            (project / missing_file).unlink()
        env = os.environ.copy()
        env.update(
            PATH=str(binary) + os.pathsep + env.get("PATH", ""),
            MOCK_LOG=str(command_log), MOCK_STATE=str(state), MOCK_CASE=scenario,
            # Eine produktive Umgebung darf den Test nicht umlenken.
            COMPOSE_PROJECT_NAME="penguin-classifier",
            PENGUIN_DATA_VOLUME="penguin-data", PENGUIN_MODELS_VOLUME="penguin-models",
            PENGUIN_PORT="8050", PENGUIN_CONTAINER_TEST_ID="production",
        )
        if scenario == "guid-failed":
            # cmd.exe und findstr bleiben erreichbar, Windows PowerShell nicht.
            env["PATH"] = str(binary) + os.pathsep + str(Path(os.environ["SystemRoot"]) / "System32")
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "call", str(project / "docker-pruefen.bat"), "--no-pause"],
            env=env, cwd=sandbox, capture_output=True, text=True, errors="replace", timeout=20,
        )
        entries = command_log.read_text(errors="replace").splitlines() if command_log.exists() else []
        calls = [entry[4:] for entry in entries if entry.startswith("CMD ")]
        environments = [entry[4:].split("|") for entry in entries if entry.startswith("ENV ")]
        logs = list((project / "tmp").glob("*.log"))
        if scenario == "guid-failed":
            assert not logs
            return result, calls, environments, ""
        assert logs, result.stdout + result.stderr
        latest = max(logs, key=lambda path: path.stat().st_mtime_ns)
        log_text = latest.read_text(errors="replace")
        assert str(latest) in result.stdout
        return result, calls, environments, log_text

    return run


def assert_isolated(calls, environments):
    assert environments
    test_id, project, data, models, port = environments[0]
    assert re.fullmatch(r"penguin-pruefung-[0-9a-f]{32}", test_id)
    assert project == test_id
    assert data == test_id + "-data"
    assert models == test_id + "-models"
    assert port == "8057"
    assert all(environment == environments[0] for environment in environments)
    for call in calls:
        assert not any(word in call.split() for word in ("build", "pull", "prune", "rm", "-v", "--volumes"))
        if call.startswith("compose "):
            assert f'compose -p "{test_id}" --project-directory "' in call
            assert 'Mit Platz" -f "' in call
            assert 'compose.yaml" ' in call


def test_container_check_recreates_container_and_checks_persistent_data(container_launcher):
    result, calls, environments, log = container_launcher()
    assert result.returncode == 0, result.stdout + result.stderr + log
    assert "BESTANDEN" in result.stdout
    assert_isolated(calls, environments)
    operations = [
        "prepare" if " python - prepare" in call else
        "verify" if " python - verify" in call else
        "up" if " up " in call else "down"
        for call in calls if " up " in call or call.endswith(" down") or " exec " in call
    ]
    assert operations == ["up", "prepare", "down", "up", "verify", "down"]
    starts = [call for call in calls if " up " in call]
    assert all(call.endswith("up -d --no-build --pull never --wait --wait-timeout 60 app") for call in starts)
    probes = [call for call in calls if " exec " in call]
    assert len(probes) == 2
    assert all(f"exec -T -e PENGUIN_CONTAINER_TEST_ID={environments[0][0]} app python - " in call for call in probes)
    assert log.count("Probe-Skript ueber stdin erhalten.") == 2


@pytest.mark.parametrize("scenario", [
    "engine-down", "missing-compose", "missing-image", "data-collision", "models-collision",
])
def test_prerequisite_or_volume_collision_never_changes_a_container(container_launcher, scenario):
    result, calls, environments, log = container_launcher(scenario)
    assert result.returncode != 0
    assert "FEHLGESCHLAGEN" in result.stdout
    assert "FEHLER" in log
    assert_isolated(calls, environments)
    assert not any(" up " in call or call.endswith(" down") or " exec " in call for call in calls)


@pytest.mark.parametrize("scenario", [
    "start-failed", "prepare-failed", "restart-failed", "verify-failed", "down-failed",
])
def test_failed_test_logs_diagnostics_and_cleans_up_only_its_container(container_launcher, scenario):
    result, calls, environments, log = container_launcher(scenario)
    assert result.returncode != 0
    assert "FEHLGESCHLAGEN" in result.stdout
    assert "BESTANDEN" not in result.stdout
    assert "Container-Protokolle" in log
    assert_isolated(calls, environments)
    assert calls[-2].endswith("logs --no-color --tail 200 app")
    assert calls[-1].endswith(" down")
    if scenario in ("start-failed", "prepare-failed", "restart-failed"):
        assert not any(" python - verify" in call for call in calls)


@pytest.mark.parametrize("missing_file", ["compose.yaml", "scripts/container_check.py"])
def test_missing_project_files_fail_without_docker(container_launcher, missing_file):
    result, calls, environments, log = container_launcher(missing_file=missing_file)
    assert result.returncode != 0
    assert "FEHLGESCHLAGEN" in result.stdout
    assert "fehlt" in log
    assert not calls
    assert not environments


def test_each_run_uses_a_new_project_and_new_volumes(container_launcher):
    first = container_launcher()
    second = container_launcher()
    assert first[0].returncode == second[0].returncode == 0
    assert first[2][0][0] != second[2][0][0]


def test_failed_guid_generation_stops_before_docker(container_launcher):
    result, calls, environments, _ = container_launcher(scenario="guid-failed")
    assert result.returncode != 0
    assert "FEHLGESCHLAGEN" in result.stdout
    assert "Test-ID konnte nicht erstellt werden" in result.stdout
    assert not calls
    assert not environments
