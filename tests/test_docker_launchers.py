"""Prüfe Startabläufe mit simulierter Docker-CLI, ohne Container zu verändern."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["windows", "posix"])
def launcher(request, tmp_path_factory):
    # cmd.exe unterstützt für Batchdateien keine beliebig langen Pfade.
    tmp_path = tmp_path_factory.mktemp("l")
    kind = request.param
    if kind == "windows" and os.name != "nt":
        pytest.skip("Windows-Startdateien benötigen cmd.exe.")
    shell = shutil.which("sh")
    if kind == "posix" and not shell:
        git_shell = Path("C:/Program Files/Git/bin/bash.exe")
        if not git_shell.is_file():
            pytest.skip("Für POSIX-Startdateien ist keine Shell verfügbar.")
        shell = str(git_shell)
    project = tmp_path / "Mit Platz"
    (project / "scripts").mkdir(parents=True)
    extension = "bat" if kind == "windows" else "sh"
    for action in ("setup", "start", "stop"):
        shutil.copyfile(ROOT / f"{action}.{extension}", project / f"{action}.{extension}")
    shutil.copyfile(ROOT / "scripts" / f"docker-launcher.{extension}",
                    project / "scripts" / f"docker-launcher.{extension}")
    shutil.copyfile(ROOT / "compose.yaml", project / "compose.yaml")
    binary = tmp_path / "Testprogramme"
    binary.mkdir()
    log = tmp_path / "docker-commands.txt"
    if kind == "windows":
        fake = binary / "docker.cmd"
        fake.write_text(
            '@echo off\n'
            'echo %*>>"%LAUNCHER_TEST_LOG%"\n'
            'if "%~1"=="info" if "%MOCK_CASE%"=="engine-down" exit /b 1\n'
            'if "%~1"=="image" if "%MOCK_CASE%"=="missing-image" exit /b 1\n'
            'if "%~2"=="version" if "%MOCK_CASE%"=="missing-compose" exit /b 1\n'
            'if "%~6"=="build" if "%MOCK_CASE%"=="build-failed" exit /b 1\n'
            'if "%~6"=="up" if "%MOCK_CASE%"=="unhealthy" exit /b 1\n'
            'exit /b 0\n', encoding="ascii",
        )
    else:
        fake = binary / "docker"
        fake.write_text(
            '#!/bin/sh\n'
            'printf "%s\\n" "$*" >> "$LAUNCHER_TEST_LOG"\n'
            'case "$1:$MOCK_CASE" in info:engine-down|image:missing-image) exit 1;; esac\n'
            '[ "$2:$MOCK_CASE" != "version:missing-compose" ] || exit 1\n'
            '[ "$6:$MOCK_CASE" != "build:build-failed" ] || exit 1\n'
            '[ "$6:$MOCK_CASE" != "up:unhealthy" ] || exit 1\n'
            'exit 0\n', encoding="ascii", newline="\n",
        )
        fake.chmod(0o755)

    def run(action="start", scenario="ok", port="8050"):
        env = os.environ.copy()
        env.update(PATH=str(binary) + os.pathsep + env.get("PATH", ""),
                   LAUNCHER_TEST_LOG=str(log), MOCK_CASE=scenario, PENGUIN_PORT=port)
        path = project / f"{action}.{extension}"
        if kind == "windows":
            command = ["cmd.exe", "/d", "/c", "call", str(path), "--no-browser", "--no-pause"]
        else:
            command = [shell, str(path), "--no-browser", "--no-pause"]
        result = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True,
                                text=True, errors="replace", timeout=20)
        calls = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
        return result, calls

    return run


def test_normal_start_is_offline_and_waits_for_a_healthy_app(launcher):
    result, calls = launcher(port="8057")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "http://127.0.0.1:8057" in result.stdout
    assert any("image inspect penguin-classifier:local" in call for call in calls)
    assert any("up -d --no-build --pull never --wait --wait-timeout 60 app" in call for call in calls)
    assert not any("build app" in call for call in calls)
    assert not any(" pull " in call for call in calls)
    assert all("Mit Platz" in call for call in calls if "--project-directory" in call)


def test_setup_builds_before_starting(launcher):
    result, calls = launcher(action="setup")
    assert result.returncode == 0, result.stdout + result.stderr
    build = next(i for i, call in enumerate(calls) if call.endswith("build app"))
    start = next(i for i, call in enumerate(calls) if " up " in call)
    assert build < start


def test_stop_keeps_volumes_and_does_not_start_another_container(launcher):
    result, calls = launcher(action="stop")
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls[-1].endswith("stop app")
    assert not any(word in call.split() for call in calls for word in ("rm", "down", "up", "prune"))


@pytest.mark.parametrize("scenario,message", [
    ("engine-down", "Docker"), ("missing-compose", "Compose"),
    ("missing-image", "Image"),
])
def test_prerequisite_errors_do_not_attempt_start(launcher, scenario, message):
    result, calls = launcher(scenario=scenario)
    assert result.returncode != 0
    assert message in result.stdout + result.stderr
    assert not any(" up " in call or "build app" in call for call in calls)


def test_failed_build_stops_setup(launcher):
    result, calls = launcher(action="setup", scenario="build-failed")
    assert result.returncode != 0
    assert any("build app" in call for call in calls)
    assert not any(" up " in call for call in calls)


def test_unhealthy_container_does_not_report_success(launcher):
    result, calls = launcher(scenario="unhealthy")
    assert result.returncode != 0
    assert any(" up " in call for call in calls)
    assert "http://127.0.0.1:" not in result.stdout


@pytest.mark.parametrize("port", ["0", "65536", "abc"])
def test_invalid_port_is_rejected_before_docker(launcher, port):
    result, calls = launcher(port=port)
    assert result.returncode != 0
    assert "65535" in result.stdout + result.stderr
    assert not calls
