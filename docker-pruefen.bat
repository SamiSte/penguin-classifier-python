@echo off
setlocal EnableExtensions DisableDelayedExpansion
for %%I in ("%~dp0.") do set "PROJECT_DIR=%%~fI"
set "NO_PAUSE="

:arguments
if "%~1"=="" goto begin
if "%~1"=="--no-pause" (
    set "NO_PAUSE=1"
    shift
    goto arguments
)
echo Verwendung: docker-pruefen.bat [--no-pause]
exit /b 2

:begin
rem Eigene Namen verhindern, dass der Test produktive Beobachtungen veraendert.
set "TEST_GUID="
for /f "delims=" %%G in ('powershell.exe -NoProfile -NonInteractive -Command "[guid]::NewGuid().ToString('N')"') do set "TEST_GUID=%%G"
if not defined TEST_GUID (
    echo FEHLGESCHLAGEN: Die eindeutige Test-ID konnte nicht erstellt werden.
    echo Bitte pruefen, ob Windows PowerShell verfuegbar ist.
    set "RESULT=1"
    goto finish
)
set "PENGUIN_CONTAINER_TEST_ID=penguin-pruefung-%TEST_GUID%"
set "COMPOSE_PROJECT_NAME=%PENGUIN_CONTAINER_TEST_ID%"
set "PENGUIN_DATA_VOLUME=%PENGUIN_CONTAINER_TEST_ID%-data"
set "PENGUIN_MODELS_VOLUME=%PENGUIN_CONTAINER_TEST_ID%-models"
set "PENGUIN_PORT=8057"
set "COMPOSE_FILE=%PROJECT_DIR%\compose.yaml"
set "PROBE_FILE=%PROJECT_DIR%\scripts\container_check.py"
set "LOG_DIR=%PROJECT_DIR%\tmp"
set "LOG_FILE=%LOG_DIR%\docker-pruefung-%TEST_GUID%.log"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%LOG_DIR%" (
    echo FEHLGESCHLAGEN: Der Protokollordner konnte nicht angelegt werden.
    set "RESULT=1"
    goto finish
)
echo Die isolierte Docker-Pruefung laeuft. Das Training kann einige Minuten dauern.
echo Protokoll: "%LOG_FILE%"
call :run >"%LOG_FILE%" 2>&1
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" (
    echo BESTANDEN: Speichern, Training, Modellwechsel und Persistenz wurden geprueft.
) else (
    echo FEHLGESCHLAGEN: Bitte das Protokoll fuer die Fehlersuche oeffnen.
)
echo Protokoll: "%LOG_FILE%"
echo Die isolierten Testvolumes bleiben fuer die Nachpruefung erhalten.

:finish
if not defined NO_PAUSE pause
exit /b %RESULT%

:run
echo Test-ID: %PENGUIN_CONTAINER_TEST_ID%
echo Projekt: %PROJECT_DIR%
echo Testport: %PENGUIN_PORT%
echo Datenvolume: %PENGUIN_DATA_VOLUME%
echo Modellvolume: %PENGUIN_MODELS_VOLUME%
set "CONTAINERS_STARTED="
if not exist "%COMPOSE_FILE%" (
    echo FEHLER: compose.yaml fehlt.
    exit /b 1
)
if not exist "%PROBE_FILE%" (
    echo FEHLER: scripts\container_check.py fehlt.
    exit /b 1
)
set "DOCKER_EXE="
for /f "delims=" %%D in ('where docker 2^>nul') do if not defined DOCKER_EXE set "DOCKER_EXE=%%D"
if defined DOCKER_EXE goto docker_found
if defined LOCALAPPDATA if exist "%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe" set "DOCKER_EXE=%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe"
if defined DOCKER_EXE goto docker_found
if defined ProgramFiles if exist "%ProgramFiles%\Docker\Docker\resources\bin\docker.exe" set "DOCKER_EXE=%ProgramFiles%\Docker\Docker\resources\bin\docker.exe"
if defined DOCKER_EXE goto docker_found
if defined LOCALAPPDATA if exist "%LOCALAPPDATA%\Docker\resources\bin\docker.exe" set "DOCKER_EXE=%LOCALAPPDATA%\Docker\resources\bin\docker.exe"
if defined DOCKER_EXE goto docker_found
echo FEHLER: Docker wurde nicht gefunden. Bitte Docker Desktop einrichten.
exit /b 1

:docker_found
call "%DOCKER_EXE%" info --format "{{.ServerVersion}}"
if errorlevel 1 (
    echo FEHLER: Die Docker-Engine ist nicht erreichbar.
    exit /b 1
)
call :compose version
if errorlevel 1 (
    echo FEHLER: Docker Compose ist nicht verfuegbar.
    exit /b 1
)
call "%DOCKER_EXE%" image inspect penguin-classifier:local
if errorlevel 1 (
    echo FEHLER: Das lokale Image fehlt. Bitte zuerst setup.bat ausfuehren.
    exit /b 1
)
call "%DOCKER_EXE%" volume inspect "%PENGUIN_DATA_VOLUME%"
if not errorlevel 1 (
    echo FEHLER: Das Daten-Testvolume existiert bereits. Abbruch ohne Aenderungen.
    exit /b 1
)
call "%DOCKER_EXE%" volume inspect "%PENGUIN_MODELS_VOLUME%"
if not errorlevel 1 (
    echo FEHLER: Das Modell-Testvolume existiert bereits. Abbruch ohne Aenderungen.
    exit /b 1
)

echo SCHRITT 1: Isolierten Testcontainer starten.
set "CONTAINERS_STARTED=1"
call :compose up -d --no-build --pull never --wait --wait-timeout 60 app
if errorlevel 1 goto failed
echo SCHRITT 2: Beobachtung speichern, trainieren und Modell uebernehmen.
call :compose exec -T -e PENGUIN_CONTAINER_TEST_ID=%PENGUIN_CONTAINER_TEST_ID% app python - prepare <"%PROBE_FILE%"
if errorlevel 1 goto failed
echo SCHRITT 3: Testcontainer entfernen. Die Testvolumes bleiben bestehen.
call :compose down
if errorlevel 1 goto failed
echo SCHRITT 4: Container aus demselben Image und denselben Volumes neu erstellen.
call :compose up -d --no-build --pull never --wait --wait-timeout 60 app
if errorlevel 1 goto failed
echo SCHRITT 5: Daten, Modellversionen und Rueckkehr zum vorherigen Modell pruefen.
call :compose exec -T -e PENGUIN_CONTAINER_TEST_ID=%PENGUIN_CONTAINER_TEST_ID% app python - verify <"%PROBE_FILE%"
if errorlevel 1 goto failed
echo SCHRITT 6: Ausschliesslich den Testcontainer beenden.
call :compose down
if errorlevel 1 goto failed
echo BESTANDEN
exit /b 0

:failed
echo FEHLGESCHLAGEN: Der letzte Pruefschritt hat einen Fehler gemeldet.
if not defined CONTAINERS_STARTED exit /b 1
echo Container-Protokolle zur Diagnose:
call :compose logs --no-color --tail 200 app
echo Isolierten Testcontainer beenden. Testvolumes bleiben bestehen.
call :compose down
exit /b 1

:compose
rem Ein explizites Projekt ist auch bei Fehlerbehandlung und down zwingend.
call "%DOCKER_EXE%" compose -p "%COMPOSE_PROJECT_NAME%" --project-directory "%PROJECT_DIR%" -f "%COMPOSE_FILE%" %*
exit /b %ERRORLEVEL%
