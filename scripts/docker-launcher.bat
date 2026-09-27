@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Die Wrapper uebergeben ausschliesslich einen festen Aktionsnamen.
set "LAUNCH_ACTION=%~1"
if "%LAUNCH_ACTION%"=="setup" goto action_valid
if "%LAUNCH_ACTION%"=="start" goto action_valid
if "%LAUNCH_ACTION%"=="stop" goto action_valid
echo Bitte setup.bat, start.bat oder stop.bat im Projektordner aufrufen.
exit /b 2

:action_valid
for %%I in ("%~dp0..") do set "PROJECT_DIR=%%~fI"
set "COMPOSE_FILE=%PROJECT_DIR%\compose.yaml"
if not exist "%COMPOSE_FILE%" (
    echo FEHLER: compose.yaml fehlt. Bitte das vollstaendige Projekt verwenden.
    exit /b 1
)

rem Nur Ziffern zulassen, bevor der Wert in Befehle oder eine URL gelangt.
rem Verzoegerte Expansion verhindert, dass Sonderzeichen erneut geparst werden.
setlocal EnableDelayedExpansion
if not defined PENGUIN_PORT set "PENGUIN_PORT=8050"
set "PORT_INVALID=!PENGUIN_PORT!"
for %%D in (0 1 2 3 4 5 6 7 8 9) do if defined PORT_INVALID set "PORT_INVALID=!PORT_INVALID:%%D=!"
if defined PORT_INVALID goto invalid_port
if not "!PENGUIN_PORT:~5!"=="" goto invalid_port

:trim_port
if not "!PENGUIN_PORT:~0,1!"=="0" goto check_port_range
set "PENGUIN_PORT=!PENGUIN_PORT:~1!"
goto trim_port

:check_port_range
if not defined PENGUIN_PORT goto invalid_port
if !PENGUIN_PORT! GTR 65535 goto invalid_port
endlocal & set "PENGUIN_PORT=%PENGUIN_PORT%"
goto find_docker

:invalid_port
endlocal
echo FEHLER: PENGUIN_PORT muss eine ganze Zahl zwischen 1 und 65535 sein.
exit /b 2

:find_docker
set "DOCKER_EXE="
for /f "delims=" %%D in ('where docker 2^>nul') do if not defined DOCKER_EXE set "DOCKER_EXE=%%D"
if defined DOCKER_EXE goto docker_found
if defined LOCALAPPDATA if exist "%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe" set "DOCKER_EXE=%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe"
if defined DOCKER_EXE goto docker_found
if defined ProgramFiles if exist "%ProgramFiles%\Docker\Docker\resources\bin\docker.exe" set "DOCKER_EXE=%ProgramFiles%\Docker\Docker\resources\bin\docker.exe"
if defined DOCKER_EXE goto docker_found
if defined LOCALAPPDATA if exist "%LOCALAPPDATA%\Docker\resources\bin\docker.exe" set "DOCKER_EXE=%LOCALAPPDATA%\Docker\resources\bin\docker.exe"
if defined DOCKER_EXE goto docker_found
echo FEHLER: Docker wurde nicht gefunden.
echo Bitte Docker Desktop vor dem ersten Einsatz installieren und einrichten.
echo Danach Docker Desktop starten und setup.bat erneut aufrufen.
exit /b 1

:docker_found
call "%DOCKER_EXE%" info --format "{{.ServerVersion}}" >nul 2>&1
if errorlevel 1 (
    echo FEHLER: Docker ist installiert, aber nicht betriebsbereit.
    echo Bitte Docker Desktop starten und warten, bis die Engine laeuft.
    echo Falls Docker bereits laeuft, bitte Installation und Zugriffsrechte pruefen.
    exit /b 1
)
call "%DOCKER_EXE%" compose version >nul 2>&1
if errorlevel 1 (
    echo FEHLER: Docker Compose ist nicht verfuegbar.
    echo Bitte die Docker-Desktop-Installation pruefen.
    exit /b 1
)
if "%LAUNCH_ACTION%"=="stop" goto stop_app
if "%LAUNCH_ACTION%"=="setup" goto setup_app
call "%DOCKER_EXE%" image inspect penguin-classifier:local >nul 2>&1
if errorlevel 1 (
    echo FEHLER: Das Anwendungs-Image fehlt.
    echo Bitte einmal setup.bat mit Internetverbindung ausfuehren.
    echo Danach kann start.bat die vorbereitete Anwendung offline starten.
    exit /b 1
)
goto start_app

:setup_app
echo Das Anwendungs-Image wird erstellt. Dies kann einige Minuten dauern.
call "%DOCKER_EXE%" compose --project-directory "%PROJECT_DIR%" -f "%COMPOSE_FILE%" build app
if errorlevel 1 (
    echo FEHLER: Das Anwendungs-Image konnte nicht erstellt werden.
    echo Bitte die Meldungen oben und die Internetverbindung pruefen.
    exit /b 1
)

:start_app
echo Die Anwendung wird gestartet und auf Betriebsbereitschaft geprueft.
call "%DOCKER_EXE%" compose --project-directory "%PROJECT_DIR%" -f "%COMPOSE_FILE%" up -d --no-build --pull never --wait --wait-timeout 60 app
if errorlevel 1 (
    echo FEHLER: Die Anwendung wurde nicht rechtzeitig betriebsbereit.
    echo Bitte die Meldungen oben und die Container-Protokolle in Docker Desktop pruefen.
    echo Bei einem belegten Port kann PENGUIN_PORT auf einen anderen Port gesetzt werden.
    exit /b 1
)
echo.
echo Pinguin-Klassifikator: http://127.0.0.1:%PENGUIN_PORT%
if not defined PENGUIN_NO_BROWSER start "" "http://127.0.0.1:%PENGUIN_PORT%"
exit /b 0

:stop_app
call "%DOCKER_EXE%" compose --project-directory "%PROJECT_DIR%" -f "%COMPOSE_FILE%" stop app
if errorlevel 1 (
    echo FEHLER: Die Anwendung konnte nicht gestoppt werden.
    exit /b 1
)
echo Die Anwendung wurde gestoppt. Beobachtungen und Modelle bleiben erhalten.
exit /b 0
