@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "PENGUIN_VIDEO_MODE=run"
set "PENGUIN_VIDEO_BROWSER=--open-browser"
set "PENGUIN_VIDEO_NO_PAUSE="
set "PENGUIN_VIDEO_RUNTIME="

:parse_arguments
if "%~1"=="" goto find_python
if /i "%~1"=="check" goto check_mode
if /i "%~1"=="--no-browser" goto no_browser
if /i "%~1"=="--no-pause" goto no_pause
echo Aufruf: video-demo.bat [check] [--no-browser] [--no-pause]
set "PENGUIN_VIDEO_EXIT=2"
goto finish

:check_mode
set "PENGUIN_VIDEO_MODE=check"
shift
goto parse_arguments

:no_browser
set "PENGUIN_VIDEO_BROWSER="
shift
goto parse_arguments

:no_pause
set "PENGUIN_VIDEO_NO_PAUSE=1"
shift
goto parse_arguments

:find_python
if defined PENGUIN_VIDEO_PYTHON goto explicit_python
call :try_python "%~dp0.venv\Scripts\python.exe"
if defined PENGUIN_VIDEO_RUNTIME goto execute
call :try_python "%USERPROFILE%\anaconda3\envs\penguin-ml\python.exe"
if defined PENGUIN_VIDEO_RUNTIME goto execute
call :try_python "python"
if defined PENGUIN_VIDEO_RUNTIME goto execute
echo FEHLER: Kein Python mit den Projektpaketen gefunden.
echo Eine vorbereitete Python-Umgebung verwenden. Es wird nichts installiert.
echo Alternativ PENGUIN_VIDEO_PYTHON auf deren python.exe setzen.
set "PENGUIN_VIDEO_EXIT=1"
goto finish

:explicit_python
call :try_python "%PENGUIN_VIDEO_PYTHON%"
if defined PENGUIN_VIDEO_RUNTIME goto execute
echo FEHLER: PENGUIN_VIDEO_PYTHON ist ungueltig oder Projektpakete fehlen.
set "PENGUIN_VIDEO_EXIT=1"
goto finish

:try_python
"%~1" -c "import dash,pandas,sklearn,plotly,joblib,numpy" >nul 2>&1
if errorlevel 1 exit /b 0
set "PENGUIN_VIDEO_RUNTIME=%~1"
exit /b 0

:execute
echo Python: %PENGUIN_VIDEO_RUNTIME%
if "%PENGUIN_VIDEO_MODE%"=="check" set "PENGUIN_VIDEO_BROWSER="
"%PENGUIN_VIDEO_RUNTIME%" "%~dp0scripts\video_demo.py" %PENGUIN_VIDEO_MODE% %PENGUIN_VIDEO_BROWSER%
set "PENGUIN_VIDEO_EXIT=%ERRORLEVEL%"

:finish
if not defined PENGUIN_VIDEO_NO_PAUSE pause
exit /b %PENGUIN_VIDEO_EXIT%
