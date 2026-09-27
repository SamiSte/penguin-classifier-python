@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "PENGUIN_LAUNCHER=%~dp0scripts\docker-launcher.bat"
set "PENGUIN_NO_BROWSER="
set "PENGUIN_NO_PAUSE="
set "PENGUIN_BAD_ARGUMENT="

:parse_arguments
if "%~1"=="" goto arguments_ready
if /i "%~1"=="--no-browser" goto no_browser
if /i "%~1"=="--no-pause" goto no_pause
set "PENGUIN_BAD_ARGUMENT=1"
shift
goto parse_arguments

:no_browser
set "PENGUIN_NO_BROWSER=1"
shift
goto parse_arguments

:no_pause
set "PENGUIN_NO_PAUSE=1"
shift
goto parse_arguments

:arguments_ready
if defined PENGUIN_BAD_ARGUMENT goto usage
call "%PENGUIN_LAUNCHER%" setup
set "PENGUIN_EXIT_CODE=%ERRORLEVEL%"
goto finish

:usage
echo Aufruf: setup.bat [--no-browser] [--no-pause]
set "PENGUIN_EXIT_CODE=2"

:finish
if not defined PENGUIN_NO_PAUSE (
    echo.
    pause
)
exit /b %PENGUIN_EXIT_CODE%
