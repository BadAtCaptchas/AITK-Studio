@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
REM Windows bootstrap adapted from ostris/ai-toolkit's run_windows.bat.
REM The in-repo manager handles the environment, dependencies, and UI.
title AITK Studio
echo.
echo   AITK Studio Manager - Windows
echo.

REM Prevent another Python environment from hijacking the manager.
set PYTHONPATH=
set PYTHONHOME=
set PYTHONSTARTUP=
set PYTHONUSERBASE=
set PIP_CONFIG_FILE=
set VIRTUAL_ENV=
set CONDA_PREFIX=
set CONDA_DEFAULT_ENV=
set PYENV_ROOT=
set PYENV_VERSION=

REM Ensure uv is available, installing locally when needed.
set "PATH=%~dp0.uv;%PATH%"
set "UV_INSTALL_DIR=%~dp0.uv"
set "UV_NO_MODIFY_PATH=1"
set "UV_PYTHON_INSTALL_DIR=%~dp0.uv\python"
where uv.exe >nul 2>&1
if errorlevel 1 (
    echo Downloading uv into .uv\ ...
    powershell -NoProfile -ExecutionPolicy ByPass -Command ^
        "irm https://astral.sh/uv/install.ps1 | iex"
    where uv.exe >nul 2>&1
    if errorlevel 1 (
        echo ERROR: uv download failed. See https://docs.astral.sh/uv/
        pause
        exit /b 1
    )
)

REM Find a Python to run the stdlib-only manager, or provision Python 3.12.
set "PY="
for %%C in (python.exe py.exe) do (
    if not defined PY (
        %%C -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
        if not errorlevel 1 set "PY=%%C"
    )
)
if not defined PY (
    echo No system Python found - provisioning one with uv...
    uv python install 3.12
    if not errorlevel 1 (
        for /f "delims=" %%P in ('uv python find 3.12') do set "PY=%%P"
    )
)
if not defined PY (
    echo ERROR: could not find or install a Python interpreter.
    pause
    exit /b 1
)

REM Sync dependencies and update code when there are no local changes.
"%PY%" -m manager update --auto
if errorlevel 1 (
    echo.
    echo Setup failed - see output above.
    pause
    exit /b 1
)
"%PY%" -m manager launch
set "LAUNCH_EXIT=%ERRORLEVEL%"
pause
exit /b %LAUNCH_EXIT%
