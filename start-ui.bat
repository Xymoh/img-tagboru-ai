@echo off
cd /d "%~dp0"
echo Starting Img-Tagboru native UI...
python -m frontend.native.main_window
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to start. Make sure Python and dependencies are installed:
    echo   pip install -r requirements.txt
    echo.
)
pause
