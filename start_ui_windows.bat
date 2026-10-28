@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3 start_ui.py
) else (
    python start_ui.py
)
if errorlevel 1 pause
