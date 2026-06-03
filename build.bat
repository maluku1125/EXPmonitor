@echo off
cd /d "%~dp0"

echo [1/3] Installing packages...
pip install pyinstaller PyQt5 pyqtgraph numpy mss pytesseract easyocr pywin32 --quiet
if errorlevel 1 ( echo pip failed & pause & exit /b 1 )

echo [2/3] Building with PyInstaller...
pyinstaller --noconfirm --onedir --windowed --name "EXPMonitor" --add-data "exp_monitor.py;." --hidden-import win32gui --hidden-import win32con --hidden-import win32ui --hidden-import win32api --hidden-import pywintypes --hidden-import mss --hidden-import mss.windows --hidden-import pytesseract --hidden-import easyocr --hidden-import pyqtgraph --hidden-import numpy --hidden-import cv2 exp_monitor_qt.py
if errorlevel 1 ( echo Build failed & pause & exit /b 1 )

echo [3/3] Cleaning up...
if exist build rmdir /s /q build
if exist EXPMonitor.spec del /q EXPMonitor.spec

echo.
echo Done! Output: dist\EXPMonitor\EXPMonitor.exe
echo Note: Tesseract must be installed separately and added to PATH.
echo.
pause
