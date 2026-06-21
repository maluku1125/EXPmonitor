@echo off
cd /d "%~dp0"
echo ============================================
echo   EXPMonitor  build  (PyInstaller, onedir)
echo ============================================
echo.

echo [1/3] Installing packages...
pip install pyinstaller PyQt5 pyqtgraph numpy opencv-python mss pywin32 Pillow pytesseract --quiet
if errorlevel 1 ( echo pip failed & pause & exit /b 1 )

echo [2/3] Building EXE...
pyinstaller --noconfirm --onedir --windowed --name "EXPMonitor" ^
  --add-data "exp_monitor.py;." ^
  --add-data "exp_template_ocr.py;." ^
  --add-data "erda_ocr.py;." ^
  --add-data "templates;templates" ^
  --hidden-import win32gui --hidden-import win32con --hidden-import win32ui ^
  --hidden-import win32api --hidden-import pywintypes ^
  --hidden-import mss --hidden-import mss.windows ^
  --hidden-import pyqtgraph --hidden-import numpy --hidden-import cv2 ^
  --hidden-import pytesseract ^
  --exclude-module easyocr --exclude-module torch --exclude-module torchvision ^
  --exclude-module matplotlib --exclude-module tkinter ^
  exp_monitor_qt.py
if errorlevel 1 ( echo Build failed & pause & exit /b 1 )

echo [3/3] Copying Erda templates next to EXE and cleaning up...
rem Erda templates live next to the exe (writable; runtime calibration saves here)
if exist templates_erda xcopy /e /i /y "templates_erda" "dist\EXPMonitor\templates_erda" >nul
if exist templates_erda_badge xcopy /e /i /y "templates_erda_badge" "dist\EXPMonitor\templates_erda_badge" >nul
if exist build rmdir /s /q build
if exist EXPMonitor.spec del /q EXPMonitor.spec

echo.
echo ============================================
echo   Done. Output folder is  dist\EXPMonitor\
echo   Run the app:  dist\EXPMonitor\EXPMonitor.exe
echo ============================================
echo   templates and recognition core are bundled inside.
echo   (Optional) Tesseract fallback: install Tesseract-OCR and add to PATH.
echo.
pause
