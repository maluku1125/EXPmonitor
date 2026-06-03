@echo off
cd /d "%~dp0"
echo ============================================
echo   EXPmonitor  Git setup
echo ============================================
echo.
where git >nul 2>nul
if errorlevel 1 (
  echo [ERROR] git not found. Install Git for Windows first:
  echo         https://git-scm.com/download/win
  pause
  exit /b 1
)
if exist .git (
  echo Removing old/broken .git ...
  rmdir /s /q .git
)
echo Initializing git ...
git init
git config user.email "maluku1125@gmail.com"
git config user.name "yu"
git add -A
git commit -m "baseline: template-OCR EXP monitor (head-cut fixed, templates rebuilt)"
echo.
echo Done.
echo   Save a checkpoint:  git add -A  then  git commit -m "message"
echo   Show history:       git log --oneline
echo   Undo uncommitted:   git checkout -- .
pause
