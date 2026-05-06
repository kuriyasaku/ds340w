@echo off
setlocal
cd /d "%~dp0"
echo This will remove outputs.
set /p confirm=Type YES to continue: 
if not "%confirm%"=="YES" (
  echo cancelled
  pause
  exit /b 0
)
rmdir /s /q outputs
mkdir outputs
pause
