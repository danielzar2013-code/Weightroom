@echo off
title Weightroom GPU
echo Weightroom GPU: connecting this PC's graphics card to the Weightroom app.
echo The first run downloads about 7 GB of programs and can take 15 to 30 minutes.
echo.
set "PS1=%TEMP%\weightroom-gpu.ps1"
curl.exe -fsSL -o "%PS1%" https://danielzar2013-code.github.io/Weightroom/gpu/windows.ps1
if errorlevel 1 (
  echo Could not download the setup script. Check your internet connection and try again.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%"
