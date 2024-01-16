@echo off
PowerShell -Command "Start-Process python -ArgumentList 'F:\Code\HMIvoice\vision_train.py' -Wait -Verb RunAs"
if %ERRORLEVEL% NEQ 0 (
    echo Failed with error #%ERRORLEVEL%.
)
pause