@echo off
CALL C:\Users\YourUsername\miniconda3\Scripts\activate.bat YourCondaEnvironmentName
SET PATH=%PATH%;C:\Some\Additional\Path
python C:\Path\To\YourScript\your_script.py
pause
