@echo off
echo Starting Auto-WZRY Launcher...
conda run --no-capture-output -n wzry python start.py
if errorlevel 1 (
    echo.
    echo Error: Failed to run start.py in 'wzry' environment.
    echo Please make sure you have installed Anaconda/Miniconda and created the 'wzry' environment.
    pause
)
