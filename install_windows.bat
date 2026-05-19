@echo off
if "%1" neq "_stayopen" (
    powershell -NoProfile -Command "& { & '%~f0' _stayopen %* 2>&1 | Tee-Object -FilePath installed.txt }"
    exit /b
)
setlocal enabledelayedexpansion

echo.
echo ==========================================
echo HomographFix Windows Installer
echo ==========================================
echo.

REM Check for Python 3.12 via launcher
echo Checking for Python 3.12...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.12 not found. Installing via winget...
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo ERROR: Could not auto-install Python 3.12. Please install manually from python.org
        pause
    )
    echo Python 3.12 installed. You may need to restart your terminal.
    py -3.12 --version
    if errorlevel 1 (
        echo ERROR: Launcher still cannot find 3.12. Restart cmd and re-run installer.
        pause
    )
    pause
)

py -3.12 --version
echo.

REM Create venv with Python 3.12
echo Creating virtual environment with Python 3.12...
py -3.12 -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
)

REM Use direct venv paths (more reliable than activate)
set VPY=venv\Scripts\python.exe
set VPIP=venv\Scripts\python.exe -m pip

REM Upgrade pip
echo Upgrading pip...
%VPIP% install --upgrade pip setuptools wheel

REM Detect CUDA / GPU
echo.
echo Detecting GPU/CUDA availability...
nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo No NVIDIA GPU detected. Installing PyTorch CPU version...
    %VPIP% install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
) else (
    echo NVIDIA GPU + CUDA detected. Installing PyTorch with CUDA support...
    %VPIP% install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu124
)
echo GPU detection and PyTorch install complete.

REM Install other packages (no pins: use latest compatible with your Python)
echo.
echo Installing transformers, spacy, and spacy-transformers...
%VPIP% install transformers spacy spacy-transformers
echo Additional packages installed.

REM Download spaCy model
echo.
echo Downloading spaCy transformer model (en_core_web_trf)...
%VPY% -m spacy download en_core_web_trf
if errorlevel 1 (
    echo ERROR: spaCy model download failed.
) else (
    echo spaCy model downloaded.
)

REM Create folders
echo.
echo Creating InputText and OutputText folders...
if not exist InputText mkdir InputText
if not exist OutputText mkdir OutputText
echo Folders ready.

REM Done
echo.
echo ==========================================
echo Installation complete! All done.
echo ==========================================
echo.
echo Next steps:
echo   1. Activate venv: venv\Scripts\activate
echo   2. Run the program: python word_hybrid_test.py
echo.
echo NOTE: First run will download roberta-large-mnli (approx 1.4 GB)
echo       from HuggingFace -- this is normal and one-time only.
echo.
echo Press any key to close...
pause
