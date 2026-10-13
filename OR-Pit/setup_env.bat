@echo off
REM OR-PiT environment setup for Windows + NVIDIA GPU.
REM   setup_env.bat                          (defaults: venv in .venv\, CUDA 12.4 wheels)
REM   set ORPIT_VENV=D:\envs\orpit ^& set ORPIT_PY=py -3.11 ^& set ORPIT_CUDA=cu118 ^& setup_env.bat
REM Put the venv outside a cloud-synced folder (OneDrive, Dropbox): PyTorch is ~3 GB.
setlocal
if "%ORPIT_VENV%"=="" set ORPIT_VENV=%~dp0.venv
if "%ORPIT_CUDA%"=="" set ORPIT_CUDA=cu124
if "%ORPIT_PY%"=="" (
  py -3.11 -c "" 2>nul && set "ORPIT_PY=py -3.11"
)
if "%ORPIT_PY%"=="" (
  py -3.12 -c "" 2>nul && set "ORPIT_PY=py -3.12"
)
if "%ORPIT_PY%"=="" (
  py -3.10 -c "" 2>nul && set "ORPIT_PY=py -3.10"
)
if "%ORPIT_PY%"=="" set ORPIT_PY=python
echo [setup] python: %ORPIT_PY%   venv: %ORPIT_VENV%   torch wheels: %ORPIT_CUDA%
%ORPIT_PY% --version || (echo [setup] Python 3.10-3.12 not found & exit /b 1)
if not exist "%ORPIT_VENV%\Scripts\python.exe" %ORPIT_PY% -m venv "%ORPIT_VENV%" || exit /b 1
"%ORPIT_VENV%\Scripts\python.exe" -m pip install --upgrade pip==24.2 || exit /b 1
"%ORPIT_VENV%\Scripts\python.exe" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/%ORPIT_CUDA% || exit /b 1
"%ORPIT_VENV%\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt" || exit /b 1
"%ORPIT_VENV%\Scripts\python.exe" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available());print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
echo [setup] done. Activate with: %ORPIT_VENV%\Scripts\activate
