@echo off
:: EKDIST-app.bat — launch the EKDIST Streamlit application
::
:: Activates the dcprogs conda environment, prepends the DLL directory so
:: numpy linalg works correctly, then starts the Streamlit server and opens
:: the browser automatically.
::
:: Double-click this file from Explorer, or run it from any terminal.

setlocal

set CONDA_ROOT=C:\ProgramData\miniconda3
set ENV_NAME=dcprogs
set ENV_DIR=%CONDA_ROOT%\envs\%ENV_NAME%
set APP=%~dp0app.py

:: Prepend the conda Library\bin directory so numpy/scipy DLLs are found
set PATH=%ENV_DIR%\Library\bin;%PATH%

:: Activate the conda environment
call "%CONDA_ROOT%\Scripts\activate.bat" %ENV_NAME%

:: Verify streamlit is available before launching
where streamlit >nul 2>&1
if errorlevel 1 (
    echo ERROR: streamlit not found in the "%ENV_NAME%" environment.
    echo Install it with:  conda activate %ENV_NAME%  ^&^&  pip install streamlit
    pause
    exit /b 1
)

echo Starting EKDIST app ...
echo   http://localhost:8501
echo.
streamlit run "%APP%" --server.port 8501 --server.headless false

endlocal
