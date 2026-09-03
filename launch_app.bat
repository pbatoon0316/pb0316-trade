@echo off
setlocal

rem Run from the folder containing this launcher, even when opened elsewhere.
pushd "%~dp0"

if not exist "home.py" (
    echo ERROR: home.py was not found in:
    echo %CD%
    goto :failed
)

rem First use Streamlit directly when the current Python/Conda environment is on PATH.
where streamlit >nul 2>&1
if not errorlevel 1 goto :run_streamlit

rem Check common per-user Anaconda and Miniconda installations.
call :try_python "%USERPROFILE%\anaconda3\python.exe"
if not errorlevel 1 goto :run_python

call :try_python "%LOCALAPPDATA%\anaconda3\python.exe"
if not errorlevel 1 goto :run_python

call :try_python "%USERPROFILE%\miniconda3\python.exe"
if not errorlevel 1 goto :run_python

call :try_python "%LOCALAPPDATA%\miniconda3\python.exe"
if not errorlevel 1 goto :run_python

call :try_python "%ProgramData%\anaconda3\python.exe"
if not errorlevel 1 goto :run_python

rem Fall back to Python commands available on PATH.
call :try_python "python"
if not errorlevel 1 goto :run_python

py -3 -c "import streamlit" >nul 2>&1
if not errorlevel 1 goto :run_py

echo ERROR: Streamlit was not found in any available Python environment.
echo.
echo Open an Anaconda Prompt and install this project's requirements with:
echo     pip install -r requirements.txt
echo.
echo Then run this launcher again.
goto :failed

:run_streamlit
echo Starting Trading Tools...
echo Keep this window open while using the app.
echo Press Ctrl+C here when you want to stop it.
echo.
streamlit run "home.py"
set "APP_EXIT=%ERRORLEVEL%"
goto :finished

:run_python
echo Starting Trading Tools...
echo Keep this window open while using the app.
echo Press Ctrl+C here when you want to stop it.
echo.
"%PYTHON_EXE%" -m streamlit run "home.py"
set "APP_EXIT=%ERRORLEVEL%"
goto :finished

:run_py
echo Starting Trading Tools...
echo Keep this window open while using the app.
echo Press Ctrl+C here when you want to stop it.
echo.
py -3 -m streamlit run "home.py"
set "APP_EXIT=%ERRORLEVEL%"
goto :finished

:try_python
set "PYTHON_EXE=%~1"
if not exist "%PYTHON_EXE%" (
    where "%PYTHON_EXE%" >nul 2>&1
    if errorlevel 1 exit /b 1
)
"%PYTHON_EXE%" -c "import streamlit" >nul 2>&1
exit /b %ERRORLEVEL%

:finished
if not "%APP_EXIT%"=="0" (
    echo.
    echo The app stopped with an error.
    pause
)
popd
exit /b %APP_EXIT%

:failed
echo.
pause
popd
exit /b 1
