@echo off
TITLE Sales Forecasting ML Dashboard Launcher

echo ========================================================
echo       🚀 Starting Sales Forecasting ML Platform...
echo ========================================================
echo.
echo  Developed by Shiva
echo  Initializing system components...
echo.

:: Check if python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python to run this application.
    pause
    exit
)

:: Run the application
echo  [INFO] Launching Streamlit Dashboard...
echo  [INFO] Your browser should open automatically.
echo.
echo  --------------------------------------------------------
echo  Keep this window open while using the dashboard.
echo  To close the application, close this window or press Ctrl+C.
echo  --------------------------------------------------------
echo.

streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The application crashed or failed to start.
    echo Checking for common issues...
    echo.
    pause
)
