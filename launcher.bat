@echo off
echo ===============================================
echo    Document-Based Chatbot Launcher
echo ===============================================
echo.
echo Select an option:
echo 1. Run CLI Chatbot
echo 2. Run Streamlit Web App
echo 3. Test Document Loader
echo 4. Test Gemini API
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo Starting CLI Chatbot...
    C:/Users/admin/AppData/Local/Programs/Python/Python313/python.exe cli_chatbot.py
) else if "%choice%"=="2" (
    echo Starting Streamlit Web App...
    C:/Users/admin/AppData/Local/Programs/Python/Python313/python.exe -m streamlit run streamlit_app.py
) else if "%choice%"=="3" (
    echo Testing Document Loader...
    C:/Users/admin/AppData/Local/Programs/Python/Python313/python.exe document_loader.py
    pause
) else if "%choice%"=="4" (
    echo Testing Gemini API...
    C:/Users/admin/AppData/Local/Programs/Python/Python313/python.exe gemini_wrapper.py
    pause
) else if "%choice%"=="5" (
    echo Goodbye!
    exit
) else (
    echo Invalid choice. Please try again.
    pause
    goto start
)

:start
pause
