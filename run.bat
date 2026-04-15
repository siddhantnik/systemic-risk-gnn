@echo off
echo ==========================================
echo Sovereign Observer - Systemic Risk Engine
echo ==========================================
echo.
echo [1/2] Starting FastAPI Backend (Port 8000)...
start "FastAPI Backend" "c:\Users\prana\New folder\python.exe" -m uvicorn backend.api.app:app --reload
echo.
echo [2/2] Starting React Frontend (Port 5173)...
start "React Frontend" cmd /c "cd frontend && npm.cmd run dev"
echo.
echo ==========================================
echo API Docs:  http://127.0.0.1:8000/docs
echo Frontend:  http://localhost:5173/
echo ==========================================
pause
