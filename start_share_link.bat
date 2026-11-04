@echo off
title AI Smart Traffic System - Live Host & Public Share
echo ========================================================
echo  Starting AI Smart Traffic Violation System
echo ========================================================
echo.

echo [1/3] Starting FastAPI Backend on port 8000...
start /b uvicorn backend.main:app --host 127.0.0.1 --port 8000

timeout /t 3 >nul

echo [2/3] Starting Streamlit Dashboard on port 8501...
start /b streamlit run dashboard/app.py --server.headless true --server.port 8501

timeout /t 3 >nul

echo [3/3] Generating Public Shareable Link via Cloudflare Tunnel...
echo.
echo ========================================================
echo Look for the https://*.trycloudflare.com link below!
echo Share that link with your friends to open on any phone/PC.
echo ========================================================
echo.
.\cloudflared.exe tunnel --url http://localhost:8501
pause

