@echo off
echo Deploying to Heroku...
echo.

REM Create Procfile for backend
echo Creating Procfile for backend...
echo web: uvicorn main:app --host 0.0.0.0 --port $PORT > backend\Procfile

REM Create runtime.txt for Python version
echo python-3.9.18 > backend\runtime.txt

echo.
echo Files created for Heroku deployment:
echo - backend/Procfile
echo - backend/runtime.txt
echo.
echo Next steps:
echo 1. cd backend
echo 2. heroku create your-app-name-backend
echo 3. git subtree push --prefix backend heroku main
echo.
echo For frontend deployment to Vercel:
echo 1. cd frontend
echo 2. npm install -g vercel
echo 3. vercel --prod
echo.
pause