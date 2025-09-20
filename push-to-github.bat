@echo off
echo Pushing Tamil News Summarizer to GitHub...
echo.

REM Initialize git if not already done
if not exist .git (
    echo Initializing Git repository...
    git init
    echo.
)

REM Add remote origin
echo Adding GitHub remote...
git remote remove origin 2>nul
git remote add origin https://github.com/Ashy-it24/SAARAM.git
echo.

REM Add all files
echo Adding all files...
git add .
echo.

REM Create .gitignore if it doesn't exist
if not exist .gitignore (
    echo Creating .gitignore...
    (
        echo # Dependencies
        echo node_modules/
        echo __pycache__/
        echo *.pyc
        echo venv/
        echo env/
        echo.
        echo # Build outputs
        echo .next/
        echo out/
        echo dist/
        echo build/
        echo.
        echo # Cache and logs
        echo *.log
        echo .cache/
        echo hf_cache/
        echo.
        echo # Environment files
        echo .env
        echo .env.local
        echo .env.production
        echo.
        echo # IDE files
        echo .vscode/
        echo .idea/
        echo *.swp
        echo *.swo
        echo.
        echo # OS files
        echo .DS_Store
        echo Thumbs.db
        echo desktop.ini
    ) > .gitignore
    git add .gitignore
)

REM Commit changes
echo Committing changes...
git commit -m "Initial commit: Tamil News Summarizer with MyMemory API toggle

Features:
- AI-powered Tamil text summarization using IndicBART
- MyMemory API integration with toggle option
- Text-to-Speech for Tamil content
- Question-answering chatbot
- Modern Next.js frontend with React
- FastAPI backend with comprehensive error handling
- Translation between Tamil and English
- Compression statistics and performance metrics"
echo.

REM Push to GitHub
echo Pushing to GitHub...
git branch -M main
git push -u origin main
echo.

echo Done! Your project has been pushed to:
echo https://github.com/Ashy-it24/SAARAM
echo.
pause