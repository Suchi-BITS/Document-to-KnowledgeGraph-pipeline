@echo off
REM setup.bat - Windows setup script for Knowledge Graph Builder

echo ==========================================
echo Knowledge Graph Builder - Setup Script
echo ==========================================
echo.

REM Check Python version
echo 1. Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)
echo    √ Python found
echo.

REM Create virtual environment
echo 2. Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo    √ Virtual environment created
) else (
    echo    √ Virtual environment already exists
)
echo.

REM Activate virtual environment
echo 3. Activating virtual environment...
call .venv\Scripts\activate.bat
echo    √ Virtual environment activated
echo.

REM Upgrade pip
echo 4. Upgrading pip...
python -m pip install --upgrade pip --quiet
echo    √ pip upgraded
echo.

REM Install dependencies
echo 5. Installing dependencies...
pip install -r requirements.txt --quiet
echo    √ Dependencies installed
echo.

REM Create directories
echo 6. Creating directory structure...
if not exist "data\input" mkdir data\input
if not exist "data\output" mkdir data\output
if not exist "notebooks" mkdir notebooks
if not exist "tests" mkdir tests
if not exist "logs" mkdir logs
echo    √ Directories created
echo.

REM Create .gitkeep files
type nul > data\input\.gitkeep
type nul > data\output\.gitkeep

REM Setup environment file
echo 7. Setting up environment configuration...
if not exist ".env" (
    copy .env.example .env
    echo    √ .env file created from template
    echo.
    echo    WARNING: Edit .env file with your API key!
    echo       You can use: notepad .env
) else (
    echo    √ .env file already exists
)
echo.

REM Create sample input file
echo 8. Creating sample input file...
if not exist "data\input\sample.txt" (
    (
        echo Marie Curie, born Maria Sklodowska in Warsaw, Poland, was a pioneering physicist and chemist.
        echo She conducted groundbreaking research on radioactivity. Together with her husband, Pierre Curie,
        echo she discovered the elements polonium and radium. Marie Curie was the first woman to win a Nobel Prize.
    ) > data\input\sample.txt
    echo    √ Sample file created
) else (
    echo    √ Sample file already exists
)
echo.

echo ==========================================
echo √ Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo   1. Edit .env with your API credentials:
echo      notepad .env
echo.
echo   2. Run the example:
echo      python main.py data\input\sample.txt
echo.
echo   3. Or start Jupyter notebook:
echo      jupyter notebook notebooks\demo.ipynb
echo.
echo   4. Run tests:
echo      pytest tests\ -v
echo.
echo For help:
echo   python main.py --help
echo.
echo ==========================================
pause