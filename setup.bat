@echo off
REM DataPrep AI Platform Setup Script for Windows
REM This script sets up the development environment

echo 🚀 Setting up DataPrep AI Platform...

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Create .env file from example if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from example...
    copy .env.example .env
    echo ✅ .env file created. Please review and update the configuration as needed.
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist backend\uploads mkdir backend\uploads
if not exist backend\alembic\versions mkdir backend\alembic\versions
if not exist frontend\public mkdir frontend\public
if not exist docs\api mkdir docs\api
if not exist docs\deployment mkdir docs\deployment
if not exist docs\development mkdir docs\development
if not exist docs\user-guide mkdir docs\user-guide

echo 🐳 Starting Docker services...
docker-compose up -d postgres redis minio

echo ⏳ Waiting for services to be ready...
timeout /t 10 /nobreak >nul

echo 🔍 Checking service health...
docker-compose ps

echo.
echo ✅ Setup complete!
echo.
echo 🎯 Next steps:
echo 1. Review and update the .env file with your configuration
echo 2. Install Python dependencies: cd backend ^&^& pip install -r requirements.txt
echo 3. Install Node.js dependencies: cd frontend ^&^& npm install
echo 4. Start the backend: cd backend ^&^& uvicorn app.main:app --reload
echo 5. Start the frontend: cd frontend ^&^& npm run dev
echo 6. Visit http://localhost:3000 to access the application
echo.
echo 📚 Documentation:
echo - API docs: http://localhost:8000/docs
echo - MinIO console: http://localhost:9001 (admin/minioadmin123)
echo - Celery Flower: http://localhost:5555
echo.
pause