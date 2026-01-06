#!/bin/bash

# DataPrep AI Platform Setup Script
# This script sets up the development environment

echo "🚀 Setting up DataPrep AI Platform..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file from example if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "✅ .env file created. Please review and update the configuration as needed."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p backend/uploads
mkdir -p backend/alembic/versions
mkdir -p frontend/public
mkdir -p docs/api
mkdir -p docs/deployment
mkdir -p docs/development
mkdir -p docs/user-guide

# Set up backend
echo "🐍 Setting up backend..."
if [ -d "backend" ]; then
    cd backend
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment and install dependencies
    echo "Installing Python dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Create initial migration (if database is available)
    echo "Creating initial database migration..."
    python -m alembic revision --autogenerate -m "Initial migration" || echo "⚠️  Database migration creation skipped (database not available)"
    
    cd ..
fi

# Set up frontend
echo "⚛️  Setting up frontend..."
if [ -d "frontend" ]; then
    cd frontend
    
    # Install Node.js dependencies
    if command -v npm &> /dev/null; then
        echo "Installing Node.js dependencies..."
        npm install
    else
        echo "⚠️  npm not found. Please install Node.js to set up the frontend."
    fi
    
    cd ..
fi

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d postgres redis minio

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Review and update the .env file with your configuration"
echo "2. Start the backend: cd backend && source venv/bin/activate && uvicorn app.main:app --reload"
echo "3. Start the frontend: cd frontend && npm run dev"
echo "4. Visit http://localhost:3000 to access the application"
echo ""
echo "📚 Documentation:"
echo "- API docs: http://localhost:8000/docs"
echo "- MinIO console: http://localhost:9001 (admin/minioadmin123)"
echo "- Celery Flower: http://localhost:5555"
echo ""
echo "🛠️  Development commands:"
echo "- Run tests: cd backend && pytest"
echo "- Run frontend tests: cd frontend && npm test"
echo "- Stop services: docker-compose down"
echo ""