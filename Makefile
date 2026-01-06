# DataPrep AI Platform Makefile
# Common development tasks

.PHONY: help setup start stop test clean build deploy

# Default target
help:
	@echo "DataPrep AI Platform - Available commands:"
	@echo ""
	@echo "  setup     - Set up the development environment"
	@echo "  start     - Start all services"
	@echo "  stop      - Stop all services"
	@echo "  test      - Run all tests"
	@echo "  clean     - Clean up containers and volumes"
	@echo "  build     - Build Docker images"
	@echo "  deploy    - Deploy to production"
	@echo "  backend   - Start backend development server"
	@echo "  frontend  - Start frontend development server"
	@echo "  migrate   - Run database migrations"
	@echo "  lint      - Run linting on all code"
	@echo ""

# Setup development environment
setup:
	@echo "🚀 Setting up DataPrep AI Platform..."
	@chmod +x setup.sh
	@./setup.sh

# Start all services
start:
	@echo "🐳 Starting all services..."
	@docker-compose up -d

# Stop all services
stop:
	@echo "🛑 Stopping all services..."
	@docker-compose down

# Run all tests
test:
	@echo "🧪 Running backend tests..."
	@cd backend && python -m pytest
	@echo "🧪 Running frontend tests..."
	@cd frontend && npm test

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	@docker-compose down -v
	@docker system prune -f

# Build Docker images
build:
	@echo "🔨 Building Docker images..."
	@docker-compose build

# Start backend development server
backend:
	@echo "🐍 Starting backend development server..."
	@cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend development server
frontend:
	@echo "⚛️  Starting frontend development server..."
	@cd frontend && npm run dev

# Run database migrations
migrate:
	@echo "🗄️  Running database migrations..."
	@cd backend && python -m alembic upgrade head

# Run linting
lint:
	@echo "🔍 Running backend linting..."
	@cd backend && flake8 app
	@cd backend && black app --check
	@cd backend && isort app --check-only
	@echo "🔍 Running frontend linting..."
	@cd frontend && npm run lint

# Install dependencies
install:
	@echo "📦 Installing backend dependencies..."
	@cd backend && pip install -r requirements.txt
	@echo "📦 Installing frontend dependencies..."
	@cd frontend && npm install

# Format code
format:
	@echo "✨ Formatting backend code..."
	@cd backend && black app
	@cd backend && isort app
	@echo "✨ Formatting frontend code..."
	@cd frontend && npm run lint --fix