#!/bin/bash

# Medical Imaging AI API Setup Script

set -e

echo "🚀 Setting up Medical Imaging AI API..."

# Check if Python 3.11+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.11+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/uploads
mkdir -p data/processed
mkdir -p models
mkdir -p logs

# Set up environment file
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment file..."
    cp env.example .env
    echo "📝 Please edit .env file with your configuration"
fi

# Set up pre-commit hooks
echo "🔗 Setting up pre-commit hooks..."
pip install pre-commit
pre-commit install

# Create database tables (if using local database)
echo "🗄️ Setting up database..."
python -c "
from src.core.database import engine, Base
Base.metadata.create_all(bind=engine)
print('Database tables created successfully')
"

echo "✅ Setup complete!"
echo ""
echo "To start the API:"
echo "  source venv/bin/activate"
echo "  make run-dev"
echo ""
echo "To run tests:"
echo "  make test"
echo ""
echo "To start with Docker:"
echo "  make docker-up"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
