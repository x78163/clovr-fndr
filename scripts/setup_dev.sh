#!/bin/bash
# Clovr Fndr - Development Environment Setup
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "Clovr Fndr - Development Setup"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# --- Python Virtual Environment ---
echo "--- Python Environment ---"
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/venv"
else
    echo "Virtual environment already exists."
fi

source "$PROJECT_ROOT/venv/bin/activate"
echo "Python: $(python --version)"

echo "Installing Python dependencies..."
pip install --upgrade pip -q
pip install -e "$PROJECT_ROOT[dev,web]" -q
echo "Python dependencies installed."
echo ""

# --- Django Setup ---
echo "--- Django Backend ---"
cd "$PROJECT_ROOT/backend"

# Copy env file if needed
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    echo "Created .env from .env.example"
fi

echo "Running migrations..."
python manage.py makemigrations --no-input
python manage.py migrate --no-input
echo "Django setup complete."
echo ""

# --- Frontend ---
echo "--- Frontend ---"
cd "$PROJECT_ROOT/frontend"

if [ ! -f "$PROJECT_ROOT/frontend/.env" ]; then
    cp "$PROJECT_ROOT/frontend/.env.example" "$PROJECT_ROOT/frontend/.env"
    echo "Created frontend .env"
fi

echo "Installing npm dependencies..."
npm install --silent 2>/dev/null
echo "Frontend setup complete."
echo ""

# --- Verify ---
echo "--- Verification ---"
cd "$PROJECT_ROOT"
echo "Checking Django..."
cd backend && python manage.py check && cd ..
echo "Django: OK"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Quick start:"
echo "  source venv/bin/activate"
echo "  make backend          # Start Django on :8000"
echo "  make frontend         # Start Vite on :5173"
echo ""
echo "Training pipeline:"
echo "  make migrate-legacy   # Import legacy training data"
echo "  make prepare-data     # Verify and split dataset"
echo "  make train            # Train YOLO11 model"
echo ""
