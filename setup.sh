#!/bin/bash
# Quick Start Setup Script for GeoRAG Explorer

set -e

echo "=================================="
echo "GeoRAG Explorer - Quick Start Setup"
echo "=================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env from template
echo ""
echo "Setting up environment configuration..."
if [ -f ".env" ]; then
    echo ".env file already exists. Skipping."
else
    cp .env.example .env
    echo ".env file created from template."
    echo "⚠️  IMPORTANT: Edit .env and add your OPENAI_API_KEY"
fi

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/reports
mkdir -p data/maps/national
mkdir -p data/maps/state
mkdir -p data/maps/geochemical
mkdir -p data/maps/corridors
mkdir -p data/maps/schist_belts
mkdir -p data/maps/geological
mkdir -p artifacts

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OPENAI_API_KEY:"
echo "   nano .env"
echo ""
echo "2. Place geological reports in data/reports/ as .txt files"
echo ""
echo "3. Start Jupyter and run notebooks:"
echo "   jupyter notebook notebooks/"
echo ""
echo "4. Run tests to verify installation:"
echo "   pytest tests/"
echo ""
echo "5. Follow these notebooks in order:"
echo "   - 01_data_exploration.ipynb"
echo "   - 02_document_processing.ipynb"
echo "   - 03_retrieval_evaluation.ipynb"
echo "   - 04_rag_evaluation.ipynb"
echo ""
