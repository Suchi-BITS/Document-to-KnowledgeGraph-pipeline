#!/bin/bash
# setup.sh - Automated setup script for Knowledge Graph Builder

set -e  # Exit on error

echo "=========================================="
echo "Knowledge Graph Builder - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Create virtual environment
echo ""
echo "2. Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "   ✓ Virtual environment created"
else
    echo "   ✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "3. Activating virtual environment..."
source .venv/bin/activate
echo "   ✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "4. Upgrading pip..."
pip install --upgrade pip --quiet
echo "   ✓ pip upgraded"

# Install dependencies
echo ""
echo "5. Installing dependencies..."
pip install -r requirements.txt --quiet
echo "   ✓ Dependencies installed"

# Create necessary directories
echo ""
echo "6. Creating directory structure..."
mkdir -p data/input
mkdir -p data/output
mkdir -p notebooks
mkdir -p tests
mkdir -p logs
echo "   ✓ Directories created"

# Create .gitkeep files
touch data/input/.gitkeep
touch data/output/.gitkeep

# Setup environment file
echo ""
echo "7. Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "   ✓ .env file created from template"
    echo ""
    echo "   ⚠️  IMPORTANT: Edit .env file with your API key!"
    echo "      Run: nano .env (or use your preferred editor)"
else
    echo "   ✓ .env file already exists"
fi

# Create sample input file
echo ""
echo "8. Creating sample input file..."
if [ ! -f "data/input/sample.txt" ]; then
    cat > data/input/sample.txt << 'EOF'
Marie Curie, born Maria Skłodowska in Warsaw, Poland, was a pioneering physicist and chemist.
She conducted groundbreaking research on radioactivity. Together with her husband, Pierre Curie,
she discovered the elements polonium and radium. Marie Curie was the first woman to win a Nobel Prize,
the first person and only woman to win the Nobel Prize twice, and the only person to win the Nobel Prize
in two different scientific fields.
EOF
    echo "   ✓ Sample file created at data/input/sample.txt"
else
    echo "   ✓ Sample file already exists"
fi

# Run tests to verify installation
echo ""
echo "9. Running test suite to verify installation..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v --tb=short || echo "   ⚠️  Some tests failed (this is OK if you haven't configured API key)"
else
    echo "   ⚠️  pytest not found, skipping tests"
fi

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API credentials:"
echo "     nano .env"
echo ""
echo "  2. Run the example:"
echo "     python main.py data/input/sample.txt"
echo ""
echo "  3. Or start Jupyter notebook:"
echo "     jupyter notebook notebooks/demo.ipynb"
echo ""
echo "  4. Run tests:"
echo "     pytest tests/ -v"
echo ""
echo "For help:"
echo "  python main.py --help"
echo ""
echo "=========================================="