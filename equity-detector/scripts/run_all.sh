#!/bin/bash

# BERT Toxicity Model Evaluation Pipeline
# ========================================
# Complete evaluation with calibration, visualization, and documentation

set -e  # Exit on error

echo "========================================================================"
echo "BERT TOXICITY MODEL EVALUATION PIPELINE"
echo "========================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Project directory: $PROJECT_DIR"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  WARNING: No virtual environment detected"
    echo "   Attempting to activate venv..."
    if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
        source "$PROJECT_DIR/venv/bin/activate"
        echo "✓ Virtual environment activated"
    else
        echo "❌ ERROR: Virtual environment not found at $PROJECT_DIR/venv"
        echo "   Please create and activate a virtual environment first:"
        echo "   python -m venv venv"
        echo "   source venv/bin/activate"
        echo "   pip install -r requirements.txt"
        exit 1
    fi
fi

# Navigate to project directory
cd "$PROJECT_DIR"

echo ""
echo "------------------------------------------------------------------------"
echo "Checking dependencies..."
echo "------------------------------------------------------------------------"

# Check if required packages are installed
python -c "import torch, transformers, sklearn, numpy, pandas, matplotlib, seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Required packages not installed"
    echo "   Please install dependencies:"
    echo "   pip install -r requirements.txt"
    exit 1
fi

echo "✓ All dependencies satisfied"
echo ""

echo "------------------------------------------------------------------------"
echo "Running complete evaluation pipeline..."
echo "------------------------------------------------------------------------"
echo ""

# Run the complete evaluation
python scripts/run_complete_evaluation.py

echo ""
echo "========================================================================"
echo "PIPELINE EXECUTION COMPLETE"
echo "========================================================================"
echo ""
echo "Results:"
echo "  📊 Visualizations:  $PROJECT_DIR/visualizations/"
echo "  📄 Documentation:   $PROJECT_DIR/docs/"
echo "  📈 Metrics:         $PROJECT_DIR/evaluation/"
echo ""
echo "Quick start:"
echo "  1. Review docs/onepager.md for executive summary"
echo "  2. View visualizations/*.png for detailed analysis"
echo "  3. Use evaluation/calibration_params.json for deployment"
echo ""

