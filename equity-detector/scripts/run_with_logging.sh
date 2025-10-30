#!/bin/bash

# BERT Evaluation Pipeline with Comprehensive Logging
# ====================================================
# Runs the complete evaluation process with detailed logging

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/evaluation_${TIMESTAMP}.log"

# Create logs directory
mkdir -p "$LOG_DIR"

# Set PYTHONPATH to include venv packages
export PYTHONPATH=/Users/seb/Desktop/EquityLens/equity-detector/venv/lib/python3.13/site-packages

cd "$PROJECT_DIR"

echo "========================================================================"
echo "BERT TOXICITY MODEL EVALUATION - WITH LOGGING"
echo "========================================================================"
echo ""
echo "Log file: $LOG_FILE"
echo ""
echo "Starting evaluation pipeline..."
echo ""

# Run with comprehensive logging
{
    echo "========================================================================"
    echo "BERT TOXICITY MODEL EVALUATION LOG"
    echo "========================================================================"
    echo "Start Time: $(date)"
    echo "Working Directory: $PROJECT_DIR"
    echo "Python Path: $PYTHONPATH"
    echo ""
    
    # Run the evaluation
    python3 scripts/run_complete_evaluation.py 2>&1
    
    EXIT_CODE=$?
    
    echo ""
    echo "========================================================================"
    echo "EVALUATION COMPLETE"
    echo "========================================================================"
    echo "End Time: $(date)"
    echo "Exit Code: $EXIT_CODE"
    
    exit $EXIT_CODE
    
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to: $LOG_FILE"
echo ""

