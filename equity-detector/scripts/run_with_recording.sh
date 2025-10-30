#!/bin/bash

# BERT Evaluation Pipeline with asciinema Recording
# ==================================================
# Records the complete evaluation process for documentation

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RECORDING_DIR="$PROJECT_DIR/recordings"

# Create recordings directory
mkdir -p "$RECORDING_DIR"

echo "========================================================================"
echo "BERT TOXICITY MODEL EVALUATION - RECORDED SESSION"
echo "========================================================================"
echo ""
echo "This session will be recorded using asciinema for documentation."
echo "Recording will be saved to: $RECORDING_DIR/"
echo ""

# Set PYTHONPATH to include venv packages
export PYTHONPATH=/Users/seb/Desktop/EquityLens/equity-detector/venv/lib/python3.13/site-packages

cd "$PROJECT_DIR"

# Function to run and record a phase
run_phase() {
    local phase_name="$1"
    local command="$2"
    local recording_file="$RECORDING_DIR/${phase_name}.cast"
    
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Recording Phase: $phase_name"
    echo "------------------------------------------------------------------------"
    echo ""
    
    # Run with asciinema recording
    asciinema rec \
        --overwrite \
        --title "BERT Evaluation - $phase_name" \
        --command "$command" \
        "$recording_file"
    
    echo ""
    echo "✓ Phase complete. Recording saved to: $recording_file"
    echo ""
}

# Check dependencies first
echo "Checking dependencies..."
python3 scripts/check_and_install_deps.py
if [ $? -ne 0 ]; then
    echo "❌ Dependencies check failed. Please install required packages."
    exit 1
fi

echo ""
echo "All dependencies satisfied. Starting evaluation pipeline..."
echo ""
echo "Note: The evaluation will run in multiple phases:"
echo "  Phase 1: Data preparation and baseline inference (~5-10 min)"
echo "  Phase 2: Calibration methods (~5-10 min)"
echo "  Phase 3: Tokenizer analysis and documentation (~2-5 min)"
echo ""
echo "Press ENTER to begin recording..."
read

# Run the complete pipeline with recording
asciinema rec \
    --overwrite \
    --title "BERT Toxicity Evaluation - Complete Pipeline" \
    --idle-time-limit 2 \
    --command "python3 scripts/run_complete_evaluation.py" \
    "$RECORDING_DIR/complete_evaluation.cast"

echo ""
echo "========================================================================"
echo "RECORDING COMPLETE"
echo "========================================================================"
echo ""
echo "Recordings saved to: $RECORDING_DIR/"
echo ""
echo "To replay the recording:"
echo "  asciinema play $RECORDING_DIR/complete_evaluation.cast"
echo ""
echo "To upload and share (optional):"
echo "  asciinema upload $RECORDING_DIR/complete_evaluation.cast"
echo ""
echo "To convert to GIF (requires agg):"
echo "  pip install agg"
echo "  agg $RECORDING_DIR/complete_evaluation.cast $RECORDING_DIR/evaluation.gif"
echo ""

