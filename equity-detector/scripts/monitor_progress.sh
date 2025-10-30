#!/bin/bash

# Progress Monitor for BERT Evaluation Pipeline
# ==============================================
# Monitors the evaluation progress in real-time

LOG_FILE=$(ls -t /Users/seb/Desktop/EquityLens/equity-detector/logs/evaluation_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No evaluation log found. Is the pipeline running?"
    exit 1
fi

echo "========================================================================"
echo "BERT EVALUATION PROGRESS MONITOR"
echo "========================================================================"
echo "Log file: $LOG_FILE"
echo ""

# Function to extract current phase
get_current_phase() {
    grep -i "PHASE [0-9]" "$LOG_FILE" | tail -1
}

# Function to get last progress update
get_last_progress() {
    tail -20 "$LOG_FILE" | grep -E "(Inference|Tokenizing|%)" | tail -3
}

# Function to count completed visualizations
count_visualizations() {
    grep -c "Saved.*visualizations" "$LOG_FILE" 2>/dev/null || echo "0"
}

# Function to get metrics
get_metrics() {
    grep -E "(F1|ROC-AUC|ECE|Brier)" "$LOG_FILE" | tail -10
}

# Monitor in real-time
while true; do
    clear
    echo "========================================================================"
    echo "BERT EVALUATION PROGRESS MONITOR"
    echo "========================================================================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Log:  $LOG_FILE"
    echo ""
    
    echo "------------------------------------------------------------------------"
    echo "CURRENT PHASE"
    echo "------------------------------------------------------------------------"
    get_current_phase
    echo ""
    
    echo "------------------------------------------------------------------------"
    echo "RECENT PROGRESS"
    echo "------------------------------------------------------------------------"
    get_last_progress
    echo ""
    
    VIZ_COUNT=$(count_visualizations)
    echo "------------------------------------------------------------------------"
    echo "COMPLETED ARTIFACTS"
    echo "------------------------------------------------------------------------"
    echo "Visualizations generated: $VIZ_COUNT"
    echo ""
    
    echo "------------------------------------------------------------------------"
    echo "RECENT METRICS"
    echo "------------------------------------------------------------------------"
    get_metrics
    echo ""
    
    echo "------------------------------------------------------------------------"
    echo "Press Ctrl+C to exit monitoring"
    echo "------------------------------------------------------------------------"
    
    sleep 5
done

