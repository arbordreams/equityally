#!/bin/bash

# Implementation Summary Display
# ==============================
# Shows all files created for the BERT evaluation pipeline

echo "========================================================================"
echo "BERT EVALUATION PIPELINE - IMPLEMENTATION SUMMARY"
echo "========================================================================"
echo ""
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

BASE_DIR="/Users/seb/Desktop/EquityLens/equity-detector"

echo "------------------------------------------------------------------------"
echo "PYTHON SCRIPTS (Core Evaluation)"
echo "------------------------------------------------------------------------"
echo ""

if [ -f "$BASE_DIR/scripts/run_evaluation_pipeline.py" ]; then
    LINES=$(wc -l < "$BASE_DIR/scripts/run_evaluation_pipeline.py")
    echo "✅ run_evaluation_pipeline.py        $(printf '%5d' $LINES) lines"
fi

if [ -f "$BASE_DIR/scripts/run_tokenizer_and_summary.py" ]; then
    LINES=$(wc -l < "$BASE_DIR/scripts/run_tokenizer_and_summary.py")
    echo "✅ run_tokenizer_and_summary.py      $(printf '%5d' $LINES) lines"
fi

if [ -f "$BASE_DIR/scripts/run_complete_evaluation.py" ]; then
    LINES=$(wc -l < "$BASE_DIR/scripts/run_complete_evaluation.py")
    echo "✅ run_complete_evaluation.py        $(printf '%5d' $LINES) lines"
fi

if [ -f "$BASE_DIR/scripts/check_and_install_deps.py" ]; then
    LINES=$(wc -l < "$BASE_DIR/scripts/check_and_install_deps.py")
    echo "✅ check_and_install_deps.py         $(printf '%5d' $LINES) lines"
fi

echo ""
echo "------------------------------------------------------------------------"
echo "SHELL SCRIPTS (Automation & Utilities)"
echo "------------------------------------------------------------------------"
echo ""

for script in run_all.sh run_with_logging.sh run_with_recording.sh monitor_progress.sh show_implementation_summary.sh; do
    if [ -f "$BASE_DIR/scripts/$script" ]; then
        LINES=$(wc -l < "$BASE_DIR/scripts/$script")
        echo "✅ $(printf '%-35s' $script) $(printf '%5d' $LINES) lines"
    fi
done

echo ""
echo "------------------------------------------------------------------------"
echo "DOCUMENTATION FILES"
echo "------------------------------------------------------------------------"
echo ""

for doc in EVALUATION_GUIDE.md EVALUATION_STATUS.md ASCIINEMA_GUIDE.md; do
    if [ -f "$BASE_DIR/$doc" ]; then
        LINES=$(wc -l < "$BASE_DIR/$doc")
        SIZE=$(ls -lh "$BASE_DIR/$doc" | awk '{print $5}')
        echo "✅ $(printf '%-35s' $doc) $(printf '%5d' $LINES) lines  $(printf '%5s' $SIZE)"
    fi
done

if [ -f "$BASE_DIR/scripts/README.md" ]; then
    LINES=$(wc -l < "$BASE_DIR/scripts/README.md")
    SIZE=$(ls -lh "$BASE_DIR/scripts/README.md" | awk '{print $5}')
    echo "✅ $(printf '%-35s' "scripts/README.md") $(printf '%5d' $LINES) lines  $(printf '%5s' $SIZE)"
fi

if [ -f "/Users/seb/Desktop/EquityLens/IMPLEMENTATION_SUMMARY_DETAILED.md" ]; then
    LINES=$(wc -l < "/Users/seb/Desktop/EquityLens/IMPLEMENTATION_SUMMARY_DETAILED.md")
    SIZE=$(ls -lh "/Users/seb/Desktop/EquityLens/IMPLEMENTATION_SUMMARY_DETAILED.md" | awk '{print $5}')
    echo "✅ $(printf '%-35s' "IMPLEMENTATION_SUMMARY_DETAILED.md") $(printf '%5d' $LINES) lines  $(printf '%5s' $SIZE)"
fi

echo ""
echo "------------------------------------------------------------------------"
echo "GENERATED DOCUMENTATION (Created During Evaluation)"
echo "------------------------------------------------------------------------"
echo ""

for doc in dataset_card.md calibration.md tokenizer_report.md model_card.md onepager.md README.md; do
    if [ -f "$BASE_DIR/docs/$doc" ]; then
        LINES=$(wc -l < "$BASE_DIR/docs/$doc")
        SIZE=$(ls -lh "$BASE_DIR/docs/$doc" | awk '{print $5}')
        echo "✅ $(printf '%-35s' "docs/$doc") $(printf '%5d' $LINES) lines  $(printf '%5s' $SIZE)"
    else
        echo "⏳ $(printf '%-35s' "docs/$doc") [Pending evaluation completion]"
    fi
done

echo ""
echo "------------------------------------------------------------------------"
echo "VISUALIZATIONS"
echo "------------------------------------------------------------------------"
echo ""

VIZ_COUNT=$(find "$BASE_DIR/visualizations" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
VIZ_SIZE=$(du -sh "$BASE_DIR/visualizations" 2>/dev/null | awk '{print $1}')

echo "Total visualizations: $VIZ_COUNT / 53 expected"
echo "Total size:          $VIZ_SIZE"
echo ""

if [ "$VIZ_COUNT" -gt 0 ]; then
    echo "Generated so far:"
    find "$BASE_DIR/visualizations" -name "*.png" 2>/dev/null | sort | while read file; do
        filename=$(basename "$file")
        filesize=$(ls -lh "$file" | awk '{print $5}')
        echo "  ✅ $(printf '%-50s' $filename) $filesize"
    done
fi

echo ""
echo "------------------------------------------------------------------------"
echo "EVALUATION ARTIFACTS"
echo "------------------------------------------------------------------------"
echo ""

EVAL_DIR="$BASE_DIR/evaluation"
if [ -d "$EVAL_DIR" ]; then
    EVAL_SIZE=$(du -sh "$EVAL_DIR" 2>/dev/null | awk '{print $1}')
    EVAL_FILES=$(find "$EVAL_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "Total artifacts:  $EVAL_FILES files"
    echo "Total size:       $EVAL_SIZE"
    echo ""
    
    if [ "$EVAL_FILES" -gt 0 ]; then
        echo "Files:"
        find "$EVAL_DIR" -type f 2>/dev/null | sort | while read file; do
            filename=$(basename "$file")
            filesize=$(ls -lh "$file" | awk '{print $5}')
            echo "  ✅ $(printf '%-40s' $filename) $filesize"
        done
    fi
else
    echo "⏳ Evaluation artifacts pending (evaluation in progress)"
fi

echo ""
echo "------------------------------------------------------------------------"
echo "LOGS"
echo "------------------------------------------------------------------------"
echo ""

LOGS_DIR="$BASE_DIR/logs"
if [ -d "$LOGS_DIR" ]; then
    LOG_COUNT=$(find "$LOGS_DIR" -name "*.log" 2>/dev/null | wc -l | tr -d ' ')
    echo "Total log files:  $LOG_COUNT"
    echo ""
    
    if [ "$LOG_COUNT" -gt 0 ]; then
        find "$LOGS_DIR" -name "*.log" 2>/dev/null | sort -r | while read file; do
            filename=$(basename "$file")
            filesize=$(ls -lh "$file" | awk '{print $5}')
            modified=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$file")
            echo "  📋 $filename  ($filesize, modified: $modified)"
        done
    fi
else
    echo "No logs directory found"
fi

echo ""
echo "------------------------------------------------------------------------"
echo "CONFIGURATION & DEPENDENCIES"
echo "------------------------------------------------------------------------"
echo ""

if [ -f "$BASE_DIR/requirements.txt" ]; then
    REQ_COUNT=$(grep -v '^#' "$BASE_DIR/requirements.txt" | grep -v '^$' | wc -l | tr -d ' ')
    echo "✅ requirements.txt              $REQ_COUNT packages"
fi

echo ""
echo "------------------------------------------------------------------------"
echo "TOTALS"
echo "------------------------------------------------------------------------"
echo ""

# Count total lines of code
PYTHON_LINES=0
SHELL_LINES=0
DOC_LINES=0

# Python scripts
for file in run_evaluation_pipeline.py run_tokenizer_and_summary.py run_complete_evaluation.py check_and_install_deps.py; do
    if [ -f "$BASE_DIR/scripts/$file" ]; then
        LINES=$(wc -l < "$BASE_DIR/scripts/$file")
        PYTHON_LINES=$((PYTHON_LINES + LINES))
    fi
done

# Shell scripts
for file in run_all.sh run_with_logging.sh run_with_recording.sh monitor_progress.sh show_implementation_summary.sh; do
    if [ -f "$BASE_DIR/scripts/$file" ]; then
        LINES=$(wc -l < "$BASE_DIR/scripts/$file")
        SHELL_LINES=$((SHELL_LINES + LINES))
    fi
done

# Documentation
for file in EVALUATION_GUIDE.md EVALUATION_STATUS.md ASCIINEMA_GUIDE.md; do
    if [ -f "$BASE_DIR/$file" ]; then
        LINES=$(wc -l < "$BASE_DIR/$file")
        DOC_LINES=$((DOC_LINES + LINES))
    fi
done

if [ -f "$BASE_DIR/scripts/README.md" ]; then
    LINES=$(wc -l < "$BASE_DIR/scripts/README.md")
    DOC_LINES=$((DOC_LINES + LINES))
fi

if [ -f "/Users/seb/Desktop/EquityLens/IMPLEMENTATION_SUMMARY_DETAILED.md" ]; then
    LINES=$(wc -l < "/Users/seb/Desktop/EquityLens/IMPLEMENTATION_SUMMARY_DETAILED.md")
    DOC_LINES=$((DOC_LINES + LINES))
fi

TOTAL_LINES=$((PYTHON_LINES + SHELL_LINES + DOC_LINES))

echo "Python Scripts:      $(printf '%6d' $PYTHON_LINES) lines"
echo "Shell Scripts:       $(printf '%6d' $SHELL_LINES) lines"
echo "Documentation:       $(printf '%6d' $DOC_LINES) lines"
echo "                     --------"
echo "Total:               $(printf '%6d' $TOTAL_LINES) lines"
echo ""

# File counts
echo "Files Created:"
echo "  Python scripts:    4"
echo "  Shell scripts:     5"
echo "  Documentation:     5 (+ 6 generated)"
echo "  Visualizations:    $VIZ_COUNT / 53 expected"
echo ""

echo "------------------------------------------------------------------------"
echo "EVALUATION STATUS"
echo "------------------------------------------------------------------------"
echo ""

# Check if evaluation is running
if pgrep -f "run_complete_evaluation.py" > /dev/null; then
    echo "Status: ⏳ EVALUATION RUNNING"
    
    # Find latest log
    LATEST_LOG=$(ls -t "$LOGS_DIR"/*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "Log:    $LATEST_LOG"
        
        # Show current phase
        CURRENT_PHASE=$(grep -i "PHASE [0-9]" "$LATEST_LOG" | tail -1)
        if [ -n "$CURRENT_PHASE" ]; then
            echo "Phase:  $CURRENT_PHASE"
        fi
        
        # Show recent progress
        RECENT_PROGRESS=$(tail -5 "$LATEST_LOG" | grep "%" | tail -1)
        if [ -n "$RECENT_PROGRESS" ]; then
            echo "Progress: $RECENT_PROGRESS"
        fi
    fi
else
    echo "Status: ⏹️  EVALUATION NOT RUNNING"
    
    # Check if completed
    LATEST_LOG=$(ls -t "$LOGS_DIR"/*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        if grep -q "EVALUATION COMPLETE" "$LATEST_LOG"; then
            echo "Result: ✅ COMPLETED"
            EXIT_CODE=$(grep "Exit Code:" "$LATEST_LOG" | tail -1 | awk '{print $NF}')
            if [ "$EXIT_CODE" = "0" ]; then
                echo "Exit:   ✅ SUCCESS (0)"
            else
                echo "Exit:   ❌ ERROR ($EXIT_CODE)"
            fi
        else
            echo "Result: ⚠️  INCOMPLETE or INTERRUPTED"
        fi
    fi
fi

echo ""
echo "------------------------------------------------------------------------"
echo "QUICK LINKS"
echo "------------------------------------------------------------------------"
echo ""

echo "View documentation:    open $BASE_DIR/EVALUATION_GUIDE.md"
echo "View status:           open $BASE_DIR/EVALUATION_STATUS.md"
echo "View implementation:   open /Users/seb/Desktop/EquityLens/IMPLEMENTATION_SUMMARY_DETAILED.md"
echo "Monitor progress:      bash $BASE_DIR/scripts/monitor_progress.sh"
echo "View latest log:       tail -f $LOGS_DIR/evaluation_*.log"
echo "List visualizations:   ls -lh $BASE_DIR/visualizations/"
echo ""

echo "========================================================================"
echo "END OF SUMMARY"
echo "========================================================================"

