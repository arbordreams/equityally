# Terminal Recording with asciinema

This guide explains how to record, replay, and share terminal sessions of the BERT evaluation pipeline using asciinema.

## What is asciinema?

asciinema is a free and open-source terminal session recorder. It records your terminal in a lightweight text-based format that can be:
- Replayed in your terminal
- Shared via asciinema.org
- Embedded in documentation
- Converted to GIF/video

## Installation

asciinema has been installed via Homebrew:

```bash
brew install asciinema
```

Verify installation:

```bash
asciinema --version
# Should show: asciinema 3.0.0
```

## Recording the Evaluation

### Option 1: Automated Recording (Recommended)

Use the provided script:

```bash
bash scripts/run_with_recording.sh
```

This will:
1. Check dependencies
2. Prompt you to start recording
3. Run the complete evaluation
4. Save recording to `recordings/complete_evaluation.cast`

### Option 2: Manual Recording

```bash
# Start recording
asciinema rec --title "BERT Evaluation" recordings/my_recording.cast

# Run evaluation
PYTHONPATH=venv/lib/python3.13/site-packages python3 scripts/run_complete_evaluation.py

# Stop recording (Ctrl+D or exit)
exit
```

## Recording Options

### Basic Recording

```bash
asciinema rec output.cast
```

### With Title

```bash
asciinema rec --title "BERT Toxicity Evaluation" output.cast
```

### With Idle Time Limit (Skip Long Pauses)

```bash
asciinema rec --idle-time-limit 2 output.cast
```

This speeds up playback by skipping pauses longer than 2 seconds.

### Overwrite Existing Recording

```bash
asciinema rec --overwrite output.cast
```

### Append to Existing Recording

```bash
asciinema rec --append output.cast
```

## Replaying Recordings

### In Your Terminal

```bash
asciinema play recordings/complete_evaluation.cast
```

### Playback Controls

- **Space**: Pause/Resume
- **.**: Step forward (when paused)
- **Ctrl+C**: Quit

### At Specific Speed

```bash
# 2x speed
asciinema play --speed 2 recording.cast

# 0.5x speed (slow motion)
asciinema play --speed 0.5 recording.cast
```

### With Idle Time Limit

```bash
# Skip pauses > 1 second during playback
asciinema play --idle-time-limit 1 recording.cast
```

## Sharing Recordings

### Upload to asciinema.org

```bash
asciinema upload recordings/complete_evaluation.cast
```

This will:
1. Upload your recording
2. Provide a shareable URL (e.g., https://asciinema.org/a/XyZ123)
3. Allow embedding in websites

### Share Via File

Simply share the `.cast` file:

```bash
# Copy to shared drive
cp recordings/complete_evaluation.cast /path/to/shared/location/

# Or send via email, Slack, etc.
```

Recipients can replay with:

```bash
asciinema play complete_evaluation.cast
```

## Converting to GIF

For static documentation (like README.md), convert to GIF:

### Install agg

```bash
# Via Cargo (Rust package manager)
cargo install --git https://github.com/asciinema/agg

# Or download binary from:
# https://github.com/asciinema/agg/releases
```

### Convert

```bash
agg recordings/complete_evaluation.cast recordings/evaluation.gif
```

### Options

```bash
# Custom size
agg --cols 120 --rows 30 input.cast output.gif

# Custom speed
agg --speed 2 input.cast output.gif

# Custom theme
agg --theme monokai input.cast output.gif
```

## Converting to SVG

For web embedding:

```bash
# Using svg-term-cli
npm install -g svg-term-cli

svg-term --in recordings/complete_evaluation.cast --out evaluation.svg
```

## Best Practices for Recording Evaluations

### 1. Clean Terminal Before Recording

```bash
clear
```

### 2. Set Terminal Size

```bash
# Standard size for good readability
# Terminal: 120 columns × 40 rows
```

### 3. Add Introductory Comment

```bash
# At start of recording
echo "BERT Toxicity Model - Complete Evaluation Pipeline"
echo "This recording demonstrates the full calibration workflow"
echo ""
```

### 4. Use Idle Time Limit

```bash
asciinema rec --idle-time-limit 2 output.cast
```

Prevents long pauses from making recording boring.

### 5. Add Annotations

While recording, add comments:

```bash
echo "# Phase 1: Data Loading..."
# Run phase 1 commands

echo "# Phase 2: Model Inference..."
# Run phase 2 commands
```

## Example Recordings Structure

```
recordings/
├── complete_evaluation.cast      # Full pipeline (3-4 hours)
├── quick_demo.cast               # Demo with sample data (5 min)
├── phase1_data.cast              # Just data loading
├── phase5_calibration.cast       # Just calibration
├── results_walkthrough.cast      # Reviewing outputs
└── README.md                     # Index of recordings
```

## Recording Different Phases

### Phase 1-2: Setup

```bash
asciinema rec --title "BERT Eval: Setup & Data" recordings/phase1_setup.cast
python3 scripts/check_and_install_deps.py
# ... show data loading
exit
```

### Phase 3-5: Core Evaluation

```bash
asciinema rec --idle-time-limit 3 --title "BERT Eval: Inference & Calibration" \
    recordings/phase3_inference.cast
# ... run evaluation
exit
```

### Phase 10: Results Review

```bash
asciinema rec --title "BERT Eval: Results Walkthrough" recordings/results.cast
cat docs/onepager.md
ls -lh visualizations/
cat evaluation/metrics_summary.csv | head -20
exit
```

## Embedding in Documentation

### Markdown

```markdown
[![asciicast](https://asciinema.org/a/XyZ123.svg)](https://asciinema.org/a/XyZ123)
```

### HTML

```html
<script src="https://asciinema.org/a/XyZ123.js" id="asciicast-XyZ123" async></script>
```

### Self-Hosted Player

```html
<div id="demo"></div>
<script src="asciinema-player.min.js"></script>
<link rel="stylesheet" href="asciinema-player.css">
<script>
  AsciinemaPlayer.create('recording.cast', document.getElementById('demo'));
</script>
```

## Managing Recordings

### List Recordings

```bash
ls -lh recordings/
```

### Get Recording Info

```bash
# First few lines contain metadata
head -5 recordings/complete_evaluation.cast
```

### File Sizes

Recordings are very compact:

- 1 hour session: ~500KB - 2MB
- Text-based (not video)
- Highly compressible

### Cleanup Old Recordings

```bash
# Remove recordings older than 30 days
find recordings/ -name "*.cast" -mtime +30 -delete
```

## Advanced Tips

### Record Specific Window Size

```bash
# Set cols/rows explicitly
asciinema rec --cols 120 --rows 40 output.cast
```

### Add Environment Information

```bash
# At recording start
echo "System: $(uname -a)"
echo "Python: $(python3 --version)"
echo "Date: $(date)"
```

### Create Chapters (Manual)

Add clear section markers:

```bash
echo ""
echo "=========================================="
echo "CHAPTER 1: DATA PREPARATION"
echo "=========================================="
# ... commands

echo ""
echo "=========================================="
echo "CHAPTER 2: MODEL INFERENCE"
echo "=========================================="
# ... commands
```

### Pause Recording

You can't pause/resume, but you can:

1. Stop recording (Ctrl+D)
2. Start new recording
3. Concatenate later (manually edit .cast files)

## Troubleshooting

### Recording Not Found

```bash
# Check path
ls -la recordings/

# Ensure directory exists
mkdir -p recordings/
```

### Playback Glitches

```bash
# Try with idle time limit
asciinema play --idle-time-limit 2 recording.cast
```

### Upload Fails

```bash
# Check internet connection
ping asciinema.org

# Try uploading again
asciinema upload --force recording.cast
```

### Large File Size

```bash
# Check file size
ls -lh recording.cast

# Reduce idle time when recording
asciinema rec --idle-time-limit 1 recording.cast

# Or apply during playback
asciinema play --idle-time-limit 1 recording.cast
```

## Example: Complete Workflow

```bash
# 1. Prepare environment
cd /Users/seb/Desktop/EquityLens/equity-detector
clear

# 2. Start recording
asciinema rec \
    --title "BERT Toxicity Evaluation - Complete Pipeline" \
    --idle-time-limit 2 \
    recordings/evaluation_$(date +%Y%m%d_%H%M%S).cast

# 3. Show system info
echo "BERT Toxicity Model - Evaluation Pipeline"
echo "Date: $(date)"
echo "Python: $(python3 --version)"
echo ""

# 4. Check dependencies
python3 scripts/check_and_install_deps.py

# 5. Run evaluation
PYTHONPATH=venv/lib/python3.13/site-packages \
    python3 scripts/run_complete_evaluation.py

# 6. Show results
echo ""
echo "Evaluation complete! Results:"
ls -lh visualizations/ | head -10
cat docs/onepager.md | head -50

# 7. Stop recording
exit
```

## Resources

- **asciinema Docs**: https://docs.asciinema.org/
- **asciinema.org**: https://asciinema.org/
- **agg (GIF generator)**: https://github.com/asciinema/agg
- **svg-term**: https://github.com/marionebl/svg-term-cli
- **Player API**: https://github.com/asciinema/asciinema-player

## Quick Reference

| Task | Command |
|------|---------|
| Record | `asciinema rec output.cast` |
| Play | `asciinema play output.cast` |
| Upload | `asciinema upload output.cast` |
| Convert to GIF | `agg input.cast output.gif` |
| Set speed | `--speed 2` |
| Skip pauses | `--idle-time-limit 2` |
| Overwrite | `--overwrite` |
| Set title | `--title "My Recording"` |

---

For the BERT evaluation pipeline, recordings are automatically created when using `scripts/run_with_recording.sh`.

