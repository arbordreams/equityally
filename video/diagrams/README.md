# Equity Ally Mermaid Diagrams

Professional diagrams for video production and documentation. All files are pure `.mmd` format - ready to copy/paste into mermaid.live or use with mermaid-cli.

## 📁 Diagram Files

### Core Pipeline Diagrams

| File | Description | Best For | Complexity |
|------|-------------|----------|------------|
| `02_simple_pipeline.mmd` | 5 datasets → BERT → calibration | **Video narration** ⭐ | Simple |
| `03_three_stages.mmd` | Data → Fine-tuning → Calibration | **Quick overview** | Simple |
| `04_accuracy_progression.mmd` | Before/after accuracy boost | **Impact visualization** | Simple |
| `08_complete_pipeline_detailed.mmd` | Full 5-stage development pipeline | Technical docs | Detailed |

### Dataset Diagrams

| File | Description | Best For | Complexity |
|------|-------------|----------|------------|
| `01_five_datasets.mmd` | All 5 datasets with details | Dataset explanation | Medium |
| `09_dataset_comparison.mmd` | Detailed dataset breakdown | Technical presentation | Detailed |

### Performance & Metrics

| File | Description | Best For | Complexity |
|------|-------------|----------|------------|
| `10_performance_metrics.mmd` | Before/after calibration metrics | Results showcase | Detailed |
| `15_calibration_methods.mmd` | Comparison of 3 calibration methods | Technical deep-dive | Detailed |

### Architecture & Technical

| File | Description | Best For | Complexity |
|------|-------------|----------|------------|
| `12_technical_architecture.mmd` | Complete system architecture | Developer docs | Detailed |
| `06_deployment_architecture.mmd` | User flow & processing | How it works | Medium |

### Impact & Business

| File | Description | Best For | Complexity |
|------|-------------|----------|------------|
| `11_before_after_comparison.mmd` | Problem vs. Solution | Pitch deck | Medium |
| `14_impact_showcase.mmd` | Mind map of all benefits | Marketing material | Medium |
| `16_use_cases.mmd` | Real-world applications | Business presentation | Detailed |

### Other

| File | Description | Best For | Complexity |
|------|-------------|----------|------------|
| `05_platform_coverage.mmd` | Platform diversity | Quick visual | Simple |
| `07_secret_sauce.mmd` | Success factors mind map | Explainer video | Simple |
| `13_development_timeline.mmd` | Gantt chart timeline | Project history | Medium |

## 🎬 Recommended for Video

For your 3-minute app demo narration:

1. **`02_simple_pipeline.mmd`** - When explaining the training process ⭐
2. **`04_accuracy_progression.mmd`** - When showing the 90% → 96.8% boost ⭐
3. **`03_three_stages.mmd`** - Alternative simple pipeline view
4. **`11_before_after_comparison.mmd`** - When explaining why Equity Ally exists

## 🚀 How to Use

### Option 1: mermaid.live (Easiest)

1. Go to https://mermaid.live
2. Open any `.mmd` file
3. Copy **entire file contents**
4. Paste into mermaid.live
5. Export as PNG or SVG

### Option 2: Command Line (Batch Export)

```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Export single diagram as PNG
mmdc -i 02_simple_pipeline.mmd -o simple_pipeline.png

# Export as SVG (scalable)
mmdc -i 02_simple_pipeline.mmd -o simple_pipeline.svg

# Batch export all diagrams
for file in *.mmd; do
  mmdc -i "$file" -o "${file%.mmd}.png"
done
```

### Option 3: VS Code Extension

1. Install "Markdown Preview Mermaid Support"
2. Create a markdown file with:
   ```markdown
   # My Diagram
   ```mermaid
   [paste .mmd contents here]
   ```
   ```
3. Open preview (⌘⇧V on Mac, Ctrl+Shift+V on Windows)

## 📊 Diagram Complexity Guide

- **Simple** (1-2 min to render): Great for videos, quick to understand
- **Medium** (2-5 min to render): Good balance of detail and clarity  
- **Detailed** (5+ min to render): Comprehensive, for documentation

## 🎨 Color Scheme

All diagrams use consistent colors:

- 🔵 Blue (`#e3f2fd`) - Data/Input stages
- 🟠 Orange (`#fff3e0`) - Processing/Preparation
- 🔴 Pink (`#fce4ec`) - Training/Fine-tuning
- 🟣 Purple (`#f3e5f5`) - Calibration
- 🟢 Green (`#c8e6c9`) - Results/Success
- 🟡 Yellow (`#fff9c4`) - Important highlights

## 💡 Pro Tips

1. **For videos**: Use simple diagrams (02, 03, 04) - they animate better
2. **For docs**: Use detailed diagrams (08, 09, 10, 12, 15)
3. **Export as SVG** if you need to scale or edit later
4. **Export as PNG** for direct video insertion (1920x1080 recommended)
5. **Test on mermaid.live first** before batch exporting

## 🔧 Troubleshooting

**Diagram won't render?**
- Make sure you copied the ENTIRE file
- Check for any special characters
- Try on mermaid.live first

**Text too small?**
- Most diagrams have `fontSize` config at the top
- Edit the `%%{init: ...}%%` line to increase font size

**Need different colors?**
- Edit the `style` lines at the bottom of each `.mmd` file
- Use hex colors like `fill:#YOUR_COLOR`

## 📝 Customization

All diagrams are fully editable. To customize:

1. Open the `.mmd` file in any text editor
2. Edit text, colors, or structure
3. Save and re-render

Example color change:
```mermaid
style NODE fill:#NEW_COLOR,stroke:#BORDER_COLOR,stroke-width:3px
```

---

**Last Updated**: October 25, 2024  
**Total Diagrams**: 16  
**Format**: Mermaid (.mmd)  
**Compatible with**: mermaid.live, mermaid-cli, GitHub, GitLab, Notion, Obsidian
