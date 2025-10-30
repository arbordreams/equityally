# Assets Directory

This directory contains visual assets and branding materials for EquityAlly.

---

## Files

### `equitylogo.svg`
Square logo for EquityAlly.

**Specifications:**
- Format: SVG (scalable vector)
- Dimensions: Square aspect ratio
- Size: ~27KB
- Colors: Primary brand colors

**Usage:**
- Favicon
- App icons
- Social media profile images
- Square thumbnails

---

### `equitylogolong.svg`
Horizontal logo for EquityAlly.

**Specifications:**
- Format: SVG (scalable vector)
- Dimensions: Wide/horizontal aspect ratio
- Size: ~30KB
- Colors: Primary brand colors

**Usage:**
- Website header
- Streamlit app pages
- Documentation headers
- Video overlays
- Presentation slides

**Implementation:**
```python
from utils.shared import load_logo

# Default usage (horizontal logo)
load_logo()

# Custom width
load_logo("assets/equitylogolong.svg", max_width="600px")

# Square logo
load_logo("assets/equitylogo.svg", max_width="200px")
```

---

## Brand Guidelines

### Colors
The logos use the EquityAlly brand color palette:
- Primary: Deep blue (#1E3A8A or similar)
- Accent: Teal/cyan highlights
- Text: Dark gray/black

### Typography
Logo includes the "EquityAlly" wordmark with:
- Modern sans-serif typeface
- Clean, professional appearance
- Tech-forward aesthetic

### Usage Guidelines
- Maintain aspect ratio (don't stretch or distort)
- Use on light backgrounds primarily
- Ensure sufficient contrast for readability
- Don't modify colors without brand approval

---

## File Format

### Why SVG?
- **Scalable**: Looks crisp at any size
- **Small**: 27-30KB vs hundreds of KB for PNG
- **Editable**: Can be modified with vector tools
- **Web-friendly**: Native browser support

### Converting to Other Formats
If you need PNG or other formats:

```bash
# Using Inkscape (command line)
inkscape equitylogo.svg --export-png=equitylogo.png --export-width=512

# Using ImageMagick
convert equitylogo.svg -resize 512x512 equitylogo.png
```

---

## Adding New Assets

When adding new visual assets:

1. **Save to this directory**: `equity-detector/assets/`
2. **Use descriptive names**: `feature-screenshot.png`, `demo-flow.gif`
3. **Optimize file sizes**: Compress images, use appropriate formats
4. **Update this README**: Document new assets

### Recommended Formats
- **Logos**: SVG
- **Screenshots**: PNG (lossless)
- **Photos**: JPG (compressed)
- **Animations**: GIF or MP4
- **Icons**: SVG or PNG

---

## Usage in Code

The `load_logo()` function in `utils/shared.py` handles logo display:

```python
def load_logo(logo_path="assets/equitylogolong.svg", max_width="500px"):
    """
    Display the Equity Ally logo
    
    Args:
        logo_path: Path to logo file (relative to equity-detector directory)
        max_width: Maximum width of the logo
    """
    # Implementation in utils/shared.py
```

**Called in:**
- `Home.py` - Homepage header
- `pages/1_🔍_Detector.py` - Detector page
- `pages/2_📊_Performance.py` - Performance page
- `pages/3_📚_Learn_More.py` - Learn More page
- `pages/4_ℹ️_About.py` - About page

---

## Related Files

### Recordings
Demo videos and screencasts are stored separately in:
- `equity-detector/recordings/`

### Visualizations
Performance charts and graphs are in:
- `equity-detector/visualizations/`

---

## License

Logo and branding assets are:
- **Copyright**: EquityAlly Development Team
- **Usage**: Internal project use
- **Distribution**: Open source project assets

If you fork or adapt this project, please create your own branding.

---

**Last Updated**: October 23, 2024

