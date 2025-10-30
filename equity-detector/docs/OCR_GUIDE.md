# OCR Setup Guide for Image Input Feature

## Overview
The Equity Ally app supports analyzing text from images using Optical Character Recognition (OCR). This guide will help you set up the required dependencies and use the feature effectively.

---

## Prerequisites

### System Requirements
The image input feature requires **Tesseract OCR** to be installed on your system. Pytesseract is a Python wrapper for Tesseract OCR Engine.

---

## Installation Instructions

### macOS
```bash
# Install Tesseract using Homebrew
brew install tesseract

# Verify installation
tesseract --version
```

If you don't have Homebrew installed:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Ubuntu/Debian Linux
```bash
# Update package list
sudo apt update

# Install Tesseract
sudo apt install tesseract-ocr

# Verify installation
tesseract --version
```

### Windows
1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer
3. Add Tesseract to your system PATH:
   - Default installation path: `C:\Program Files\Tesseract-OCR`
   - Add this path to your system's PATH environment variable
4. Verify installation by opening Command Prompt and running:
   ```cmd
   tesseract --version
   ```

### Custom Installation Path
If Tesseract is installed in a non-standard location, you may need to specify the path:
```python
import pytesseract

# Windows example
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# macOS/Linux example
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
```

---

## Python Package Installation

After installing Tesseract OCR on your system:

```bash
# Activate your virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install Python requirements
pip install -r requirements.txt
```

---

## Using the Image Input Feature

1. **Launch the app:**
   ```bash
   streamlit run Home.py
   ```

2. **Navigate to the Detector page:**
   - Click on **"🔍 Detector"** in the sidebar
   - Switch to the **"🖼️ Image Input (OCR)"** tab

3. **Upload an image:**
   - Supported formats: PNG, JPG, JPEG, BMP, TIFF
   - The image should contain clear, readable text

4. **Review extracted text:**
   - The app will automatically extract text from the image
   - Extracted text will be displayed in a text area
   - You can review and verify the extracted text before analysis

5. **Analyze:**
   - Click the **"Analyze Text"** button
   - The extracted text will be analyzed for bullying/toxicity
   - Results will be displayed with the same metrics as text input

---

## Supported Image Types

- Screenshots from social media
- Photos of text messages
- Scanned documents
- Images with text overlays
- Memes with text
- Chat conversations
- Email content

---

## Tips for Best Results

### Image Quality
- Use high-resolution images
- Ensure good contrast between text and background
- Avoid blurry or distorted images
- Keep image files reasonably sized (< 10 MB recommended)

### Text Clarity
- Clear, printed text works best
- Handwritten text may have lower accuracy
- Make sure text is horizontal (not rotated)
- Avoid images with multiple text orientations

### Language Support
- By default, Tesseract is configured for English
- Additional language packs can be installed if needed

---

## Troubleshooting

### "tesseract is not installed" error
- Verify Tesseract is installed: `tesseract --version`
- Ensure Tesseract is in your system PATH
- Try restarting your terminal/IDE after installation

### "No text could be extracted" warning
- Check image quality and text clarity
- Ensure the image actually contains text
- Try preprocessing the image (increase contrast, reduce noise)

### Poor extraction quality
- Improve image resolution
- Increase contrast between text and background
- Use image preprocessing tools before upload

---

## Advanced Configuration

### Custom Tesseract Configuration
You can customize OCR behavior by modifying the `extract_text_from_image()` function:

```python
# Example: Use specific PSM mode
extracted_text = pytesseract.image_to_string(image, config='--psm 6')
```

**PSM (Page Segmentation Modes):**
- `--psm 6`: Assume a single uniform block of text (default)
- `--psm 3`: Fully automatic page segmentation
- `--psm 11`: Sparse text. Find as much text as possible

### Multi-Language Support
```python
# For multiple languages
extracted_text = pytesseract.image_to_string(image, lang='eng+spa')
```

---

## Example Use Cases

### Screenshot Analysis
- Analyze chat conversations from screenshots
- Review social media posts captured as images
- Check email content from image attachments

### Document Scanning
- Scan printed documents for harmful content
- Process scanned letters or notes
- Analyze text from photos

### Social Media Monitoring
- Check image posts with text overlays
- Analyze memes with text content
- Review stories and status updates

---

## Performance Notes

- OCR processing typically takes 1-5 seconds depending on image size and complexity
- The bullying detection analysis runs immediately after text extraction
- Monte Carlo Dropout can be enabled for more robust predictions on extracted text

---

## Additional Resources

- **Tesseract Documentation**: https://tesseract-ocr.github.io/
- **Pytesseract GitHub**: https://github.com/madmaze/pytesseract
- **Image preprocessing tips**: https://tesseract-ocr.github.io/tessapi/

---

**Need more help?** See the main documentation at `docs/README.md`

**Last Updated:** October 24, 2025

