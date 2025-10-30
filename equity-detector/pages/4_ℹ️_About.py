"""
Equity Ally - About
Project information and mission
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.shared import page_config, apply_custom_css, load_logo, page_navigation, navigation_footer

# Page configuration
page_config("Equity Ally - About", "ℹ️", "wide")

# Apply custom CSS
apply_custom_css()

# === HEADER ===
load_logo("assets/equitylogolong.svg", max_width="500px")

st.markdown("""
<div style='text-align: center; padding: 1rem 0 2rem 0;'>
    <h2 style='font-size: 2.5rem; color: #e8eaed; margin-bottom: 1rem;'>ℹ️ About Equity Ally</h2>
    <p style='font-size: 1.15rem; color: #c1c7d0; max-width: 800px; margin: 0 auto; line-height: 1.7;'>
        Democratizing AI-powered content safety for everyone
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# === MISSION ===
st.markdown("## 🎯 Our Mission")

st.markdown("""
**Democratize AI-powered content safety, making world-class protection accessible to everyone—regardless of budget, technical expertise, or resources.**

Every community deserves tools to create safe, inclusive digital spaces where young people can learn, connect, and thrive without fear of harassment or bullying.
""")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("### The Problem")

st.markdown("""
• **59%** of U.S. teens have experienced cyberbullying

• Victims are **2-9× more likely** to consider suicide

• Only **1 in 10** teens reports it to adults

• Small organizations lack access to enterprise AI safety tools

• Commercial solutions are expensive and compromise privacy
""")

st.markdown("### Core Values")

st.markdown("""
🌍 **Accessibility:** Free and open-source

🔒 **Privacy:** Data stays on your device

🎯 **Excellence:** 96.8% accuracy with Isotonic calibration, comprehensive evaluation

🤝 **Education:** Teach better communication

⚖️ **Transparency:** Open methodology with 44+ visualizations
""")

# === AI ASSISTANT OVERVIEW ===
st.markdown("## 🤖 AI Assistant")

st.markdown("""
Equity Ally democratizes AI-powered content safety, bringing open-sourced, enterprise-grade moderation to schools, nonprofits, and small platforms for free. The app is able to run entirely on-device by using a compact 418MB fine-tuned BERT model which is able to achieve 96.8% accuracy on validation. The BERT model allows for instant, offline detection without the need for cloud services or paid APIs.

In Single Text mode, Equity Ally analyzes messages in under 100 ms on CPU, returning a calibrated risk score, severity label, and threshold context. Monte Carlo Dropout performs multiple inference passes to gauge uncertainty on borderline cases, helping moderators focus where it matters most. Pytesseract OCR-based image analysis also detects harmful text within screenshots or photos.

For larger organizations, Bulk Analysis supports CSV and Excel uploads, analyzing thousands of datapoints, then detailed visualizations, sortable tables, and downloadable reports that summarize trends and severity levels. The Performance Dashboard includes 44 transparent visualizations demonstrating model behavior and architecture, while the Learn More section offers explanations, workshop templates, safety-education tools, and relevant resources to promote digital citizenship.

Open-source and optimized for universal use on nearly any device, Equity Ally combines speed, transparency, and education. It empowers students, educators, and communities to understand, prevent, and discuss online harm, enabling advanced AI moderation to be free, open-sourced, educational, and effective for everyone.
""")

st.markdown("---")

# === TECHNICAL DETAILS ===
st.markdown("## 🔬 Technical Excellence")

st.markdown("""
<div class='custom-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-color: #8b5cf6;'>
    <h4 style='color: #ffffff; margin-bottom: 1rem;'>Comprehensive Model Evaluation (October 2025)</h4>
    <div style='color: #e0e7ff; line-height: 1.8;'>
        <strong style='color: #ffffff;'>Complete Development Pipeline:</strong>
        <ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>
            <li><strong>Fine-Tuned:</strong> 120,000+ samples from 5 toxicity datasets</li>
            <li><strong>96.8% Accuracy:</strong> Validation set with Isotonic calibration</li>
            <li><strong>91.5% Test Accuracy:</strong> Strong generalization to unseen data</li>
            <li><strong>96.8% ROC-AUC:</strong> Near state-of-the-art discrimination ability</li>
            <li><strong>3 Calibration Methods:</strong> Isotonic (best), Temperature, Platt evaluated</li>
            <li><strong>44 Visualizations:</strong> ROC curves, reliability diagrams, confusion matrices</li>
            <li><strong>Complete Documentation:</strong> Model card, calibration report, training details</li>
        </ul>
        <p style='margin-top: 1rem; font-weight: 600;'>
            📁 See <code style='background: rgba(255,255,255,0.2); padding: 0.2rem 0.5rem; border-radius: 4px;'>docs/onepager.md</code> for executive summary
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# === THANK YOU ===
st.markdown("""
## 🙏 Thank You for Using Equity Ally

Together, we can build safer, more inclusive online communities where everyone can thrive.

**Let's make the internet a better place, one message at a time.**
""")

# Page navigation
page_navigation()

# Navigation footer
navigation_footer()

