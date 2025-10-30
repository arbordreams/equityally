"""
Equity Ally - Home Page
AI-Powered Content Safety Platform
"""

import streamlit as st
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.shared import page_config, apply_custom_css, load_logo, page_navigation, navigation_footer

# Page configuration
page_config("Equity Ally - Home", "🛡️", "wide")

# Apply custom CSS
apply_custom_css()

# Initialize session state
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None

# ===== HERO SECTION =====
load_logo("assets/equitylogolong.svg", max_width="600px")

st.markdown("""
<div style='text-align: center; margin-bottom: 3rem;'>
    <h1 style='font-size: 2.5rem; margin-bottom: 1rem; color: #e8eaed;'>
        Democratizing AI-Powered Content Safety
    </h1>
    <p style='font-size: 1.3rem; color: #c1c7d0; max-width: 800px; margin: 0 auto 2rem auto; line-height: 1.7;'>
        Fine-tuned BERT achieving <strong style='color: #10b981;'>96.8% ROC-AUC</strong> 
        through multi-dataset training + Isotonic calibration. Runs on <strong style='color: #84A4FC;'>any device</strong> 
        — from Chromebooks to enterprise servers. Just <strong style='color: #10b981;'>418MB</strong>, 
        completely <strong style='color: #10b981;'>free</strong> and <strong style='color: #10b981;'>open-source</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

# CTA Buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔍 Try the Detector", type="primary", use_container_width=True, key="btn_try_detector_top"):
            st.switch_page("pages/1_🔍_Detector.py")
    with col_btn2:
        if st.button("📊 View Performance", use_container_width=True, key="btn_view_performance_top"):
            st.switch_page("pages/2_📊_Performance.py")

st.markdown("---")

# ===== THE PROBLEM SECTION =====
st.markdown("<div id='the-problem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin: 3rem 0 2rem 0;'>
    <h2 style='font-size: 2.5rem; color: #e8eaed;'>The Cyberbullying Crisis</h2>
    <p style='font-size: 1.15rem; color: #c1c7d0; max-width: 700px; margin: 1rem auto;'>
        Online harassment is a growing epidemic affecting millions of young people worldwide
    </p>
</div>
""", unsafe_allow_html=True)

# Statistics in cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='custom-card' style='display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; min-height: 200px;'>
        <h3 style='font-size: 3rem; color: #c1c7d0; margin: 0 0 0.5rem 0; padding: 0; text-align: center; width: 100%;'>59%</h3>
        <p style='color: #c1c7d0; font-size: 0.95rem; line-height: 1.6; margin: 0.5rem 0; text-align: center; width: 100%;'>
            of U.S. teens have experienced cyberbullying
        </p>
        <p style='color: #6b7280; font-size: 0.8rem; margin: 0.75rem 0 1rem 0; text-align: center; width: 100%;'>
            Pew Research Center, 2022
        </p>
        <a href='https://www.pewresearch.org/internet/2022/08/10/teens-social-media-and-technology-2022/' 
           target='_blank' 
           style='color: #9ca3af; text-decoration: none; font-size: 0.9rem; margin-top: auto; transition: color 0.2s ease;'
           onmouseover="this.style.color='#c1c7d0';"
           onmouseout="this.style.color='#9ca3af';">
            View Source →
        </a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='custom-card' style='display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; min-height: 200px;'>
        <h3 style='font-size: 3rem; color: #c1c7d0; margin: 0 0 0.5rem 0; padding: 0; text-align: center; width: 100%;'>2-9×</h3>
        <p style='color: #c1c7d0; font-size: 0.95rem; line-height: 1.6; margin: 0.5rem 0; text-align: center; width: 100%;'>
            more likely to consider suicide when cyberbullied
        </p>
        <p style='color: #6b7280; font-size: 0.8rem; margin: 0.75rem 0 1rem 0; text-align: center; width: 100%;'>
            Hinduja & Patchin, 2010
        </p>
        <a href='https://www.tandfonline.com/doi/abs/10.1080/13811118.2010.494133' 
           target='_blank' 
           style='color: #9ca3af; text-decoration: none; font-size: 0.9rem; margin-top: auto; transition: color 0.2s ease;'
           onmouseover="this.style.color='#c1c7d0';"
           onmouseout="this.style.color='#9ca3af';">
            View Source →
        </a>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='custom-card' style='display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; min-height: 200px;'>
        <h3 style='font-size: 3rem; color: #c1c7d0; margin: 0 0 0.5rem 0; padding: 0; text-align: center; width: 100%;'>1/10</h3>
        <p style='color: #c1c7d0; font-size: 0.95rem; line-height: 1.6; margin: 0.5rem 0; text-align: center; width: 100%;'>
            teens report cyberbullying to parents or adults
        </p>
        <p style='color: #6b7280; font-size: 0.8rem; margin: 0.75rem 0 1rem 0; text-align: center; width: 100%;'>
            Cyberbullying Research Center
        </p>
        <a href='https://cyberbullying.org/facts' 
           target='_blank' 
           style='color: #9ca3af; text-decoration: none; font-size: 0.9rem; margin-top: auto; transition: color 0.2s ease;'
           onmouseover="this.style.color='#c1c7d0';"
           onmouseout="this.style.color='#9ca3af';">
            View Source →
        </a>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='custom-card' style='display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; min-height: 200px;'>
        <h3 style='font-size: 3rem; color: #c1c7d0; margin: 0 0 0.5rem 0; padding: 0; text-align: center; width: 100%;'>240M+</h3>
        <p style='color: #c1c7d0; font-size: 0.95rem; line-height: 1.6; margin: 0.5rem 0; text-align: center; width: 100%;'>
            flagged content violations on social platforms annually
        </p>
        <p style='color: #6b7280; font-size: 0.8rem; margin: 0.75rem 0 1rem 0; text-align: center; width: 100%;'>
            Meta Transparency Report
        </p>
        <a href='https://transparency.meta.com/reports/community-standards-enforcement/' 
           target='_blank' 
           style='color: #9ca3af; text-decoration: none; font-size: 0.9rem; margin-top: auto; transition: color 0.2s ease;'
           onmouseover="this.style.color='#c1c7d0';"
           onmouseout="this.style.color='#9ca3af';">
            View Source →
        </a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box' style='border-left-color: #dc2626; text-align: center;'>
    <strong style='font-size: 1.15rem;'>Why This Matters</strong><br/><br/>
    <span style='font-size: 1.05rem;'>
    With thousands of students in schools and youth programs worldwide needing protection,
    accessible tools are essential for creating safer online spaces. Equity Ally brings enterprise-grade 
    AI protection to schools, nonprofits, and communities—completely free and open-source.
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===== OUR SOLUTION SECTION =====
st.markdown("""
<div style='text-align: center; margin: 3rem auto 2rem auto;'>
    <h2 style='font-size: 2.5rem; color: #e8eaed; margin-bottom: 1rem;'>Efficient BERT-Based Detection</h2>
    <p style='font-size: 1.15rem; color: #c1c7d0; max-width: 800px; margin: 0 auto;'>
        Accessible AI safety powered by fine-tuned BERT — efficient, private, and built for everyone
    </p>
</div>
""", unsafe_allow_html=True)

# The Equity Ally Advantage
st.markdown("""
<div class='custom-card' style='background: linear-gradient(135deg, #1a1d23 0%, #0d1117 100%); border-color: #1463F3; text-align: center;'>
    <h3 style='color: #84A4FC; font-size: 1.5rem; margin-bottom: 1.5rem;'>
        💡 Why Equity Ally: AI Safety for Everyone
    </h3>
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem;'>
        <div style='text-align: center;'>
            <strong style='color: #e8eaed; font-size: 1.1rem;'>🚀 Small & Efficient</strong>
            <p style='color: #c1c7d0; margin-top: 0.5rem; line-height: 1.7;'>
            Only 110M parameters — highly efficient transformer architecture. 
            The zipped model is under 420MB, smaller than a single song album.
            </p>
        </div>
        <div style='text-align: center;'>
            <strong style='color: #e8eaed; font-size: 1.1rem;'>🎯 Rigorously Trained & Calibrated</strong>
            <p style='color: #c1c7d0; margin-top: 0.5rem; line-height: 1.7;'>
            <strong style='color: #10b981;'>Fine-tuned on 120,000+ samples</strong> from 5 toxicity datasets, 
            then <strong style='color: #10b981;'>Isotonic calibrated</strong> achieving 96.8% ROC-AUC and 71% ECE reduction. 
            Comprehensive multi-dataset approach ensures robust detection across platforms (Wikipedia, 
            Twitter, forums).
            </p>
        </div>
        <div style='text-align: center;'>
            <strong style='color: #e8eaed; font-size: 1.1rem;'>💻 Runs Locally</strong>
            <p style='color: #c1c7d0; margin-top: 0.5rem; line-height: 1.7;'>
            No internet required for detection. Works on consumer laptops, even Chromebooks. 
            Perfect for schools with limited bandwidth.
            </p>
        </div>
        <div style='text-align: center;'>
            <strong style='color: #e8eaed; font-size: 1.1rem;'>⚡ Fast & Free</strong>
            <p style='color: #c1c7d0; margin-top: 0.5rem; line-height: 1.7;'>
            Less than 100ms inference time on CPU. Runs completely offline. 
            Process thousands of messages per hour on basic hardware.
            </p>
        </div>
        <div style='text-align: center;'>
            <strong style='color: #e8eaed; font-size: 1.1rem;'>🔒 Privacy-First</strong>
            <p style='color: #c1c7d0; margin-top: 0.5rem; line-height: 1.7;'>
            All data stays on your device. No external servers needed. 
            Compliant with COPPA, FERPA, and student privacy laws.
            </p>
        </div>
        <div style='text-align: center;'>
            <strong style='color: #e8eaed; font-size: 1.1rem;'>🌍 Open Source</strong>
            <p style='color: #c1c7d0; margin-top: 0.5rem; line-height: 1.7;'>
            Accessible to schools, nonprofits, and communities without budgets. 
            No subscriptions, no paywalls, no vendor lock-in.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# How It Works - Research Foundation
st.markdown("""
<h3 style='color: #e8eaed; font-size: 1.8rem; margin: 2rem 0 1rem 0; text-align: center;'>
    🧠 How It Works: Two-Stage AI Pipeline
</h3>
""", unsafe_allow_html=True)

st.markdown("""
<div style='color: #c1c7d0; line-height: 1.8; font-size: 1.05rem; margin-bottom: 2rem; text-align: center; max-width: 900px; margin-left: auto; margin-right: auto;'>
Equity Ally combines state-of-the-art natural language processing with rigorous probability calibration 
to deliver <strong style='color: #10b981;'>world-class toxicity detection on any device</strong>. 
Built on peer-reviewed research and evaluated with comprehensive benchmarking.
</div>
""", unsafe_allow_html=True)

# Pipeline visualization
st.markdown("""
<div class='custom-card' style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-color: #3b82f6;'>
    <h4 style='color: #84A4FC; text-align: center; font-size: 1.4rem; margin-bottom: 1.5rem;'>
        📊 Complete Development Pipeline
    </h4>
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; color: #c1c7d0;'>
        <div style='text-align: center; padding: 1rem; background: rgba(59, 130, 246, 0.1); border-radius: 8px;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>1️⃣</div>
            <strong style='color: #84A4FC; font-size: 1.1rem;'>Multi-Dataset Training</strong>
            <p style='margin-top: 0.5rem; font-size: 0.95rem;'>
                120,000+ samples from 5 toxicity datasets (Wikipedia, Twitter, Forums, Q&A)
            </p>
        </div>
        <div style='text-align: center; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 8px;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>2️⃣</div>
            <strong style='color: #10b981; font-size: 1.1rem;'>BERT Fine-Tuning</strong>
            <p style='margin-top: 0.5rem; font-size: 0.95rem;'>
                3-4 epochs on binary toxicity classification (110M parameters)
            </p>
        </div>
        <div style='text-align: center; padding: 1rem; background: rgba(139, 92, 246, 0.1); border-radius: 8px;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>3️⃣</div>
            <strong style='color: #a78bfa; font-size: 1.1rem;'>Isotonic Calibration</strong>
            <p style='margin-top: 0.5rem; font-size: 0.95rem;'>
                Post-hoc calibration for reliable probability estimates (71% ECE reduction)
            </p>
        </div>
        <div style='text-align: center; padding: 1rem; background: rgba(245, 158, 11, 0.1); border-radius: 8px;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>4️⃣</div>
            <strong style='color: #f59e0b; font-size: 1.1rem;'>Production Deployment</strong>
            <p style='margin-top: 0.5rem; font-size: 0.95rem;'>
                Fast CPU inference (<100ms) with optional Monte Carlo uncertainty
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Core Technologies - Collapsible Details
with st.expander("🔬 Core Technologies & Research Foundation", expanded=False):
    st.markdown("""
    <div style='color: #c1c7d0; line-height: 1.8; font-size: 1rem; margin-bottom: 1.5rem;'>
    Equity Ally is built on cutting-edge research from leading AI institutions, combining transformer 
    architectures, Bayesian uncertainty quantification, and probability calibration techniques.
    </div>
    """, unsafe_allow_html=True)
    
    # BERT Architecture
    st.markdown("""
    <h4 style='color: #84A4FC; font-size: 1.2rem; margin-top: 1rem; margin-bottom: 0.75rem;'>
        1. BERT: Bidirectional Language Understanding
    </h4>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='color: #c1c7d0; line-height: 1.7; margin-bottom: 1.5rem;'>
    <strong style='color: #e8eaed;'>Research:</strong> Devlin et al. (2018) - 
    <a href='https://arxiv.org/abs/1810.04805' target='_blank' style='color: #84A4FC;'>arXiv:1810.04805</a><br/>
    <strong style='color: #e8eaed;'>Innovation:</strong> Bidirectional context understanding (reads left + right)<br/>
    <strong style='color: #e8eaed;'>Our Model:</strong> bert-base-uncased, 110M parameters, fine-tuned on 120K+ toxicity samples
    </div>
    """, unsafe_allow_html=True)
    
    # Isotonic Calibration
    st.markdown("""
    <h4 style='color: #84A4FC; font-size: 1.2rem; margin-top: 1rem; margin-bottom: 0.75rem;'>
        2. Isotonic Regression: Probability Calibration
    </h4>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='color: #c1c7d0; line-height: 1.7; margin-bottom: 1.5rem;'>
    <strong style='color: #e8eaed;'>Research:</strong> Guo et al. (2017) - 
    <a href='http://proceedings.mlr.press/v70/guo17a.html' target='_blank' style='color: #84A4FC;'>On Calibration of Modern Neural Networks</a><br/>
    <strong style='color: #e8eaed;'>Innovation:</strong> Non-parametric method for reliable probability estimates<br/>
    <strong style='color: #e8eaed;'>Our Results:</strong> <strong style='color: #10b981;'>71% ECE reduction</strong> 
    (15.23% → 4.31%) while improving F1 score by 16%
    </div>
    """, unsafe_allow_html=True)

    # Monte Carlo Dropout
    st.markdown("""
    <h4 style='color: #84A4FC; font-size: 1.2rem; margin-top: 1rem; margin-bottom: 0.75rem;'>
        3. Monte Carlo Dropout: Uncertainty Quantification
    </h4>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='color: #c1c7d0; line-height: 1.7; margin-bottom: 1.5rem;'>
    <strong style='color: #e8eaed;'>Research:</strong> Gal & Ghahramani (2016) - 
    <a href='https://proceedings.mlr.press/v48/gal16.html' target='_blank' style='color: #84A4FC;'>ICML 2016</a><br/>
    <strong style='color: #e8eaed;'>Innovation:</strong> Dropout as Bayesian approximation for confidence intervals<br/>
    <strong style='color: #e8eaed;'>Our Implementation:</strong> Optional 20-30 forward passes for uncertainty estimates (identifies borderline cases)
    </div>
    """, unsafe_allow_html=True)

    # Transformer Architecture
    st.markdown("""
    <h4 style='color: #84A4FC; font-size: 1.2rem; margin-top: 1rem; margin-bottom: 0.75rem;'>
        4. Attention Mechanism: Transformer Architecture
    </h4>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='color: #c1c7d0; line-height: 1.7; margin-bottom: 1rem;'>
    <strong style='color: #e8eaed;'>Research:</strong> Vaswani et al. (2017) - 
    <a href='https://arxiv.org/abs/1706.03762' target='_blank' style='color: #84A4FC;'>NeurIPS 2017</a><br/>
    <strong style='color: #e8eaed;'>Innovation:</strong> Self-attention for parallel sequence processing<br/>
    <strong style='color: #e8eaed;'>BERT Foundation:</strong> 12 layers, 12 attention heads per layer, enables fast CPU inference
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ===== KEY FEATURES =====
st.markdown("""
<div style='text-align: center; margin: 3rem 0 2rem 0;'>
    <h2 style='font-size: 2.5rem; color: #e8eaed;'>Key Features</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='custom-card'>
        <h3 style='color: #84A4FC; margin-bottom: 1rem;'>🛡️ Real-Time Detection</h3>
        <p style='color: #c1c7d0; line-height: 1.7;'>
        Analyze text and images (OCR) in seconds. Process thousands of messages per hour 
        on consumer hardware. Perfect for real-time chat moderation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='custom-card'>
        <h3 style='color: #84A4FC; margin-bottom: 1rem;'>🤖 AI-Powered Insights</h3>
        <p style='color: #c1c7d0; line-height: 1.7;'>
        Get explanations, not just classifications. Understand WHY content was flagged 
        and WHAT to do about it with AI-powered contextual analysis and recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='custom-card'>
        <h3 style='color: #84A4FC; margin-bottom: 1rem;'>🎯 Exceptional Performance</h3>
        <p style='color: #c1c7d0; line-height: 1.7;'>
        <strong style='color: #10b981;'>96.8% ROC-AUC</strong> through two-stage pipeline: 
        fine-tuning on 120K+ samples from 5 datasets, then Isotonic calibration achieving 
        <strong style='color: #10b981;'>71% ECE reduction</strong> (superior probability calibration) and 
        <strong style='color: #10b981;'>67.2% F1 score</strong> with 86.7% recall.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='custom-card'>
        <h3 style='color: #84A4FC; margin-bottom: 1rem;'>✨ Content Improvement</h3>
        <p style='color: #c1c7d0; line-height: 1.7;'>
        Learn how to rephrase harmful content constructively. Educational, not punitive—helping 
        people communicate better.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='custom-card'>
        <h3 style='color: #84A4FC; margin-bottom: 1rem;'>📊 Uncertainty Quantification</h3>
        <p style='color: #c1c7d0; line-height: 1.7;'>
        Know when the model is uncertain with Monte Carlo Dropout. Get confidence scores 
        for every prediction to make better decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='custom-card'>
        <h3 style='color: #84A4FC; margin-bottom: 1rem;'>🔒 Privacy-First Design</h3>
        <p style='color: #c1c7d0; line-height: 1.7;'>
        Your API key, your data. BERT detection happens entirely on-device with zero external 
        calls. COPPA and FERPA compliant.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ===== CALL TO ACTION =====
st.markdown("""
<div style='text-align: center; margin: 3rem 0 2rem 0;'>
    <h2 style='font-size: 2.5rem; color: #e8eaed;'>Start Protecting Your Community Today</h2>
        <p style='font-size: 1.15rem; color: #c1c7d0; max-width: 700px; margin: 1rem auto 2rem auto;'>
        Deploy efficient, accessible BERT-powered content safety today
        </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("🔍 Try Detector", type="primary", use_container_width=True, key="btn_try_detector_bottom"):
            st.switch_page("pages/1_🔍_Detector.py")
    with col_b:
        if st.button("📊 View Performance", use_container_width=True, key="btn_view_performance_bottom"):
            st.switch_page("pages/2_📊_Performance.py")
    with col_c:
        if st.button("📚 Learn More", use_container_width=True, key="btn_learn_more_bottom"):
            st.switch_page("pages/3_📚_Learn_More.py")

st.markdown("<br><br>", unsafe_allow_html=True)

# ===== CITATIONS & FOOTER =====
st.markdown("## 📖 Research Citations & Data Sources")

st.markdown("""
**Cyberbullying Statistics:**

• Pew Research Center. (2022). ["Teens, Social Media and Technology 2022."](https://www.pewresearch.org/internet/2022/08/10/teens-social-media-and-technology-2022/)

• Hinduja, S., & Patchin, J. W. (2010). ["Bullying, Cyberbullying, and Suicide."](https://www.tandfonline.com/doi/abs/10.1080/13811118.2010.494133) Archives of Suicide Research, 14(3), 206-221.

• Cyberbullying Research Center. (2023). ["Cyberbullying Facts and Statistics."](https://cyberbullying.org/facts)

• Meta. (2024). ["Community Standards Enforcement Report."](https://transparency.meta.com/reports/community-standards-enforcement/)


**AI & Machine Learning Research:**

• Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."](https://arxiv.org/abs/1810.04805) arXiv preprint arXiv:1810.04805.

• Gal, Y., & Ghahramani, Z. (2016). ["Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning."](https://proceedings.mlr.press/v48/gal16.html) Proceedings of The 33rd International Conference on Machine Learning (ICML).

• Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). ["Attention Is All You Need."](https://arxiv.org/abs/1706.03762) Advances in Neural Information Processing Systems (NeurIPS).

• Howard, J., & Ruder, S. (2018). ["Universal Language Model Fine-tuning for Text Classification."](https://arxiv.org/abs/1801.06146) ACL.


**Training Datasets:**

• Trained on 120,000+ samples from publicly available toxicity and cyberbullying datasets.

• Balanced representation across demographic groups and content types.

**Public Dataset Sources:**

• [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) - Kaggle dataset with 160,000+ Wikipedia comments labeled for toxicity

• [Hate Speech and Offensive Language Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language) - 25,000 tweets labeled by crowdworkers

• [Civil Comments Dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) - 2 million public comments from Civil Comments platform

• [Twitter Cyberbullying Dataset](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification) - Multi-class cyberbullying detection dataset

• [Formspring Cyberbullying Dataset](https://www.kaggle.com/datasets/swetaagrawal/formspring-data-for-cyberbullying-detection) - Social Q&A platform data for cyberbullying research
""")

# Page navigation
page_navigation()

# Navigation footer
navigation_footer()


