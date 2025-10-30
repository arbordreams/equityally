"""
Shared utilities for Equity Ally multi-page application
Includes CSS styling, logo loading, page configuration, and reusable components
"""

import streamlit as st
import base64
from pathlib import Path


def page_config(page_title="Equity Ally", page_icon="🛡️", layout="wide"):
    """
    Configure Streamlit page settings consistently across all pages
    
    Args:
        page_title: Title for the browser tab
        page_icon: Emoji or path to icon
        layout: "wide" or "centered"
    """
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )


def get_base64_logo(logo_path):
    """
    Load and encode logo to base64 for embedding
    
    Args:
        logo_path: Path to the logo file (relative to equity-detector directory)
        
    Returns:
        Base64 encoded string or None if file not found
    """
    try:
        # Resolve path relative to the parent directory (equity-detector)
        current_dir = Path(__file__).resolve().parent.parent
        full_path = current_dir / logo_path
        with open(full_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        return None


def load_logo(logo_path="assets/equitylogolong.svg", max_width="500px"):
    """
    Display the Equity Ally logo
    
    Args:
        logo_path: Path to logo file (relative to equity-detector directory)
        max_width: Maximum width of the logo
    """
    logo_base64 = get_base64_logo(logo_path)
    
    if logo_base64:
        st.markdown(f"""
        <div class='logo-container' style='text-align: center; margin: 2rem 0;'>
            <img src='data:image/svg+xml;base64,{logo_base64}' 
                 alt='Equity Ally Logo' 
                 style='max-width: {max_width}; height: auto;'/>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback text-based logo
        st.markdown("""
        <div style='text-align: center; margin: 2rem 0;'>
            <h1 style='color: #e8eaed; font-size: 3rem; margin: 0;'>Equity Ally</h1>
        </div>
        """, unsafe_allow_html=True)


def apply_custom_css():
    """
    Apply the complete Equity Ally dark mode theme CSS
    """
    st.markdown("""
<style>
    /* Import Google Fonts - Professional sans-serif */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');
    
    /* Color Palette - Dark Mode
    - Blue Ribbon (Primary): #1463F3
    - Deep Dark: #0a0e12
    - Dark Gray: #1a1d23
    - Medium Gray: #2d3139
    - Light Text: #e8eaed
    - Muted Text: #9ca3af
    - Malibu (Light Blue): #84A4FC
    - Logo Blue: #4b5ae4
    */
    
    /* Smooth Scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Force dark background on everything */
    .stApp {
        background: #0a0e12 !important;
    }
    
    /* Main container styling - Dark background */
    .main {
        background: #0a0e12 !important;
        padding: 2rem 3rem;
        color: #e8eaed !important;
    }
    
    /* Header styling - Light text for dark mode */
    h1 {
        font-family: 'DM Sans', sans-serif !important;
        color: #e8eaed !important;
        font-weight: 700 !important;
        font-size: 3.5rem !important;
        text-align: center !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.03em !important;
    }
    
    h2 {
        font-family: 'DM Sans', sans-serif !important;
        color: #e8eaed !important;
        font-weight: 600 !important;
        margin-top: 3rem !important;
        font-size: 1.8rem !important;
        letter-spacing: -0.02em !important;
    }
    
    h3 {
        color: #e8eaed !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
        letter-spacing: -0.01em !important;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #c1c7d0;
        font-size: 1.25rem;
        font-weight: 400;
        margin-bottom: 1rem;
    }
    
    /* Text and labels - Light for dark mode with improved contrast */
    .main p, .main li, .main span {
        color: #c1c7d0 !important;
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
    }
    
    .main label {
        color: #e8eaed !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    
    /* Text area styling - Dark mode */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #2d3139 !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
        background: #1a1d23 !important;
        color: #e8eaed !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #1463F3 !important;
        box-shadow: 0 0 0 3px rgba(20, 99, 243, 0.2) !important;
        outline: none !important;
    }
    
    /* Placeholder text styling - Make it visible in dark mode */
    .stTextArea textarea::placeholder {
        color: #c1c7d0 !important;
        opacity: 1 !important;
    }
    
    /* Text input styling - Dark mode */
    .stTextInput input {
        border-radius: 10px !important;
        border: 2px solid #2d3139 !important;
        padding: 0.75rem !important;
        transition: all 0.3s ease !important;
        background: #1a1d23 !important;
        color: #e8eaed !important;
    }
    
    .stTextInput input:focus {
        border-color: #1463F3 !important;
        box-shadow: 0 0 0 3px rgba(20, 99, 243, 0.2) !important;
        outline: none !important;
    }
    
    /* Placeholder text styling for text inputs */
    .stTextInput input::placeholder {
        color: #c1c7d0 !important;
        opacity: 1 !important;
    }
    
    /* Select box styling - Dark mode */
    .stSelectbox > div > div {
        border-radius: 10px !important;
        border: 2px solid #2d3139 !important;
        background: #1a1d23 !important;
        color: #e8eaed !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #1463F3 !important;
    }
    
    /* Radio button styling - Dark mode */
    .stRadio > label {
        font-weight: 500 !important;
        color: #e8eaed !important;
    }
    
    .stRadio > div {
        background: #1a1d23;
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid #2d3139;
    }
    
    /* File uploader styling - Dark mode */
    .stFileUploader {
        background: #1a1d23;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px dashed #2d3139;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #1463F3;
        background: #1f2329;
    }
    
    /* Button styling - Dark gray with white text for high visibility */
    .stButton > button {
        background: #2d3139 !important;
        color: #ffffff !important;
        border: 2px solid #2d3139 !important;
        border-radius: 12px !important;
        padding: 0.95rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(20, 99, 243, 0.3) !important;
        width: 100% !important;
        letter-spacing: 0.02em !important;
        min-height: 48px !important;
    }
    
    .stButton > button:hover {
        background: #1463F3 !important;
        color: #ffffff !important;
        border-color: #1463F3 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(20, 99, 243, 0.5) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    .stButton > button:disabled {
        background: #1a1d23 !important;
        color: #6b7280 !important;
        border-color: #1a1d23 !important;
        cursor: not-allowed !important;
        opacity: 0.6 !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* Metric styling - Blue accent for dark mode */
    [data-testid="stMetricValue"] {
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        color: #84A4FC !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #c1c7d0 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    
    /* Progress bar - Blue gradient */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1463F3 0%, #84A4FC 100%) !important;
        height: 10px !important;
        border-radius: 5px !important;
    }
    
    /* Alert boxes - Dark mode */
    .stAlert {
        border-radius: 12px !important;
        border-left: 4px solid !important;
        padding: 1rem 1.25rem !important;
        background: #1a1d23 !important;
    }
    
    /* Info box - Dark mode */
    div[data-baseweb="notification"] {
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(20, 99, 243, 0.3) !important;
        background: #1a1d23 !important;
    }
    
    /* Expander styling - NEVER WHITE - All states forced to dark */
    .streamlit-expanderHeader,
    .streamlit-expanderHeader:hover,
    .streamlit-expanderHeader:active,
    .streamlit-expanderHeader:focus,
    .streamlit-expanderHeader:focus-within,
    .streamlit-expanderHeader[aria-expanded="true"],
    .streamlit-expanderHeader[aria-expanded="false"],
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:active,
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] details summary,
    [data-testid="stExpander"] details[open] summary {
        background: #1a1d23 !important;
        background-color: #1a1d23 !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        color: #e8eaed !important;
        padding: 1.25rem 1.5rem !important;
        border: 2px solid #2d3139 !important;
        transition: all 0.3s ease !important;
    }
    
    /* Expander header when expanded - keep same dark background */
    .streamlit-expanderHeader[aria-expanded="true"],
    [data-testid="stExpander"] details[open] summary {
        background: #1a1d23 !important;
        background-color: #1a1d23 !important;
        border-radius: 12px 12px 0 0 !important;
    }
    
    /* Expander header hover state - slightly lighter but never white */
    .streamlit-expanderHeader:hover,
    [data-testid="stExpander"] summary:hover {
        background: #1f2329 !important;
        background-color: #1f2329 !important;
        border-color: #1463F3 !important;
    }
    
    /* All nested elements inside expander - force dark/transparent */
    .streamlit-expanderHeader *,
    .streamlit-expanderHeader button,
    .streamlit-expanderHeader button:hover,
    .streamlit-expanderHeader button:active,
    .streamlit-expanderHeader button:focus,
    .streamlit-expanderHeader div,
    .streamlit-expanderHeader div:hover,
    .streamlit-expanderHeader div:active,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] summary button,
    [data-testid="stExpander"] summary div {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Expander content area */
    .streamlit-expanderContent {
        padding: 1.5rem !important;
        background: #1a1d23 !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    /* Expander content text styling */
    .streamlit-expanderContent p,
    .streamlit-expanderContent li,
    .streamlit-expanderContent span,
    .streamlit-expanderContent div {
        color: #c1c7d0 !important;
    }
    
    .streamlit-expanderContent strong {
        color: #e8eaed !important;
    }
    
    /* Tab styling - Dark mode */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1a1d23;
        padding: 0.5rem;
        border-radius: 12px;
        border: 2px solid #2d3139;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        color: #c1c7d0 !important;
        background: transparent !important;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #1463F3 0%, #4b5ae4 100%) !important;
        box-shadow: 0 2px 8px rgba(20, 99, 243, 0.4) !important;
        color: #ffffff !important;
    }
    
    /* Dataframe styling - Dark mode */
    .dataframe {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 2px solid #2d3139 !important;
        background: #1a1d23 !important;
    }
    
    /* Sidebar styling - Darker gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #1a1d23 100%) !important;
        padding: 2rem 1rem !important;
        border-right: 1px solid #2d3139 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #e8eaed !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    
    /* Divider - Dark mode */
    hr {
        margin: 2.5rem 0 !important;
        border: none !important;
        border-top: 2px solid #2d3139 !important;
    }
    
    /* Card components - Dark mode */
    .custom-card {
        background: #1a1d23;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.5rem;
        border: 2px solid #2d3139;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        box-shadow: 0 8px 24px rgba(20, 99, 243, 0.3);
        transform: translateY(-2px);
        border-color: #1463F3;
    }
    
    /* Success/Error states */
    .element-container div[data-testid="stMarkdownContainer"] p {
        line-height: 1.7 !important;
        color: #c1c7d0 !important;
    }
    
    /* Checkbox styling - Dark mode */
    .stCheckbox {
        padding: 0.5rem 0;
    }
    
    .stCheckbox label {
        font-weight: 500 !important;
        color: #e8eaed !important;
    }
    
    /* Slider styling - Simple white bar design */
    .stSlider {
        padding: 0.5rem 0 1rem 0;
    }
    
    /* Slider label */
    .stSlider > label {
        color: #e8eaed !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Slider track (the background bar) */
    .stSlider [data-baseweb="slider"] [role="presentation"] {
        background: #ffffff !important;
        height: 4px !important;
        border-radius: 2px !important;
    }
    
    /* Slider thumb (the draggable circle) */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #ffffff !important;
        width: 18px !important;
        height: 18px !important;
        border: 2px solid #2d3139 !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Hide the value tooltip */
    .stSlider [data-baseweb="popover"] {
        display: none !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Professional accent colors - Dark mode theme */
    .success-accent {
        color: #047857 !important;
    }
    
    .warning-accent {
        color: #f59e0b !important;
    }
    
    .error-accent {
        color: #dc2626 !important;
    }
    
    .primary-accent {
        color: #84A4FC !important;
    }
    
    /* Plotly chart styling - Dark mode */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(20, 99, 243, 0.3);
        border: 2px solid #2d3139;
    }
    
    /* Additional professional touches - Dark mode */
    [data-testid="stMarkdownContainer"] {
        color: #c1c7d0;
    }
    
    /* Code blocks - Dark mode */
    code {
        background: #1a1d23 !important;
        color: #84A4FC !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 6px !important;
        border: 1px solid #2d3139 !important;
    }
    
    pre {
        background: #1a1d23 !important;
        border: 2px solid #2d3139 !important;
        border-radius: 12px !important;
    }
    
    /* Logo styling */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .logo-container img {
        max-width: 500px;
        height: auto;
    }
    
    /* Keyboard Accessibility - WCAG 2.2 AA Focus Indicators */
    *:focus-visible {
        outline: 3px solid #1463F3 !important;
        outline-offset: 3px !important;
        border-radius: 4px !important;
    }
    
    button:focus-visible,
    input:focus-visible,
    textarea:focus-visible,
    select:focus-visible {
        outline: 3px solid #1463F3 !important;
        outline-offset: 2px !important;
    }
    
    .stButton > button:focus-visible {
        outline: 3px solid #1463F3 !important;
        outline-offset: 3px !important;
        box-shadow: 0 0 0 3px rgba(20, 99, 243, 0.3) !important;
    }
    
    /* Step indicators */
    .step-indicator {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1463F3 0%, #4b5ae4 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 1rem;
        margin-right: 0.75rem;
        box-shadow: 0 2px 8px rgba(20, 99, 243, 0.4);
    }
    
    .step-header {
        display: flex;
        align-items: center;
        margin-bottom: 1.25rem;
        color: #e8eaed;
        font-weight: 600;
        font-size: 1.15rem;
    }
    
    /* Info boxes with better contrast */
    .info-box {
        background: #1a1d23;
        border-left: 4px solid #1463F3;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        color: #c1c7d0;
        line-height: 1.7;
    }
    
    .info-box strong {
        color: #e8eaed;
    }
</style>
""", unsafe_allow_html=True)


def api_key_sidebar():
    """
    Compact API key input in sidebar with collapsible help
    Returns the API key or None
    """
    with st.sidebar:
        # Initialize session state for API key
        if 'openai_api_key' not in st.session_state:
            st.session_state.openai_api_key = None
        
        # Compact header with status indicator
        if st.session_state.openai_api_key:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        padding: 0.5rem 0.75rem; border-radius: 8px; margin-bottom: 0.75rem;
                        text-align: center; border: 2px solid #10b981;'>
                <strong style='color: #ffffff; font-size: 0.95rem;'>🤖 AI Active</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: #1a1d23; padding: 0.5rem 0.75rem; border-radius: 8px; 
                        margin-bottom: 0.75rem; text-align: center; border: 2px solid #2d3139;'>
                <strong style='color: #c1c7d0; font-size: 0.95rem;'>🤖 AI Assistant</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Compact API key input
        api_key = st.text_input(
            "OpenAI API Key (Optional)",
            value=st.session_state.openai_api_key or "",
            type="password",
            help="Enable GPT-4o features",
            placeholder="sk-...",
            label_visibility="collapsed"
        )
        
        # Update session state
        if api_key:
            st.session_state.openai_api_key = api_key
        
        # Compact status or help
        if st.session_state.openai_api_key:
            st.markdown("""
            <div style='font-size: 0.85rem; color: #10b981; margin-top: -0.5rem; margin-bottom: 0.5rem;'>
                ✓ GPT-4o enabled
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.expander("ℹ️ Get API Key", expanded=False):
                st.markdown("""
                <div style='font-size: 0.85rem; line-height: 1.5;'>
                1. Visit <a href='https://platform.openai.com/api-keys' target='_blank' style='color: #84A4FC;'>OpenAI Platform</a><br/>
                2. Sign up/login<br/>
                3. Create new key<br/>
                4. Paste above<br/><br/>
                <em style='color: #9ca3af;'>Key stored in session only</em>
                </div>
                """, unsafe_allow_html=True)
        
        return st.session_state.openai_api_key


def api_key_compact(unique_key="default"):
    """
    Ultra-compact inline API key input for pages
    Can be placed anywhere in the main content
    Returns the API key or None
    
    Args:
        unique_key: Unique identifier to prevent duplicate widget keys across pages
    """
    # Initialize session state for API key
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    
    # Compact inline version
    col1, col2 = st.columns([3, 1])
    
    with col1:
        api_key = st.text_input(
            "OpenAI API Key (Optional for AI features)",
            value=st.session_state.openai_api_key or "",
            type="password",
            help="Enable GPT-4o explanations and recommendations. Enter once and it works across all pages.",
            placeholder="sk-...",
            key=f"api_key_input_{unique_key}"
        )
        
        # Update session state if changed
        if api_key:
            st.session_state.openai_api_key = api_key
    
    with col2:
        if st.session_state.openai_api_key:
            st.markdown("""
            <div style='background: #10b981; padding: 0.65rem 1rem; border-radius: 8px; 
                        text-align: center; margin-top: 1.75rem;'>
                <strong style='color: #ffffff; font-size: 0.9rem;'>✓ Active</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: #1a1d23; padding: 0.65rem 1rem; border-radius: 8px; 
                        text-align: center; margin-top: 1.75rem; border: 2px solid #2d3139;'>
                <a href='https://platform.openai.com/api-keys' target='_blank' 
                   style='color: #84A4FC; text-decoration: none; font-size: 0.9rem;'>Get Key</a>
            </div>
            """, unsafe_allow_html=True)
    
    return st.session_state.openai_api_key


def show_ai_status():
    """
    Compact status indicator for AI features
    """
    if st.session_state.get('openai_api_key'):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #047857 0%, #059669 100%); 
                    padding: 0.6rem 1rem; border-radius: 8px; margin-bottom: 1rem;
                    border: 2px solid #10b981; text-align: center;'>
            <strong style='color: #ffffff; font-size: 0.95rem;'>🤖 AI Enhanced • GPT-4o Active</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: #1a1d23; padding: 0.6rem 1rem; border-radius: 8px; 
                    margin-bottom: 1rem; border: 2px solid #2d3139; text-align: center;'>
            <strong style='color: #e8eaed; font-size: 0.95rem;'>⚡ BERT Active (96.8% ROC-AUC)</strong>
            <span style='color: #9ca3af; font-size: 0.85rem;'> • Optional: Add OpenAI key below for GPT insights</span>
        </div>
        """, unsafe_allow_html=True)


def page_navigation():
    """
    Simple navigation bar at the bottom for easy page switching
    """
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1d23 0%, #0d1117 100%); 
                border: 2px solid #2d3139; border-radius: 16px; 
                padding: 1.5rem 2rem; margin: 2rem 0;'>
        <p style='text-align: center; color: #84A4FC; font-weight: 600; 
                  font-size: 1.1rem; margin-bottom: 1rem;'>
            🧭 Navigate
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create navigation buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🏠 Home", use_container_width=True, key="nav_home"):
            st.switch_page("Home.py")
    
    with col2:
        if st.button("🔍 Detector", use_container_width=True, key="nav_detector"):
            st.switch_page("pages/1_🔍_Detector.py")
    
    with col3:
        if st.button("📊 Performance", use_container_width=True, key="nav_performance"):
            st.switch_page("pages/2_📊_Performance.py")
    
    with col4:
        if st.button("📚 Learn More", use_container_width=True, key="nav_learn"):
            st.switch_page("pages/3_📚_Learn_More.py")
    
    with col5:
        if st.button("ℹ️ About", use_container_width=True, key="nav_about"):
            st.switch_page("pages/4_ℹ️_About.py")


def navigation_footer():
    """
    Consistent footer across all pages
    """
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2.5rem 0; color: #c1c7d0;'>
        <p style='font-size: 1rem; margin-bottom: 0.75rem;'>
            <strong style='color: #e8eaed; font-weight: 600; font-size: 1.1rem;'>Equity Ally</strong> 
            <span style='color: #6b7280;'>—</span> 
            <span style='color: #c1c7d0;'>AI-Powered Content Safety Platform</span>
        </p>
        <p style='font-size: 0.95rem; color: #c1c7d0; letter-spacing: 0.02em; line-height: 1.6;'>
            <strong style='color: #10b981;'>96.8% ROC-AUC</strong> &nbsp;|&nbsp; 110M Parameters &nbsp;|&nbsp; 418MB Model
        </p>
        <p style='font-size: 0.9rem; color: #c1c7d0; margin-top: 1.25rem; line-height: 1.6;'>
            Promoting safer, more inclusive online communities through responsible AI
        </p>
        <p style='font-size: 0.85rem; color: #6b7280; margin-top: 1rem;'>
            Open-Source AI for Social Good
        </p>
    </div>
    """, unsafe_allow_html=True)


