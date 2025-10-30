"""
Equity Ally - Content Detector
Enhanced with AI-powered contextual analysis
"""

import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pytesseract
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.shared import page_config, apply_custom_css, load_logo, api_key_sidebar, api_key_compact, show_ai_status, page_navigation, navigation_footer
from utils.bert_model import load_model, predict_single, get_severity_level, calculate_mapk, predict_batch
from utils.openai_helper import explain_detection, get_recommendations, suggest_improvements, verify_with_gpt, chat_about_model

# Page configuration
page_config("Equity Ally - Detector", "🔍", "wide")

# Apply custom CSS
apply_custom_css()

# Initialize session state
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None
if 'ai_explanation' not in st.session_state:
    st.session_state.ai_explanation = None
if 'ai_recommendations' not in st.session_state:
    st.session_state.ai_recommendations = None
if 'ai_improvements' not in st.session_state:
    st.session_state.ai_improvements = None

# Load BERT model
model, tokenizer, device = load_model()

# === OCR Function ===
def extract_text_from_image(image):
    """Extract text from image using pytesseract OCR"""
    try:
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text.strip()
    except Exception as e:
        st.error(f"⚠️ OCR extraction failed. Error: {str(e)}")
        st.info("💡 Tip: Make sure Tesseract OCR is installed on your system. See the OCR setup guide for installation instructions.")
        return None

# === HEADER ===
load_logo("assets/equitylogolong.svg", max_width="500px")

st.markdown("""
<div style='text-align: center; padding: 1rem 0 2rem 0;'>
    <p style='font-size: 1.25rem; color: #c1c7d0; font-weight: 400; margin-bottom: 0.75rem; letter-spacing: 0.01em;'>
        AI-Powered Content Safety Platform
    </p>
    <p style='font-size: 1.05rem; color: #c1c7d0; max-width: 700px; margin: 0 auto; line-height: 1.7;'>
        BERT-powered content safety that runs entirely on your device<br/>
        Efficient, private, and accessible to everyone
    </p>
</div>
""", unsafe_allow_html=True)

# Show AI status
show_ai_status()

# Inline API key input for AI features
with st.expander("🔑 Optional: Enable AI Features (GPT-4o)", expanded=not st.session_state.get('openai_api_key')):
    st.markdown("""
    <p style='color: #c1c7d0; font-size: 0.95rem; margin-bottom: 0.75rem; line-height: 1.6;'>
        Add your OpenAI API key to unlock enhanced AI features including explanations, 
        recommendations, and content improvement suggestions.
    </p>
    """, unsafe_allow_html=True)
    api_key = api_key_compact("detector_page")
    
st.markdown("---")

# === STEP 1: INPUT SELECTION ===
st.markdown("""
<div class='step-header'>
    <span class='step-indicator'>1</span>
    <span>Choose Your Input Method</span>
</div>
""", unsafe_allow_html=True)

# Initialize variables
user_input = ""
extracted_text_from_image = ""

# Input method selector
input_method = st.radio(
    "Select how you want to provide content for analysis:",
    ["✍️ Type or Paste Text", "🖼️ Upload Image (OCR)", "📊 Upload CSV/Bulk Analysis"],
    horizontal=True
)

st.markdown("<br>", unsafe_allow_html=True)

if input_method == "✍️ Type or Paste Text":
    # Text input
    st.markdown("<p style='color: #c1c7d0; margin-bottom: 0.5rem; font-size: 1.05rem;'>Paste or type the content you'd like to analyze. This could be a message, comment, or any text.</p>", unsafe_allow_html=True)
    user_input = st.text_area(
        "Enter text for analysis:",
        "",
        height=200,
        placeholder="Example: 'Great work on the presentation!' or paste any text you'd like to check...",
        key="text_input",
        help="Type or paste the text you want to analyze for potentially concerning content",
        label_visibility="collapsed"
    )
    
elif input_method == "🖼️ Upload Image (OCR)":
    st.markdown("""
    <div class='info-box'>
        <strong>📷 Image Text Extraction</strong><br/>
        Upload an image containing text (such as screenshots, photos of messages, or documents) and we'll automatically extract the text using OCR technology for analysis.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader(
        "Choose an image file (PNG, JPG, JPEG, BMP, TIFF)",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        key="image_upload",
        help="Upload a clear image with visible text for best results"
    )
    
    if uploaded_image is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)
        
        with col2:
            with st.spinner("🔍 Extracting text from image..."):
                extracted_text_from_image = extract_text_from_image(image)
            
            if extracted_text_from_image:
                st.success("✅ Text successfully extracted from your image!")
                st.text_area(
                    "Extracted Text (you can edit if needed):",
                    extracted_text_from_image,
                    height=200,
                    key="extracted_display",
                    help="Review and edit the extracted text if needed before analysis"
                )
                user_input = extracted_text_from_image
            else:
                st.error("⚠️ We couldn't extract text from this image. Please try a clearer image or use the text input option instead.")

else:  # CSV/Bulk Upload
    st.markdown("""
    <div class='info-box'>
        <strong>📊 Bulk Analysis Mode</strong><br/>
        Upload a CSV file containing multiple text entries for batch analysis. Get comprehensive statistics, visualizations, and downloadable results.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style='color: #c1c7d0; margin-bottom: 0.5rem; font-size: 1.05rem;'>
    <strong>CSV Format Requirements:</strong><br/>
    • Must have a column containing text to analyze<br/>
    • Optional: Include an 'id' or 'name' column for tracking<br/>
    • Supported formats: .csv, .xlsx, .xls
    </p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="bulk_upload",
        help="Upload a file with text data for bulk analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
            
            # Show preview
            with st.expander("📋 Preview Data (First 5 rows)", expanded=True):
                st.dataframe(df.head(), use_container_width=True)
            
            # Column selection
            st.markdown("<p style='color: #e8eaed; font-weight: 600; font-size: 1.05rem; margin-top: 1rem;'>Select the column containing text to analyze:</p>", unsafe_allow_html=True)
            text_column = st.selectbox(
                "Text column:",
                options=df.columns.tolist(),
                help="Select the column that contains the text you want to analyze"
            )
            
            # Optional ID column
            id_column = st.selectbox(
                "Optional: Select ID/Name column (for tracking):",
                options=["None"] + df.columns.tolist(),
                help="Select a column to use as identifier for each row"
            )
            
            # Store for batch processing
            if 'bulk_data' not in st.session_state:
                st.session_state.bulk_data = None
            
            st.session_state.bulk_data = {
                'df': df,
                'text_column': text_column,
                'id_column': id_column if id_column != "None" else None
            }
            
        except Exception as e:
            st.error(f"⚠️ Error reading file: {str(e)}")
            st.info("💡 Make sure your file is a valid CSV or Excel file with proper formatting.")

st.markdown("---")

# === ADVANCED OPTIONS ===
with st.expander("⚙️ Advanced Options - Monte Carlo Dropout (For Researchers & Advanced Users)"):
    st.markdown("""
    <div style='color: #c1c7d0; line-height: 1.5; font-size: 0.9rem;'>
    <strong style='color: #e8eaed;'>Monte Carlo Dropout</strong> runs multiple predictions with dropout enabled for robust results and uncertainty measurement.
    📊 Ensemble averaging • 📉 Uncertainty quantification • 🎯 Better edge case handling
    </div>
    """, unsafe_allow_html=True)
    
    n_passes = st.slider("Number of Forward Passes", min_value=1, max_value=100, value=1, step=1,
                        help="1 = Standard inference (fastest), 2+ = Monte Carlo Dropout (more robust). 20-30 recommended for best balance.")
    
    # Add tick marks for better UX
    st.markdown("""
    <div style='display: flex; justify-content: space-between; margin-top: -1.25rem; margin-bottom: 0.5rem; padding: 0 0.25rem;'>
        <span style='color: #10b981; font-size: 0.7rem; font-weight: 600;'>1</span>
        <span style='color: #84cc16; font-size: 0.7rem; font-weight: 500;'>25</span>
        <span style='color: #fbbf24; font-size: 0.7rem; font-weight: 500;'>50</span>
        <span style='color: #fb923c; font-size: 0.7rem; font-weight: 500;'>75</span>
        <span style='color: #ef4444; font-size: 0.7rem; font-weight: 600;'>100</span>
    </div>
    """, unsafe_allow_html=True)
    
    use_mc_dropout = n_passes > 1
    
    # Dynamic color indicator based on current value
    def get_color_and_zone(value):
        if value <= 20:
            return "#10b981", "🟢 Low Intensity", "Fast & efficient"
        elif value <= 40:
            return "#84cc16", "🟡 Moderate", "Balanced approach"
        elif value <= 60:
            return "#fbbf24", "🟠 Medium-High", "More robust"
        elif value <= 80:
            return "#fb923c", "🟠 High Intensity", "Very thorough"
        else:
            return "#ef4444", "🔴 Maximum", "Most robust & slowest"
    
    color, zone, description = get_color_and_zone(n_passes)
    st.markdown(f"""
    <div style='background: rgba(26, 29, 35, 0.5); border-left: 3px solid {color}; 
                padding: 0.5rem 0.75rem; border-radius: 6px; margin-bottom: 0.5rem;'>
        <div style='display: flex; align-items: center; gap: 0.5rem;'>
            <div style='background: {color}; width: 8px; height: 8px; border-radius: 50%; 
                        box-shadow: 0 0 8px {color};'></div>
            <div style='font-size: 0.85rem;'>
                <strong style='color: #e8eaed;'>{zone}</strong>
                <span style='color: #9ca3af; margin-left: 0.35rem;'>• {description}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if use_mc_dropout:
        estimated_time = n_passes * 0.07  # Rough estimate
        st.markdown(f"""
        <div style='background: rgba(26, 29, 35, 0.3); padding: 0.4rem 0.75rem; 
                    border-radius: 6px; font-size: 0.85rem; color: #9ca3af;'>
            <strong style='color: #e8eaed;'>Config:</strong> {n_passes} passes • 
            <strong style='color: #e8eaed;'>Time:</strong> ~{estimated_time:.1f}s
            {' • ✅ <strong style="color: #10b981;">Recommended</strong>' if 15 <= n_passes <= 30 else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: rgba(26, 29, 35, 0.3); padding: 0.4rem 0.75rem; 
                    border-radius: 6px; font-size: 0.85rem; color: #9ca3af;'>
            <strong style='color: #e8eaed;'>Mode:</strong> Standard (single pass) • 
            <strong style='color: #e8eaed;'>Time:</strong> ~0.1s
        </div>
        """, unsafe_allow_html=True)

# === STEP 2: ANALYZE ===
st.markdown("""
<div class='step-header' style='margin-top: 2rem;'>
    <span class='step-indicator'>2</span>
    <span>Analyze Content</span>
</div>
""", unsafe_allow_html=True)

# Different button for bulk vs single analysis
is_bulk_mode = input_method == "📊 Upload CSV/Bulk Analysis"
bulk_ready = is_bulk_mode and st.session_state.get('bulk_data') is not None
single_ready = not is_bulk_mode and user_input

analyze_button_text = "🔍 Analyze All Entries" if is_bulk_mode else "🔍 Analyze Text"
analyze_disabled = not (bulk_ready or single_ready)

if st.button(analyze_button_text, type="primary", disabled=analyze_disabled):
    # === BULK ANALYSIS MODE ===
    if is_bulk_mode and bulk_ready and model and tokenizer:
        bulk_data = st.session_state.bulk_data
        df = bulk_data['df'].copy()
        text_column = bulk_data['text_column']
        id_column = bulk_data['id_column']
        
        # Get texts to analyze
        texts = df[text_column].astype(str).tolist()
        
        # Filter out empty or very short texts
        valid_indices = [i for i, text in enumerate(texts) if len(str(text).strip()) > 5]
        valid_texts = [texts[i] for i in valid_indices]
        
        if len(valid_texts) == 0:
            st.error("⚠️ No valid text entries found in the selected column.")
        else:
            st.info(f"📊 Analyzing {len(valid_texts)} text entries...")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Batch analysis
            use_mc = use_mc_dropout if 'use_mc_dropout' in locals() else False
            passes = n_passes if 'n_passes' in locals() else 20
            
            results = []
            start_time = time.time()
            
            for idx, text in enumerate(valid_texts):
                status_text.text(f"Analyzing entry {idx + 1} of {len(valid_texts)}...")
                progress_bar.progress((idx + 1) / len(valid_texts))
                
                result = predict_single(text, model, tokenizer, device, use_mc_dropout=use_mc, n_passes=passes)
                
                results.append({
                    'text': text[:100] + ('...' if len(text) > 100 else ''),  # Truncate for display
                    'full_text': text,
                    'prob_safe': result['prob_not_bullying'],
                    'prob_concerning': result['prob_bullying'],
                    'prediction': 'Concerning' if result['prediction'] == 1 else 'Safe',
                    'prediction_int': result['prediction'],
                    'uncertainty': result['uncertainty_bullying'] if use_mc else 0,
                    'original_index': valid_indices[idx]
                })
            
            end_time = time.time()
            total_time = end_time - start_time
            
            progress_bar.empty()
            status_text.empty()
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            
            # Add ID column if provided
            if id_column:
                results_df.insert(0, 'ID', [df.iloc[i][id_column] for i in valid_indices])
            
            # === DISPLAY BULK RESULTS ===
            st.markdown("---")
            st.markdown("""
            <div class='step-header'>
                <span class='step-indicator'>3</span>
                <span>Bulk Analysis Results</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Summary metrics
            st.write("## 📊 Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            num_safe = len(results_df[results_df['prediction'] == 'Safe'])
            num_concerning = len(results_df[results_df['prediction'] == 'Concerning'])
            avg_risk = results_df['prob_concerning'].mean() * 100
            
            with col1:
                st.metric("Total Analyzed", len(results_df))
            with col2:
                st.metric("Safe Content", num_safe, delta=f"{num_safe/len(results_df)*100:.1f}%")
            with col3:
                st.metric("Concerning Content", num_concerning, delta=f"{num_concerning/len(results_df)*100:.1f}%", delta_color="inverse")
            with col4:
                st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
            
            st.markdown(f"""
            <div style='background: #1a1d23; padding: 0.75rem 1.25rem; border-radius: 10px; 
                        margin: 1rem 0; border: 2px solid #10b981;'>
                <div style='text-align: center;'>
                    <strong style='color: #10b981; font-size: 1.05rem;'>⚡ Batch analysis completed in {total_time:.1f}s ({total_time/len(results_df):.2f}s per entry)</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # === VISUALIZATIONS ===
            st.write("### 📈 Analysis Visualizations")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Distribution Pie Chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Safe Content', 'Concerning Content'],
                    values=[num_safe, num_concerning],
                    marker=dict(colors=['#047857', '#b91c1c']),
                    hole=0.4,
                    textinfo='label+percent+value',
                    textfont=dict(size=14, color='#262730', family='Inter', weight='bold'),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )])
                
                fig_pie.update_layout(
                    title={
                        'text': 'Classification Distribution',
                        'font': {'size': 16, 'color': '#262730', 'family': 'Inter'}
                    },
                    height=400,
                    paper_bgcolor='#ffffff',
                    font=dict(color='#262730', family='Inter'),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.15,
                        xanchor="center",
                        x=0.5
                    )
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with viz_col2:
                # Risk Score Histogram
                fig_hist = go.Figure()
                
                fig_hist.add_trace(go.Histogram(
                    x=results_df['prob_concerning'] * 100,
                    nbinsx=20,
                    marker=dict(
                        color='#3b82f6',  # Nice blue color instead of black
                        line=dict(color='#1e40af', width=1)
                    ),
                    hovertemplate='Risk Score: %{x:.1f}%<br>Count: %{y}<extra></extra>'
                ))
                
                # Add threshold line
                fig_hist.add_vline(
                    x=40,
                    line_dash="dash",
                    line_color="#262730",
                    line_width=2,
                    annotation_text="Threshold (40%)",
                    annotation_position="top right"
                )
                
                fig_hist.update_layout(
                    title={
                        'text': 'Risk Score Distribution',
                        'font': {'size': 16, 'color': '#1f2937', 'family': 'Inter'}
                    },
                    xaxis_title='Risk Score (%)',
                    yaxis_title='Frequency',
                    height=400,
                    paper_bgcolor='#ffffff',
                    plot_bgcolor='#f9fafb',
                    font=dict(color='#1f2937', family='Inter'),
                    xaxis=dict(
                        title_font=dict(color='#1f2937'),
                        tickfont=dict(color='#1f2937'),
                        gridcolor='#e5e7eb',
                        showgrid=True
                    ),
                    yaxis=dict(
                        title_font=dict(color='#1f2937'),
                        tickfont=dict(color='#1f2937'),
                        gridcolor='#e5e7eb',
                        showgrid=True
                    ),
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Box plot and severity breakdown
            viz_col3, viz_col4 = st.columns(2)
            
            with viz_col3:
                # Box plot comparing distributions
                fig_box = go.Figure()
                
                safe_scores = results_df[results_df['prediction'] == 'Safe']['prob_concerning'] * 100
                concerning_scores = results_df[results_df['prediction'] == 'Concerning']['prob_concerning'] * 100
                
                fig_box.add_trace(go.Box(
                    y=safe_scores,
                    name='Safe Content',
                    marker=dict(color='#047857'),
                    boxmean='sd',
                    hovertemplate='<b>Safe Content</b><br>Risk Score: %{y:.1f}%<extra></extra>'
                ))
                
                fig_box.add_trace(go.Box(
                    y=concerning_scores,
                    name='Concerning Content',
                    marker=dict(color='#b91c1c'),
                    boxmean='sd',
                    hovertemplate='<b>Concerning Content</b><br>Risk Score: %{y:.1f}%<extra></extra>'
                ))
                
                fig_box.update_layout(
                    title={
                        'text': 'Risk Score by Classification',
                        'font': {'size': 16, 'color': '#262730', 'family': 'Inter'}
                    },
                    yaxis_title='Risk Score (%)',
                    height=400,
                    paper_bgcolor='#ffffff',
                    plot_bgcolor='#ffffff',
                    font=dict(color='#262730', family='Inter'),
                    xaxis=dict(color='#262730'),
                    yaxis=dict(
                        color='#262730',
                        gridcolor='#e5e7eb',
                        showgrid=True
                    ),
                    showlegend=True
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            with viz_col4:
                # Severity level breakdown
                def get_severity_category(prob):
                    if prob < 0.3:
                        return "Low Concern"
                    elif prob < 0.7:
                        return "Moderate Concern"
                    else:
                        return "High Concern"
                
                results_df['severity'] = results_df['prob_concerning'].apply(get_severity_category)
                severity_counts = results_df['severity'].value_counts()
                
                fig_severity = go.Figure(data=[go.Bar(
                    x=severity_counts.index,
                    y=severity_counts.values,
                    marker=dict(
                        color=['#047857', '#f59e0b', '#b91c1c'],
                        line=dict(color='#262730', width=2)
                    ),
                    text=severity_counts.values,
                    textposition='auto',
                    textfont=dict(size=16, color='#ffffff', family='Inter', weight='bold'),
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                )])
                
                fig_severity.update_layout(
                    title={
                        'text': 'Severity Level Breakdown',
                        'font': {'size': 16, 'color': '#262730', 'family': 'Inter'}
                    },
                    xaxis_title='Severity Level',
                    yaxis_title='Count',
                    height=400,
                    paper_bgcolor='#ffffff',
                    plot_bgcolor='#ffffff',
                    font=dict(color='#262730', family='Inter'),
                    xaxis=dict(
                        color='#262730',
                        gridcolor='#e5e7eb'
                    ),
                    yaxis=dict(
                        color='#262730',
                        gridcolor='#e5e7eb',
                        showgrid=True
                    ),
                    showlegend=False
                )
                st.plotly_chart(fig_severity, use_container_width=True)
            
            # === DETAILED RESULTS TABLE ===
            st.write("### 📋 Detailed Results")
            
            # Add filtering options
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                filter_option = st.selectbox(
                    "Filter results:",
                    ["All", "Safe Only", "Concerning Only", "High Risk Only (>70%)"]
                )
            
            with filter_col2:
                sort_option = st.selectbox(
                    "Sort by:",
                    ["Original Order", "Risk Score (High to Low)", "Risk Score (Low to High)"]
                )
            
            # Apply filters
            display_df = results_df.copy()
            
            if filter_option == "Safe Only":
                display_df = display_df[display_df['prediction'] == 'Safe']
            elif filter_option == "Concerning Only":
                display_df = display_df[display_df['prediction'] == 'Concerning']
            elif filter_option == "High Risk Only (>70%)":
                display_df = display_df[display_df['prob_concerning'] > 0.7]
            
            # Apply sorting
            if sort_option == "Risk Score (High to Low)":
                display_df = display_df.sort_values('prob_concerning', ascending=False)
            elif sort_option == "Risk Score (Low to High)":
                display_df = display_df.sort_values('prob_concerning', ascending=True)
            
            # Format for display
            display_cols = ['text', 'prob_safe', 'prob_concerning', 'prediction']
            if id_column:
                display_cols.insert(0, 'ID')
            if use_mc:
                display_cols.append('uncertainty')
            
            # Format probabilities as percentages
            display_df_formatted = display_df[display_cols].copy()
            display_df_formatted['prob_safe'] = (display_df_formatted['prob_safe'] * 100).round(2).astype(str) + '%'
            display_df_formatted['prob_concerning'] = (display_df_formatted['prob_concerning'] * 100).round(2).astype(str) + '%'
            if use_mc and 'uncertainty' in display_df_formatted.columns:
                display_df_formatted['uncertainty'] = (display_df_formatted['uncertainty'] * 100).round(2).astype(str) + '%'
            
            # Rename columns for clarity
            column_renames = {
                'text': 'Text Preview',
                'prob_safe': 'Safe %',
                'prob_concerning': 'Concerning %',
                'prediction': 'Classification'
            }
            if use_mc and 'uncertainty' in display_df_formatted.columns:
                column_renames['uncertainty'] = 'Uncertainty %'
            
            display_df_formatted = display_df_formatted.rename(columns=column_renames)
            
            st.dataframe(
                display_df_formatted,
                use_container_width=True,
                height=400
            )
            
            st.info(f"📊 Showing {len(display_df)} of {len(results_df)} total results")
            
            # === DOWNLOAD RESULTS ===
            st.markdown("### 💾 Download Results", unsafe_allow_html=True)
            
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                # Prepare full results for download
                download_df = results_df.copy()
                download_df = download_df.drop('text', axis=1)  # Remove truncated text
                download_df = download_df.rename(columns={'full_text': 'text'})
                
                # Convert to CSV
                csv = download_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"equity_analysis_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with download_col2:
                # Summary statistics for download
                summary_stats = {
                    'Total Entries': len(results_df),
                    'Safe Content': num_safe,
                    'Concerning Content': num_concerning,
                    'Safe Percentage': f"{num_safe/len(results_df)*100:.2f}%",
                    'Concerning Percentage': f"{num_concerning/len(results_df)*100:.2f}%",
                    'Average Risk Score': f"{avg_risk:.2f}%",
                    'Min Risk Score': f"{results_df['prob_concerning'].min()*100:.2f}%",
                    'Max Risk Score': f"{results_df['prob_concerning'].max()*100:.2f}%",
                    'Analysis Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Analysis Time': f"{total_time:.2f} seconds",
                    'Monte Carlo Dropout': 'Yes' if use_mc else 'No'
                }
                
                summary_text = "\n".join([f"{k}: {v}" for k, v in summary_stats.items()])
                st.download_button(
                    label="📄 Download Summary",
                    data=summary_text,
                    file_name=f"equity_analysis_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # === INSIGHTS AND RECOMMENDATIONS ===
            st.markdown("### 💡 Key Insights", unsafe_allow_html=True)
            
            insights = []
            
            if num_concerning / len(results_df) > 0.5:
                insights.append("🚨 **High Alert**: More than 50% of analyzed content was flagged as concerning. Immediate review recommended.")
            elif num_concerning / len(results_df) > 0.25:
                insights.append("⚠️ **Moderate Concern**: 25-50% of content flagged. Consider reviewing community guidelines.")
            else:
                insights.append("✅ **Overall Positive**: Less than 25% of content flagged as concerning.")
            
            high_risk = len(results_df[results_df['prob_concerning'] > 0.7])
            if high_risk > 0:
                insights.append(f"🔴 **{high_risk} high-risk entries** (>70% risk score) require immediate attention.")
            
            moderate_risk = len(results_df[(results_df['prob_concerning'] >= 0.30) & (results_df['prob_concerning'] < 0.40)])
            if moderate_risk > 0:
                insights.append(f"🟡 **{moderate_risk} moderate-risk entries** (30-40% risk) may benefit from contextual review.")
            
            if use_mc:
                high_uncertainty = len(results_df[results_df['uncertainty'] > 0.05])
                if high_uncertainty > 0:
                    insights.append(f"🎲 **{high_uncertainty} entries** show high prediction uncertainty - recommend manual review.")
            
            for insight in insights:
                st.markdown(f"""
                <div style='background: #1a1d23; padding: 1rem 1.25rem; border-radius: 8px; 
                            margin: 0.5rem 0; border-left: 4px solid #1463F3;'>
                    <p style='color: #c1c7d0; margin: 0; font-size: 1.05rem;'>{insight}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Clear button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Analyze Another File", type="secondary", use_container_width=True):
                st.session_state.bulk_data = None
                st.rerun()
    
    # === SINGLE TEXT ANALYSIS MODE ===
    elif model and tokenizer and user_input:
        # Perform BERT analysis
        use_mc = use_mc_dropout if 'use_mc_dropout' in locals() else False
        passes = n_passes if 'n_passes' in locals() else 20
        
        # Track latency
        start_time = time.time()
        
        if use_mc:
            with st.spinner(f'Running {passes} forward passes with Monte Carlo Dropout...'):
                result = predict_single(user_input, model, tokenizer, device, use_mc_dropout=True, n_passes=passes)
        else:
            with st.spinner('Analyzing content...'):
                result = predict_single(user_input, model, tokenizer, device, use_mc_dropout=False)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Format latency for display (more generous for large values)
        if latency_ms >= 1000:
            latency_display = f"{latency_ms/1000:.1f}s"
        else:
            latency_display = f"{latency_ms:.0f}ms"
        
        # Extract results
        prob_not_bullying = result['prob_not_bullying']
        prob_bullying = result['prob_bullying']
        uncertainty_not_bullying = result['uncertainty_not_bullying']
        uncertainty_bullying = result['uncertainty_bullying']
        prediction = result['prediction']
        mc_mode = result['mc_dropout']
        all_predictions = result['all_predictions']
        
        # Store in session state for AI Assistant
        st.session_state.last_analysis = {
            'text': user_input,
            'prob_bullying': prob_bullying,
            'prob_not_bullying': prob_not_bullying,
            'prediction': prediction,
            'uncertainty': uncertainty_bullying,
            'mc_mode': mc_mode,
            'latency_ms': latency_ms
        }
        
        # Determine prediction label and severity (using calibrated threshold)
        threshold_used = result.get('threshold_used', 0.40)
        label = "Concerning Content" if prob_bullying >= threshold_used else "Content Appears Safe"
        severity, severity_color, severity_desc = get_severity_level(prob_bullying)
        
        # === DISPLAY RESULTS ===
        st.markdown("---")
        
        # Step 3: Results
        st.markdown("""
        <div class='step-header'>
            <span class='step-indicator'>3</span>
            <span>Review Analysis Results</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("## 📊 Analysis Results")
        
        # Display performance metrics
        st.markdown(f"""
        <div style='background: #1a1d23; padding: 0.75rem 1.25rem; border-radius: 10px; 
                    margin-bottom: 1.5rem; border: 2px solid #10b981;'>
            <div style='text-align: center; margin-bottom: 0.5rem;'>
                <strong style='color: #10b981; font-size: 1.05rem;'>⚡ Analysis completed in {latency_display}</strong>
            </div>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; color: #c1c7d0; font-size: 0.9rem;'>
                <div style='text-align: center;'>
                    📊 <strong>Accuracy:</strong> 91.5% (Calibrated)
                </div>
                <div style='text-align: center;'>
                    🎪 <strong>ROC-AUC:</strong> 96.8%
                </div>
                <div style='text-align: center;'>
                    🔒 <strong>Privacy:</strong> Offline, on-device
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Main prediction
        col1, col2 = st.columns([2, 1])
        with col1:
            if label == "Concerning Content":
                st.warning(f"### ⚠️ {label}")
            else:
                st.success(f"### ✅ {label}")
        
        with col2:
            st.metric("Risk Level", f"{severity_color} {severity}", help=severity_desc)
        
        # Confidence Scores
        if mc_mode:
            st.write(f"### 🎲 Monte Carlo Dropout Results (Averaged over {passes} predictions)")
            st.markdown(f"<p style='color: #c1c7d0; font-size: 1.05rem;'>Based on {passes} independent predictions for increased reliability.</p>", unsafe_allow_html=True)
        else:
            st.write("### Confidence Scores (Model Probabilities)")
            st.markdown("<p style='color: #c1c7d0; font-size: 1.05rem;'>These scores show how confident the model is in each classification.</p>", unsafe_allow_html=True)
        
        # Custom HTML Progress bars with individual colors
        col_prog1, col_prog2 = st.columns(2)
        
        with col_prog1:
            st.markdown("<p style='color: #e8eaed; font-weight: 600; font-size: 1.05rem;'>Safe Content:</p>", unsafe_allow_html=True)
            # Custom green progress bar
            st.markdown(f"""
            <div style='background-color: #e5e7eb; border-radius: 10px; height: 20px; overflow: hidden;'>
                <div style='background-color: #047857; width: {prob_not_bullying*100}%; height: 100%; transition: width 0.3s ease;'></div>
            </div>
            """, unsafe_allow_html=True)
            if mc_mode:
                st.markdown(f"<p style='color: #c1c7d0; font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem;'>{prob_not_bullying*100:.2f}% ± {uncertainty_not_bullying*100:.2f}%</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color: #c1c7d0; font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem;'>{prob_not_bullying*100:.2f}%</p>", unsafe_allow_html=True)
        
        with col_prog2:
            st.markdown("<p style='color: #e8eaed; font-weight: 600; font-size: 1.05rem;'>Concerning Content:</p>", unsafe_allow_html=True)
            # Custom red progress bar
            st.markdown(f"""
            <div style='background-color: #e5e7eb; border-radius: 10px; height: 20px; overflow: hidden;'>
                <div style='background-color: #b91c1c; width: {prob_bullying*100}%; height: 100%; transition: width 0.3s ease;'></div>
            </div>
            """, unsafe_allow_html=True)
            if mc_mode:
                st.markdown(f"<p style='color: #c1c7d0; font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem;'>{prob_bullying*100:.2f}% ± {uncertainty_bullying*100:.2f}%</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color: #c1c7d0; font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem;'>{prob_bullying*100:.2f}%</p>", unsafe_allow_html=True)
        
        # Uncertainty metrics for MC Dropout
        if mc_mode:
            st.markdown(f"""
            <div class='info-box'>
                <strong>📊 Uncertainty Analysis:</strong><br/>
                • The ± values show the variation across {passes} independent predictions<br/>
                • <strong>Lower uncertainty</strong> = more confident and reliable prediction<br/>
                • <strong>Higher uncertainty</strong> = model is less certain, treat as borderline case<br/>
                • Average uncertainty: {((uncertainty_not_bullying + uncertainty_bullying) / 2)*100:.2f}%
            </div>
            """, unsafe_allow_html=True)
        
        # Visual Charts
        st.write("### 📊 Analysis Visualizations")
        
        # Two column layout: Risk Score and Decision Boundary
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Risk Level Gauge
            risk_score = prob_bullying * 100
            gauge_color = '#047857' if risk_score < 50 else ('#f59e0b' if risk_score < 80 else '#b91c1c')
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score", 'font': {'size': 20, 'color': '#262730', 'family': 'Inter'}},
                delta={'reference': 80, 'increasing': {'color': '#b91c1c'}, 'decreasing': {'color': '#047857'}},
                number={'suffix': "%", 'font': {'size': 40, 'color': '#262730'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#262730'},
                    'bar': {'color': gauge_color, 'thickness': 0.75},
                    'bgcolor': '#f9fafb',
                    'borderwidth': 2,
                    'bordercolor': '#e5e7eb',
                    'steps': [
                        {'range': [0, 50], 'color': '#d1fae5'},
                        {'range': [50, 80], 'color': '#fef3c7'},
                        {'range': [80, 90], 'color': '#fed7aa'},
                        {'range': [90, 100], 'color': '#fecaca'}
                    ],
                    'threshold': {
                        'line': {'color': '#262730', 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            fig_gauge.update_layout(
                height=350,
                margin=dict(t=40, b=20, l=20, r=20),
                paper_bgcolor='#ffffff',
                font=dict(color='#262730', family='Inter')
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with chart_col2:
            # Decision Boundary Visualization
            threshold = threshold_used  # Use the calibrated optimal threshold
            fig_decision = go.Figure()
            
            # Add threshold line
            fig_decision.add_shape(
                type="line",
                x0=threshold, y0=0, x1=threshold, y1=1,
                line=dict(color="#262730", width=3, dash="dash"),
            )
            
            # Add colored zones
            fig_decision.add_shape(
                type="rect",
                x0=0, y0=0, x1=threshold, y1=1,
                fillcolor="#047857", opacity=0.1, line_width=0
            )
            fig_decision.add_shape(
                type="rect",
                x0=threshold, y0=0, x1=1, y1=1,
                fillcolor="#b91c1c", opacity=0.1, line_width=0
            )
            
            # Add prediction point
            marker_color = '#b91c1c' if prob_bullying >= threshold else '#047857'
            fig_decision.add_trace(go.Scatter(
                x=[prob_bullying],
                y=[0.5],
                mode='markers+text',
                marker=dict(size=25, color=marker_color, line=dict(color='#262730', width=2)),
                text=[f'{prob_bullying*100:.1f}%'],
                textposition='top center',
                textfont=dict(size=14, color='#262730', family='Inter', weight='bold'),
                name='Prediction',
                hovertemplate=f'<b>Prediction</b><br>Concerning Probability: {prob_bullying*100:.2f}%<br>Classification: {label}<extra></extra>'
            ))
            
            # Add annotations
            fig_decision.add_annotation(
                x=threshold/2, y=0.85,
                text="Safe Zone",
                showarrow=False,
                font=dict(size=16, color='#047857', family='Inter', weight='bold')
            )
            fig_decision.add_annotation(
                x=(1+threshold)/2, y=0.85,
                text="Concerning Zone",
                showarrow=False,
                font=dict(size=16, color='#b91c1c', family='Inter', weight='bold')
            )
            fig_decision.add_annotation(
                x=threshold, y=0.15,
                text=f"Threshold<br>{threshold*100:.0f}%",
                showarrow=False,
                font=dict(size=12, color='#262730', family='Inter')
            )
            
            fig_decision.update_layout(
                title={
                    'text': 'Decision Boundary Analysis',
                    'font': {'size': 16, 'color': '#262730', 'family': 'Inter'}
                },
                xaxis=dict(
                    title="Concerning Content Probability",
                    range=[0, 1],
                    tickformat='.0%',
                    color='#262730',
                    gridcolor='#e5e7eb',
                    showgrid=True
                ),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                height=350,
                margin=dict(t=60, b=60, l=60, r=20),
                paper_bgcolor='#ffffff',
                plot_bgcolor='#ffffff',
                font=dict(color='#262730', family='Inter'),
                showlegend=False
            )
            st.plotly_chart(fig_decision, use_container_width=True)
        
        # MC Dropout visualizations
        if mc_mode and all_predictions:
            st.write("### 🎲 Monte Carlo Dropout Uncertainty Analysis")
            st.markdown(f"""
            <p style='color: #6b7280; font-size: 1rem; margin-bottom: 1.5rem;'>
                Advanced statistical analysis based on {passes} independent forward passes through the model
            </p>
            """, unsafe_allow_html=True)
            
            # Row 1: Distribution Analysis
            mc_row1_col1, mc_row1_col2 = st.columns(2)
            
            with mc_row1_col1:
                # Enhanced Histogram with KDE overlay
                fig_hist = go.Figure()
                
                # Histogram bars
                fig_hist.add_trace(go.Histogram(
                    x=all_predictions['not_bullying'],
                    name='Safe Content',
                    marker_color='#047857',
                    opacity=0.7,
                    nbinsx=25,
                    hovertemplate='Probability: %{x:.3f}<br>Count: %{y}<extra></extra>'
                ))
                fig_hist.add_trace(go.Histogram(
                    x=all_predictions['bullying'],
                    name='Concerning Content',
                    marker_color='#b91c1c',
                    opacity=0.7,
                    nbinsx=25,
                    hovertemplate='Probability: %{x:.3f}<br>Count: %{y}<extra></extra>'
                ))
                
                # Add mean lines
                fig_hist.add_vline(
                    x=prob_not_bullying,
                    line_dash="dash",
                    line_color="#047857",
                    line_width=2,
                    annotation_text=f"Safe Mean: {prob_not_bullying*100:.1f}%",
                    annotation_position="top"
                )
                fig_hist.add_vline(
                    x=prob_bullying,
                    line_dash="dash",
                    line_color="#b91c1c",
                    line_width=2,
                    annotation_text=f"Concerning Mean: {prob_bullying*100:.1f}%",
                    annotation_position="top"
                )
                
                fig_hist.update_layout(
                    title={
                        'text': 'Prediction Distribution Histogram',
                        'font': {'size': 16, 'color': '#262730', 'family': 'Inter'}
                    },
                    barmode='overlay',
                    xaxis_title='Probability',
                    yaxis_title='Frequency',
                    height=350,
                    showlegend=True,
                    hovermode='x unified',
                    paper_bgcolor='#ffffff',
                    plot_bgcolor='#ffffff',
                    font=dict(color='#262730', family='Inter', size=13),
                    xaxis=dict(
                        color='#262730',
                        gridcolor='#e5e7eb',
                        showgrid=True,
                        tickformat='.0%'
                    ),
                    yaxis=dict(color='#262730', gridcolor='#e5e7eb', showgrid=True),
                    legend=dict(
                        font=dict(size=13, color='#262730'),
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with mc_row1_col2:
                # Enhanced Box Plot with detailed statistics
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=all_predictions['not_bullying'],
                    name='Safe Content',
                    marker=dict(
                        color='#047857',
                        line=dict(color='#262730', width=1.5)
                    ),
                    boxmean='sd',
                    fillcolor='rgba(4, 120, 87, 0.5)',
                    line=dict(color='#047857', width=2),
                    hovertemplate='<b>Safe Content</b><br>Value: %{y:.3f}<extra></extra>'
                ))
                fig_box.add_trace(go.Box(
                    y=all_predictions['bullying'],
                    name='Concerning Content',
                    marker=dict(
                        color='#b91c1c',
                        line=dict(color='#262730', width=1.5)
                    ),
                    boxmean='sd',
                    fillcolor='rgba(185, 28, 28, 0.5)',
                    line=dict(color='#b91c1c', width=2),
                    hovertemplate='<b>Concerning Content</b><br>Value: %{y:.3f}<extra></extra>'
                ))
                
                # Add threshold line
                fig_box.add_hline(
                    y=threshold_used,
                    line_dash="dash",
                    line_color="#262730",
                    line_width=2,
                    annotation_text=f"Decision Threshold ({threshold_used*100:.0f}%)",
                    annotation_position="right"
                )
                
                fig_box.update_layout(
                    title={
                        'text': 'Statistical Distribution Box Plot',
                        'font': {'size': 16, 'color': '#262730', 'family': 'Inter'}
                    },
                    yaxis_title='Probability',
                    height=350,
                    showlegend=True,
                    paper_bgcolor='#ffffff',
                    plot_bgcolor='#ffffff',
                    font=dict(color='#262730', family='Inter', size=13),
                    xaxis=dict(color='#262730'),
                    yaxis=dict(
                        color='#262730',
                        gridcolor='#e5e7eb',
                        showgrid=True,
                        tickformat='.0%',
                        range=[0, 1]
                    ),
                    legend=dict(
                        font=dict(size=13, color='#262730'),
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Row 2: Violin Plot and Statistics
            mc_row2_col1, mc_row2_col2 = st.columns([1.2, 0.8])
            
            with mc_row2_col1:
                # Violin Plot for detailed distribution shape
                fig_violin = go.Figure()
                
                fig_violin.add_trace(go.Violin(
                    y=all_predictions['not_bullying'],
                    name='Safe Content',
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor='rgba(4, 120, 87, 0.5)',
                    line_color='#047857',
                    marker=dict(color='#047857', line=dict(color='#262730', width=1)),
                    hovertemplate='<b>Safe Content</b><br>Probability: %{y:.3f}<extra></extra>',
                    points='outliers'
                ))
                
                fig_violin.add_trace(go.Violin(
                    y=all_predictions['bullying'],
                    name='Concerning Content',
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor='rgba(185, 28, 28, 0.5)',
                    line_color='#b91c1c',
                    marker=dict(color='#b91c1c', line=dict(color='#262730', width=1)),
                    hovertemplate='<b>Concerning Content</b><br>Probability: %{y:.3f}<extra></extra>',
                    points='outliers'
                ))
                
                fig_violin.update_layout(
                    title={
                        'text': 'Violin Plot - Distribution Shape Analysis',
                        'font': {'size': 16, 'color': '#262730', 'family': 'Inter'}
                    },
                    yaxis_title='Probability',
                    height=350,
                    showlegend=True,
                    paper_bgcolor='#ffffff',
                    plot_bgcolor='#ffffff',
                    font=dict(color='#262730', family='Inter', size=13),
                    xaxis=dict(color='#262730'),
                    yaxis=dict(
                        color='#262730',
                        gridcolor='#e5e7eb',
                        showgrid=True,
                        tickformat='.0%',
                        range=[0, 1]
                    ),
                    violinmode='group',
                    legend=dict(
                        font=dict(size=13, color='#262730'),
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig_violin, use_container_width=True)
            
            with mc_row2_col2:
                # Detailed Statistics Panel
                safe_min = min(all_predictions['not_bullying'])
                safe_max = max(all_predictions['not_bullying'])
                safe_q1 = np.percentile(all_predictions['not_bullying'], 25)
                safe_q3 = np.percentile(all_predictions['not_bullying'], 75)
                safe_median = np.median(all_predictions['not_bullying'])
                
                conc_min = min(all_predictions['bullying'])
                conc_max = max(all_predictions['bullying'])
                conc_q1 = np.percentile(all_predictions['bullying'], 25)
                conc_q3 = np.percentile(all_predictions['bullying'], 75)
                conc_median = np.median(all_predictions['bullying'])
                
                st.markdown("""
                <div style='background: #1a1d23; padding: 1.25rem; border-radius: 8px; border: 2px solid #2d3139; height: 100%;'>
                    <h4 style='color: #e8eaed; font-family: Inter; margin-top: 0; margin-bottom: 1rem; font-size: 1.1rem;'>📊 Statistical Summary</h4>
                """, unsafe_allow_html=True)
                
                # Safe Content stats
                st.markdown("<p style='margin: 0 0 0.5rem 0;'><strong style='color: #10b981; font-size: 1rem;'>Safe Content:</strong></p>", unsafe_allow_html=True)
                st.markdown(f"""
                <ul style='margin: 0 0 1rem 0; padding-left: 1.25rem; list-style: disc;'>
                    <li>Mean: {prob_not_bullying*100:.2f}%</li>
                    <li>Median: {safe_median*100:.2f}%</li>
                    <li>Std Dev: ±{uncertainty_not_bullying*100:.2f}%</li>
                    <li>Q1-Q3: {safe_q1*100:.2f}% - {safe_q3*100:.2f}%</li>
                    <li>Range: {safe_min*100:.2f}% - {safe_max*100:.2f}%</li>
                </ul>
                """, unsafe_allow_html=True)
                
                # Concerning Content stats
                st.markdown("<p style='margin: 0 0 0.5rem 0;'><strong style='color: #ef4444; font-size: 1rem;'>Concerning Content:</strong></p>", unsafe_allow_html=True)
                st.markdown(f"""
                <ul style='margin: 0; padding-left: 1.25rem; list-style: disc;'>
                    <li>Mean: {prob_bullying*100:.2f}%</li>
                    <li>Median: {conc_median*100:.2f}%</li>
                    <li>Std Dev: ±{uncertainty_bullying*100:.2f}%</li>
                    <li>Q1-Q3: {conc_q1*100:.2f}% - {conc_q3*100:.2f}%</li>
                    <li>Range: {conc_min*100:.2f}% - {conc_max*100:.2f}%</li>
                </ul>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # ===== AI ASSISTANT PANEL =====
        if st.session_state.get('openai_api_key'):
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; margin: 2rem 0 1.5rem 0;'>
                <h2 style='font-size: 2rem; color: #e8eaed;'>🤖 Optional AI Assistant</h2>
                <p style='font-size: 1.05rem; color: #c1c7d0;'>
                    Get AI-powered explanations and recommendations (uses GPT-4o)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            ai_col1, ai_col2, ai_col3 = st.columns(3)
            
            with ai_col1:
                if st.button("💬 Explain This Result", key="btn_explain", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing the result..."):
                        explanation = explain_detection(
                            user_input, label, prob_bullying, prob_not_bullying, severity
                        )
                    if explanation:
                        st.markdown("""
                        <div class='custom-card' style='margin-top: 1rem; border-color: #1463F3;'>
                            <h4 style='color: #84A4FC; margin-bottom: 1rem;'>💬 AI Explanation</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"<div style='color: #c1c7d0; line-height: 1.7;'>{explanation}</div>", unsafe_allow_html=True)
            
            with ai_col2:
                if st.button("📋 Get Recommendations", key="btn_recommend", use_container_width=True):
                    with st.spinner("🤖 AI is generating recommendations..."):
                        recommendations = get_recommendations(
                            user_input, label, prob_bullying, severity
                        )
                    if recommendations:
                        st.markdown("""
                        <div class='custom-card' style='margin-top: 1rem; border-color: #10b981;'>
                            <h4 style='color: #10b981; margin-bottom: 1rem;'>📋 Recommended Actions</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"<div style='color: #c1c7d0; line-height: 1.7;'>{recommendations}</div>", unsafe_allow_html=True)
            
            with ai_col3:
                if prob_bullying >= 0.30:  # Only show for moderate-to-high risk content
                    if st.button("✨ Improve Content", key="btn_improve", use_container_width=True):
                        with st.spinner("🤖 AI is suggesting improvements..."):
                            improvements = suggest_improvements(user_input, prob_bullying)
                        if improvements:
                            st.markdown("""
                            <div class='custom-card' style='margin-top: 1rem; border-color: #f59e0b;'>
                                <h4 style='color: #f59e0b; margin-bottom: 1rem;'>✨ Content Improvement Suggestions</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"<div style='color: #c1c7d0; line-height: 1.7;'>{improvements}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='text-align: center; padding: 1rem; color: #6b7280; font-size: 0.9rem;'>
                        Content appears safe.<br/>No improvements needed.
                    </div>
                    """, unsafe_allow_html=True)
            
            # AI Verification for high uncertainty cases
            if mc_mode and uncertainty_bullying > 0.05:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='info-box' style='border-left-color: #f59e0b;'>
                    <strong>⚠️ High Uncertainty Detected</strong><br/>
                    This prediction has significant uncertainty ({uncertainty_bullying*100:.2f}%). 
                    Use AI verification for contextual analysis and a second opinion.
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔍 Verify with AI", key="btn_verify", use_container_width=False):
                    with st.spinner("🤖 AI is performing contextual verification..."):
                        verification = verify_with_gpt(user_input, prediction, uncertainty_bullying)
                    
                    if verification:
                        st.markdown("""
                        <div class='custom-card' style='border-color: #1463F3;'>
                            <h4 style='color: #84A4FC; margin-bottom: 1rem;'>🔍 AI Verification Result</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        verified_label = "Concerning" if verification['verified_prediction'] == 1 else "Safe"
                        st.markdown(f"""
                        <div style='color: #c1c7d0; line-height: 1.8;'>
                            <strong style='color: #e8eaed;'>AI Verdict:</strong> {verified_label}<br/>
                            <strong style='color: #e8eaed;'>Confidence:</strong> {verification['confidence']*100:.1f}%<br/>
                            <strong style='color: #e8eaed;'>Reasoning:</strong> {verification['reasoning']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show combined result
                        if verification['verified_prediction'] == prediction:
                            st.success("✅ AI verification agrees with BERT prediction")
                        else:
                            st.warning("⚠️ AI verification disagrees with BERT - recommend human review")
            
            # === CUSTOM AI CHAT ===
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align: center; margin: 1.5rem 0 1rem 0;'>
                <h3 style='font-size: 1.5rem; color: #e8eaed;'>💬 Ask the AI Assistant</h3>
                <p style='font-size: 0.95rem; color: #c1c7d0;'>
                    Ask questions about this analysis, content moderation, or how the model works
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize chat session state
            if 'ai_chat_history' not in st.session_state:
                st.session_state.ai_chat_history = []
            
            # Chat input
            user_question = st.text_input(
                "Your question:",
                placeholder="e.g., 'Why was this flagged?', 'What should I do next?', 'How does the model work?'",
                key="ai_chat_input",
                help="Ask anything about the analysis, content moderation best practices, or how Equity Ally works"
            )
            
            col_ask, col_clear = st.columns([4, 1])
            
            with col_ask:
                if st.button("🚀 Ask AI", key="btn_ask_ai", use_container_width=True, type="primary"):
                    if user_question and user_question.strip():
                        with st.spinner("🤖 AI is thinking..."):
                            # Add context about current analysis
                            context = f"Current analysis: Text analyzed with {prob_bullying*100:.1f}% concerning probability, classified as '{label}', severity: {severity}"
                            
                            answer = chat_about_model(user_question, context=context)
                        
                        if answer:
                            # Add to chat history
                            st.session_state.ai_chat_history.append({
                                'question': user_question,
                                'answer': answer
                            })
                            
                            # Clear input by rerunning
                            st.rerun()
                    else:
                        st.warning("⚠️ Please enter a question first")
            
            with col_clear:
                if st.button("🗑️ Clear", key="btn_clear_chat", use_container_width=True):
                    st.session_state.ai_chat_history = []
                    st.rerun()
            
            # Display chat history (most recent first)
            if st.session_state.ai_chat_history:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📜 Conversation History")
                
                for i, chat in enumerate(reversed(st.session_state.ai_chat_history)):
                    st.markdown(f"""
                    <div class='custom-card' style='margin-bottom: 1rem; border-color: #1463F3;'>
                        <div style='margin-bottom: 0.75rem;'>
                            <strong style='color: #84A4FC; font-size: 1rem;'>❓ Your Question:</strong>
                            <p style='color: #e8eaed; margin: 0.5rem 0; line-height: 1.6;'>{chat['question']}</p>
                        </div>
                        <div>
                            <strong style='color: #10b981; font-size: 1rem;'>🤖 AI Response:</strong>
                            <p style='color: #c1c7d0; margin: 0.5rem 0; line-height: 1.7;'>{chat['answer']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # === NEXT STEPS ===
        st.markdown("---")
        st.write("## 💡 Recommended Next Steps")
        
        if prob_bullying < 0.30:
            st.markdown("""
            <div class='info-box' style='border-left-color: #047857;'>
                <strong style='color: #e8eaed;'>✅ No Action Required</strong><br/><br/>
                This content appears safe and appropriate. No moderation action is needed at this time.<br/><br/>
                <strong>Suggestions:</strong><br/>
                • Continue to monitor for context if this is part of a larger conversation<br/>
                • Encourage this type of positive, constructive communication
            </div>
            """, unsafe_allow_html=True)
        elif prob_bullying < threshold_used:
            st.markdown("""
            <div class='info-box' style='border-left-color: #f59e0b;'>
                <strong style='color: #e8eaed;'>⚠️ Review Recommended</strong><br/><br/>
                This content falls in a moderate concern category. Consider the context and intent before taking action.<br/><br/>
                <strong>Recommended Actions:</strong><br/>
                • Review the full conversation context if available<br/>
                • Consider if cultural or linguistic nuances might affect interpretation<br/>
                • Flag for manual review by a human moderator if uncertain<br/>
                • Document for pattern analysis if from a repeat source<br/><br/>
                <strong>Remember:</strong> Context matters. Sarcasm, quotes, and educational discussions may be flagged.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='info-box' style='border-left-color: #dc2626;'>
                <strong style='color: #e8eaed;'>🚨 Immediate Review Recommended</strong><br/><br/>
                This content has a high probability of containing harmful or inappropriate material.<br/><br/>
                <strong>Recommended Actions:</strong><br/>
                • Prioritize for immediate human moderator review<br/>
                • Consider temporary content flagging or hiding pending review<br/>
                • Review user history for patterns of concerning behavior<br/>
                • Document for compliance and safety records<br/>
                • Consider reaching out with educational resources about community guidelines<br/><br/>
                <strong>Important:</strong> Even high-confidence predictions can be wrong. Always use human judgment for final decisions.
            </div>
            """, unsafe_allow_html=True)
        
        # Universal guidance
        st.markdown("""
        <div style='color: #c1c7d0; line-height: 1.7; margin-top: 1.5rem; padding: 1.25rem; background: #1a1d23; border-radius: 8px;'>
        <strong style='color: #e8eaed; font-size: 1.05rem;'>🌟 Best Practices for Content Moderation:</strong><br/><br/>
        
        1. <strong>Human oversight is essential</strong> - AI is a tool to assist, not replace human judgment<br/>
        2. <strong>Consider context</strong> - Look at conversation history, relationships, and cultural factors<br/>
        3. <strong>Be consistent</strong> - Apply guidelines fairly across all users and content<br/>
        4. <strong>Educate users</strong> - Focus on teaching positive communication rather than just punishing<br/>
        5. <strong>Review borderline cases</strong> - When in doubt, have a second moderator review<br/>
        6. <strong>Monitor for bias</strong> - Be aware that AI models can reflect biases in training data
        </div>
        """, unsafe_allow_html=True)
        
        # Clear button
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("🔄 Analyze Another Text", type="secondary", use_container_width=True):
                st.rerun()

    elif not user_input:
        st.markdown("""
        <div class='info-box' style='border-left-color: #f59e0b; text-align: center;'>
            <strong style='color: #e8eaed; font-size: 1.1rem;'>📝 Ready to Analyze</strong><br/><br/>
            <span style='color: #c1c7d0;'>Please enter or upload some text content above to begin the analysis.</span>
        </div>
        """, unsafe_allow_html=True)

# Page navigation
page_navigation()

# Navigation footer
navigation_footer()


