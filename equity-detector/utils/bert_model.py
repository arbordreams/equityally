"""
BERT model loading and inference utilities for Equity Ally
Centralized model management for use across multiple pages
Includes isotonic calibration for optimal performance (96.8% accuracy)
"""

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import os
import sys
import json
from pathlib import Path


def get_model_path():
    """
    Robustly determine the model path across different environments
    Tries multiple strategies to find the BERT model directory
    """
    # Strategy 1: Relative to this file (works locally and most deployments)
    try:
        current_file = Path(__file__).resolve()
        # This file is at: equity-detector/utils/bert_model.py
        # Model is at: equity-detector/BERT_Model
        model_path = current_file.parent.parent / "BERT_Model"
        if model_path.exists() and (model_path / "config.json").exists():
            return str(model_path)
    except Exception as e:
        st.warning(f"Strategy 1 failed: {e}")
    
    # Strategy 2: Relative to current working directory
    try:
        cwd = Path.cwd()
        # Try: cwd/equity-detector/BERT_Model
        model_path = cwd / "equity-detector" / "BERT_Model"
        if model_path.exists() and (model_path / "config.json").exists():
            return str(model_path)
        
        # Try: cwd/BERT_Model
        model_path = cwd / "BERT_Model"
        if model_path.exists() and (model_path / "config.json").exists():
            return str(model_path)
    except Exception as e:
        st.warning(f"Strategy 2 failed: {e}")
    
    # Strategy 3: Search in parent directories
    try:
        current = Path(__file__).resolve()
        for _ in range(5):  # Search up to 5 levels
            current = current.parent
            model_path = current / "BERT_Model"
            if model_path.exists() and (model_path / "config.json").exists():
                return str(model_path)
            
            # Also try with equity-detector prefix
            model_path = current / "equity-detector" / "BERT_Model"
            if model_path.exists() and (model_path / "config.json").exists():
                return str(model_path)
    except Exception as e:
        st.warning(f"Strategy 3 failed: {e}")
    
    # Strategy 4: Absolute path from environment or default
    try:
        # Check if there's an environment variable
        if 'MODEL_PATH' in os.environ:
            model_path = Path(os.environ['MODEL_PATH'])
            if model_path.exists() and (model_path / "config.json").exists():
                return str(model_path)
    except Exception as e:
        st.warning(f"Strategy 4 failed: {e}")
    
    # If all strategies fail, return a default path for error messaging
    return str(Path(__file__).resolve().parent.parent / "BERT_Model")


# Get model path using robust strategy
MODEL_PATH = get_model_path()

# Isotonic calibration configuration
CALIBRATION_ENABLED = True  # Enable isotonic calibration for best performance
OPTIMAL_THRESHOLD = 0.40  # F1-optimal threshold for calibrated predictions (updated from evaluation)


@st.cache_resource
def load_isotonic_calibration():
    """
    Load isotonic calibration model for probability calibration
    
    Returns:
        dict: Calibration data with x/y points for interpolation, or None if not found
    """
    try:
        # Find calibration file relative to this script
        current_dir = Path(__file__).resolve().parent.parent
        calibration_path = current_dir / "evaluation" / "isotonic_calibration.json"
        
        if not calibration_path.exists():
            st.warning(f"⚠️ Calibration file not found at {calibration_path}. Using uncalibrated predictions.")
            return None
        
        with open(calibration_path, 'r') as f:
            calibration_data = json.load(f)
        
        # Convert lists to numpy arrays for faster interpolation
        calibration_data['x_calibration'] = np.array(calibration_data['x_calibration'])
        calibration_data['y_calibration'] = np.array(calibration_data['y_calibration'])
        
        # Only show success message once (cache_resource handles this)
        # Removed st.success to prevent spam during bulk analysis
        
        return calibration_data
    
    except Exception as e:
        st.warning(f"⚠️ Could not load calibration: {e}. Using uncalibrated predictions.")
        return None


def apply_calibration(probs, calibration_data):
    """
    Apply isotonic calibration to probabilities
    
    Args:
        probs: Uncalibrated probability or array of probabilities
        calibration_data: Calibration data from load_isotonic_calibration()
    
    Returns:
        Calibrated probability/probabilities
    """
    if calibration_data is None or not CALIBRATION_ENABLED:
        return probs
    
    try:
        # Use numpy interpolation for fast calibration
        probs_calibrated = np.interp(
            probs,
            calibration_data['x_calibration'],
            calibration_data['y_calibration']
        )
        return probs_calibrated
    except Exception as e:
        st.warning(f"Calibration failed: {e}. Using uncalibrated probabilities.")
        return probs


@st.cache_resource
def load_model():
    """
    Load BERT model and tokenizer with caching for performance
    
    Returns:
        tuple: (model, tokenizer, device) or (None, None, None) if error
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Add custom CSS to make status boxes more compact
    st.markdown("""
        <style>
        .stAlert {
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            margin-top: 0.25rem !important;
            margin-bottom: 0.25rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Comprehensive diagnostics
    with st.expander("🔍 Model Loading Diagnostics", expanded=False):
        st.write(f"**MODEL_PATH**: `{MODEL_PATH}`")
        st.write(f"**Current Working Directory**: `{Path.cwd()}`")
        st.write(f"**This File Location**: `{Path(__file__).resolve()}`")
        st.write(f"**Model Path Exists**: {Path(MODEL_PATH).exists()}")
        
        if Path(MODEL_PATH).exists():
            model_files = list(Path(MODEL_PATH).iterdir())
            st.write(f"**Files in Model Directory** ({len(model_files)} files):")
            for f in sorted(model_files):
                file_size = f.stat().st_size if f.is_file() else 0
                st.write(f"  - {f.name} ({file_size:,} bytes)")
            
            # Check for required files
            required_files = ['config.json', 'vocab.txt', 'tokenizer_config.json']
            model_file_exists = (Path(MODEL_PATH) / 'model.safetensors').exists() or \
                               (Path(MODEL_PATH) / 'pytorch_model.bin').exists()
            
            missing_files = [f for f in required_files if not (Path(MODEL_PATH) / f).exists()]
            if missing_files:
                st.error(f"❌ Missing required files: {', '.join(missing_files)}")
            if not model_file_exists:
                st.error("❌ Missing model weights file (model.safetensors or pytorch_model.bin)")
                
                # Check if it's a Git LFS pointer
                safetensors_path = Path(MODEL_PATH) / 'model.safetensors'
                if safetensors_path.exists() and safetensors_path.stat().st_size < 1000:
                    st.error("⚠️ model.safetensors appears to be a Git LFS pointer file, not the actual model!")
                    st.code(safetensors_path.read_text()[:500])
            else:
                st.success("✅ All required files present")
        else:
            st.error(f"❌ Model directory not found at: {MODEL_PATH}")
            # Show what directories do exist
            parent_path = Path(MODEL_PATH).parent
            if parent_path.exists():
                st.write(f"**Contents of parent directory** `{parent_path}`:")
                for item in sorted(parent_path.iterdir()):
                    st.write(f"  - {item.name}{'/' if item.is_dir() else ''}")
    
    # Attempt to load model
    try:
        model_path = Path(MODEL_PATH)
        
        # Verify essential files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
        
        config_file = model_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found in {MODEL_PATH}")
        
        # Check for model weights
        safetensors_file = model_path / "model.safetensors"
        pytorch_file = model_path / "pytorch_model.bin"
        
        if safetensors_file.exists():
            # Check if it's a Git LFS pointer (small file)
            if safetensors_file.stat().st_size < 10000:  # Less than 10KB is likely a pointer
                raise ValueError(
                    "model.safetensors appears to be a Git LFS pointer file. "
                    "The actual model file was not downloaded. "
                    "Please ensure Git LFS is properly configured and the model file is pulled."
                )
        elif not pytorch_file.exists():
            raise FileNotFoundError(
                f"Model weights not found. Expected either model.safetensors or pytorch_model.bin in {MODEL_PATH}"
            )
        
        st.info(f"📥 Loading model from: {MODEL_PATH}")
        
        # Try loading with local_files_only first
        try:
            model = BertForSequenceClassification.from_pretrained(
                MODEL_PATH, 
                local_files_only=True,
                torch_dtype=torch.float32
            )
            tokenizer = BertTokenizer.from_pretrained(
                MODEL_PATH, 
                local_files_only=True
            )
        except Exception as e:
            # If local_files_only fails, try without it (allows downloading config if needed)
            st.warning(f"Local-only loading failed, trying with network access: {e}")
            model = BertForSequenceClassification.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float32
            )
            tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        
        model.to(device)
        model.eval()
        
        st.success(f"✅ Model loaded successfully on {device}")
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"❌ Unable to load the AI model")
        st.error(f"**Error**: {str(e)}")
        
        # Show full traceback in expander
        with st.expander("📋 Full Error Traceback", expanded=False):
            import traceback
            st.code(traceback.format_exc())
        
        st.info("""
        💡 **Troubleshooting Tips:**
        1. Ensure all model files are properly committed to Git
        2. If using Git LFS, verify the model file was actually downloaded (not just a pointer)
        3. Check that the model directory structure is correct
        4. Verify file permissions allow reading the model files
        """)
        
        return None, None, None


def enable_dropout(model):
    """
    Enable dropout layers during inference for Monte Carlo Dropout
    
    Args:
        model: The BERT model
    """
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def mc_dropout_predict(text, model, tokenizer, device, n_passes=20):
    """
    Perform Monte Carlo Dropout inference for more robust predictions
    
    Args:
        text: Input text to analyze
        model: BERT model
        tokenizer: BERT tokenizer
        device: torch device
        n_passes: Number of forward passes (default 20)
    
    Returns:
        tuple: (mean_prob_not_bullying, mean_prob_bullying, 
                std_prob_not_bullying, std_prob_bullying, all_predictions)
    """
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_tensors='pt',
        truncation=True
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Enable dropout for MC Dropout
    enable_dropout(model)
    
    predictions_not_bullying = []
    predictions_bullying = []
    
    with torch.no_grad():
        for _ in range(n_passes):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]
            predictions_not_bullying.append(probabilities[0].item())
            predictions_bullying.append(probabilities[1].item())
    
    # Set model back to eval mode
    model.eval()
    
    mean_prob_not_bullying = np.mean(predictions_not_bullying)
    mean_prob_bullying = np.mean(predictions_bullying)
    std_prob_not_bullying = np.std(predictions_not_bullying)
    std_prob_bullying = np.std(predictions_bullying)
    
    all_predictions = {
        'not_bullying': predictions_not_bullying,
        'bullying': predictions_bullying
    }
    
    return mean_prob_not_bullying, mean_prob_bullying, std_prob_not_bullying, std_prob_bullying, all_predictions


def predict_single(text, model, tokenizer, device, use_mc_dropout=False, n_passes=20):
    """
    Make a single prediction on text with isotonic calibration
    
    Args:
        text: Input text
        model: BERT model
        tokenizer: BERT tokenizer
        device: torch device
        use_mc_dropout: Whether to use Monte Carlo Dropout
        n_passes: Number of MC Dropout passes if enabled
        
    Returns:
        dict: Prediction results with calibrated probabilities and uncertainty
    """
    # Load calibration model
    calibration_data = load_isotonic_calibration()
    
    if use_mc_dropout:
        prob_not_bullying, prob_bullying, uncertainty_not_bullying, uncertainty_bullying, all_preds = mc_dropout_predict(
            text, model, tokenizer, device, n_passes=n_passes
        )
        
        # Apply calibration to the mean probability
        prob_bullying_cal = apply_calibration(prob_bullying, calibration_data)
        prob_not_bullying_cal = 1.0 - prob_bullying_cal
        
        return {
            'prob_not_bullying': prob_not_bullying_cal,
            'prob_bullying': prob_bullying_cal,
            'prob_not_bullying_uncal': prob_not_bullying,
            'prob_bullying_uncal': prob_bullying,
            'uncertainty_not_bullying': uncertainty_not_bullying,
            'uncertainty_bullying': uncertainty_bullying,
            'prediction': 1 if prob_bullying_cal >= OPTIMAL_THRESHOLD else 0,
            'mc_dropout': True,
            'all_predictions': all_preds,
            'calibrated': calibration_data is not None,
            'threshold_used': OPTIMAL_THRESHOLD
        }
    else:
        # Standard prediction
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        prob_not_bullying = probabilities[0].item()
        prob_bullying = probabilities[1].item()
        
        # Apply isotonic calibration
        prob_bullying_cal = apply_calibration(prob_bullying, calibration_data)
        prob_not_bullying_cal = 1.0 - prob_bullying_cal
        
        return {
            'prob_not_bullying': prob_not_bullying_cal,
            'prob_bullying': prob_bullying_cal,
            'prob_not_bullying_uncal': prob_not_bullying,
            'prob_bullying_uncal': prob_bullying,
            'uncertainty_not_bullying': 0.0,
            'uncertainty_bullying': 0.0,
            'prediction': 1 if prob_bullying_cal >= OPTIMAL_THRESHOLD else 0,
            'mc_dropout': False,
            'all_predictions': None,
            'calibrated': calibration_data is not None,
            'threshold_used': OPTIMAL_THRESHOLD
        }


def predict_batch(texts, model, tokenizer, device, use_mc_dropout=False, n_passes=20):
    """
    Make predictions for a batch of texts
    
    Args:
        texts: List of input texts
        model: BERT model
        tokenizer: BERT tokenizer
        device: torch device
        use_mc_dropout: Whether to use Monte Carlo Dropout
        n_passes: Number of MC Dropout passes if enabled
    
    Returns:
        tuple: (predictions, probabilities, uncertainties)
    """
    predictions = []
    probabilities = []
    uncertainties = [] if use_mc_dropout else None
    
    for text in texts:
        result = predict_single(text, model, tokenizer, device, use_mc_dropout, n_passes)
        predictions.append(result['prediction'])
        probabilities.append(result['prob_bullying'])
        if use_mc_dropout:
            uncertainties.append(result['uncertainty_bullying'])
    
    return predictions, probabilities, uncertainties


def get_severity_level(prob_bullying):
    """
    Determine severity level based on probability
    
    Args:
        prob_bullying: Probability of concerning content (0-1)
        
    Returns:
        tuple: (severity_name, severity_color, severity_description)
    """
    if prob_bullying < 0.3:
        return "Low Concern", "🟢", "Content appears safe"
    elif prob_bullying < 0.7:
        return "Moderate Concern", "🟡", "May contain concerning elements"
    else:
        return "High Concern", "🔴", "Likely contains concerning content"


def calculate_mapk(y_true, y_pred_proba, k=1):
    """
    Calculate Mean Average Precision @ K for binary classification
    
    Args:
        y_true: Array of true labels (0 or 1)
        y_pred_proba: Array of predicted probabilities for class 1
        k: Number of top predictions to consider
    
    Returns:
        float: MAP@K score
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0
    
    average_precisions = []
    
    for i in range(n_samples):
        true_label = y_true[i]
        pred_prob = y_pred_proba[i]
        
        # For binary classification, rank by confidence
        if k >= 1:
            predicted_class = 1 if pred_prob >= 0.5 else 0
            if predicted_class == true_label:
                average_precisions.append(1.0)
            else:
                average_precisions.append(0.0)
    
    return np.mean(average_precisions) if average_precisions else 0.0


