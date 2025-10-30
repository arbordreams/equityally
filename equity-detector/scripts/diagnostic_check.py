#!/usr/bin/env python3
"""
Comprehensive Diagnostic Check
===============================
Verifies model, data, and configuration before full evaluation.
"""

import sys
from pathlib import Path
import json

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

BASE_DIR = Path("/Users/seb/Desktop/EquityLens/equity-detector")
MODEL_DIR = BASE_DIR / "BERT_Model"
DATA_DIR = BASE_DIR / "equity-training-datasets"

def check_model():
    """Check model configuration"""
    print("="*80)
    print("MODEL DIAGNOSTIC")
    print("="*80)
    
    # Load config
    config_path = MODEL_DIR / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"\nModel Configuration:")
    print(f"  Architecture: {config.get('architectures', ['Unknown'])[0]}")
    print(f"  Problem Type: {config.get('problem_type', 'Unknown')}")
    print(f"  Hidden Size: {config.get('hidden_size', 0)}")
    print(f"  Num Labels: {config.get('num_labels', 'Not specified')}")
    print(f"  Vocab Size: {config.get('vocab_size', 0)}")
    
    # Load model
    print(f"\nLoading model from {MODEL_DIR}...")
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    
    # Check model outputs
    print(f"\nModel Structure:")
    print(f"  Classifier output features: {model.classifier.out_features}")
    print(f"  Expected labels: {model.num_labels}")
    
    # Test inference
    print(f"\nTesting inference...")
    test_text = "This is a test comment"
    inputs = tokenizer(test_text, return_tensors="pt", max_length=256, truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    print(f"  Input shape: {inputs['input_ids'].shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits: {logits.numpy()}")
    
    num_output_classes = logits.shape[1]
    print(f"\n✓ Model outputs {num_output_classes} classes")
    
    return num_output_classes

def check_data():
    """Check data files"""
    print("\n" + "="*80)
    print("DATA DIAGNOSTIC")
    print("="*80)
    
    # Check if CSVs exist
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    test_labels_path = DATA_DIR / "test_labels.csv"
    
    for path in [train_path, test_path, test_labels_path]:
        if not path.exists():
            print(f"❌ Missing: {path}")
            return False
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ Found: {path.name} ({size_mb:.1f} MB)")
    
    # Load and inspect
    print(f"\nLoading data samples...")
    train_df = pd.read_csv(train_path, nrows=100)
    test_df = pd.read_csv(test_path, nrows=100)
    test_labels_df = pd.read_csv(test_labels_path, nrows=100)
    
    print(f"\nTrain columns: {list(train_df.columns)}")
    print(f"Test columns: {list(test_df.columns)}")
    print(f"Test labels columns: {list(test_labels_df.columns)}")
    
    # Check label columns
    all_label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    available_labels = [col for col in all_label_cols if col in train_df.columns]
    
    print(f"\nAvailable labels: {available_labels}")
    
    # Check text column
    if 'comment_text' in train_df.columns:
        print(f"✓ Text column 'comment_text' found")
        print(f"\nSample text: {train_df['comment_text'].iloc[0][:100]}...")
    else:
        print(f"❌ Text column 'comment_text' not found")
        return False
    
    # Check label prevalence
    print(f"\nLabel prevalence in sample:")
    for label in available_labels:
        if label in train_df.columns:
            count = train_df[label].sum()
            total = len(train_df)
            print(f"  {label:15s}: {count}/{total} ({100*count/total:.1f}%)")
    
    return available_labels

def recommend_configuration(num_classes, available_labels):
    """Recommend configuration based on model and data"""
    print("\n" + "="*80)
    print("CONFIGURATION RECOMMENDATION")
    print("="*80)
    
    if num_classes == 2:
        print(f"\n✓ Model is BINARY classification (2 classes)")
        print(f"  Recommendation: Use only 'toxic' label")
        print(f"  Config: LABEL_COLS = ['toxic']")
        return ["toxic"]
    
    elif num_classes == 6:
        print(f"\n✓ Model is MULTI-LABEL classification (6 classes)")
        print(f"  Recommendation: Use all 6 labels")
        print(f"  Config: LABEL_COLS = {available_labels[:6]}")
        return available_labels[:6]
    
    else:
        print(f"\n⚠️  Unexpected number of classes: {num_classes}")
        print(f"  Using first {num_classes} labels")
        return available_labels[:num_classes]

def check_scripts():
    """Check scripts for common issues"""
    print("\n" + "="*80)
    print("SCRIPT DIAGNOSTIC")
    print("="*80)
    
    scripts_to_check = [
        "run_evaluation_pipeline.py",
        "run_tokenizer_and_summary.py",
        "run_complete_evaluation.py"
    ]
    
    for script in scripts_to_check:
        script_path = BASE_DIR / "scripts" / script
        if script_path.exists():
            size_kb = script_path.stat().st_size / 1024
            lines = sum(1 for _ in open(script_path))
            print(f"✓ {script:35s} ({lines:4d} lines, {size_kb:.1f} KB)")
        else:
            print(f"❌ {script:35s} MISSING")
    
    return True

def main():
    """Run all diagnostics"""
    print("\n" + "="*80)
    print("BERT EVALUATION - PRE-FLIGHT DIAGNOSTIC")
    print("="*80)
    print()
    
    # Check model
    try:
        num_classes = check_model()
    except Exception as e:
        print(f"\n❌ Model check failed: {e}")
        return 1
    
    # Check data
    try:
        available_labels = check_data()
    except Exception as e:
        print(f"\n❌ Data check failed: {e}")
        return 1
    
    # Recommend configuration
    recommended_labels = recommend_configuration(num_classes, available_labels)
    
    # Check scripts
    check_scripts()
    
    # Final recommendations
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n1. Model Configuration:")
    print(f"   - Output classes: {num_classes}")
    print(f"   - Use labels: {recommended_labels}")
    
    print(f"\n2. Script Configuration:")
    print(f"   - Update LABEL_COLS in scripts to: {recommended_labels}")
    
    print(f"\n3. Expected Evaluation:")
    if num_classes == 2:
        print(f"   - Binary toxicity detection")
        print(f"   - Single ROC/PR curve")
        print(f"   - Single reliability diagram")
        print(f"   - Simpler visualizations (1 label vs 6)")
    else:
        print(f"   - Multi-label classification")
        print(f"   - {num_classes} ROC/PR curves")
        print(f"   - {num_classes} reliability diagrams")
    
    print(f"\n4. Calibration Methods:")
    print(f"   - Temperature Scaling (single T)")
    print(f"   - Platt Scaling (per-label)")
    print(f"   - Isotonic Regression (per-label)")
    
    print(f"\n5. Performance Expectations:")
    if num_classes == 2:
        print(f"   - Faster evaluation (~10-15 min)")
        print(f"   - Clearer calibration improvement")
        print(f"   - Single F1/AUC metric")
    
    print(f"\n" + "="*80)
    print(f"✅ DIAGNOSTIC COMPLETE - Ready to run evaluation")
    print(f"="*80)
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

