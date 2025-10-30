#!/usr/bin/env python3
"""
BERT Toxicity Model Evaluation Pipeline
========================================
Comprehensive evaluation with post-hoc calibration, threshold optimization,
tokenizer analytics, and extensive visualization/documentation.

No model training - inference and calibration only.
"""

import os
import sys
import json
import warnings
import pickle
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve, roc_curve,
    auc, confusion_matrix, accuracy_score, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.special import expit as sigmoid
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
MAX_LEN = 256
BATCH_SIZE = 64  # Increased for faster processing
NUM_WORKERS = 8  # More workers for parallel data loading
USE_SUBSET = True  # Use subset for faster evaluation
SUBSET_SIZE = 10000  # Sample size for faster iteration

BASE_DIR = Path("/Users/seb/Desktop/EquityLens/equity-detector")
MODEL_DIR = BASE_DIR / "BERT_Model"
DATA_DIR = BASE_DIR / "equity-training-datasets"
VIZ_DIR = BASE_DIR / "visualizations"
DOCS_DIR = BASE_DIR / "docs"
EVAL_DIR = BASE_DIR / "evaluation"

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# Model outputs 2 classes (binary) - use aggregated toxic label for best accuracy
USE_AGGREGATED_TOXIC = True  # Combines all toxicity types for maximum performance
TEXT_COL = "comment_text"
EVAL_LABEL = "any_toxic"  # Aggregated label name

# Set seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_message(msg, level="INFO"):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

def ensure_dir(path):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data, path):
    """Save data to JSON"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path):
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)

# ============================================================================
# DATASET CLASS
# ============================================================================

class ToxicityDataset(Dataset):
    """Dataset for toxicity comments"""
    
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels)
        }

# ============================================================================
# CALIBRATION METHODS
# ============================================================================

class TemperatureScaling:
    """Temperature scaling calibration"""
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits, labels):
        """Fit temperature on validation set"""
        logits = np.array(logits)
        labels = np.array(labels)
        
        def nll_loss(T):
            """Negative log-likelihood"""
            scaled_probs = sigmoid(logits / T)
            # Binary cross-entropy
            eps = 1e-8
            scaled_probs = np.clip(scaled_probs, eps, 1 - eps)
            loss = -np.mean(labels * np.log(scaled_probs) + (1 - labels) * np.log(1 - scaled_probs))
            return loss
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        log_message(f"Temperature scaling: T = {self.temperature:.4f}")
        return self
    
    def transform(self, logits):
        """Apply temperature scaling"""
        return sigmoid(np.array(logits) / self.temperature)

class PlattScaling:
    """Platt scaling (per-label logistic calibration)"""
    
    def __init__(self):
        self.models = []
    
    def fit(self, logits, labels):
        """Fit per-label logistic regression"""
        logits = np.array(logits)
        labels = np.array(labels)
        
        self.models = []
        for i in range(labels.shape[1]):
            lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
            lr.fit(logits[:, i:i+1], labels[:, i])
            self.models.append(lr)
        
        log_message(f"Platt scaling: Fitted {len(self.models)} models")
        return self
    
    def transform(self, logits):
        """Apply Platt scaling"""
        logits = np.array(logits)
        calibrated = np.zeros_like(logits)
        for i, model in enumerate(self.models):
            calibrated[:, i] = model.predict_proba(logits[:, i:i+1])[:, 1]
        return calibrated

class IsotonicCalibration:
    """Isotonic regression calibration (per-label)"""
    
    def __init__(self):
        self.models = []
    
    def fit(self, probs, labels):
        """Fit per-label isotonic regression"""
        probs = np.array(probs)
        labels = np.array(labels)
        
        self.models = []
        for i in range(labels.shape[1]):
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(probs[:, i], labels[:, i])
            self.models.append(iso)
        
        log_message(f"Isotonic regression: Fitted {len(self.models)} models")
        return self
    
    def transform(self, probs):
        """Apply isotonic regression"""
        probs = np.array(probs)
        calibrated = np.zeros_like(probs)
        for i, model in enumerate(self.models):
            calibrated[:, i] = model.transform(probs[:, i])
        return calibrated

# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_ece(y_true, y_pred, n_bins=15):
    """Compute Expected Calibration Error"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred[mask].mean()
            ece += mask.sum() / len(y_pred) * np.abs(bin_acc - bin_conf)
    
    return ece

def compute_all_metrics(y_true, y_pred_probs, thresholds=None):
    """Compute comprehensive metrics"""
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    if thresholds is None:
        thresholds = [0.5] * y_true.shape[1]
    
    # Apply thresholds
    y_pred = (y_pred_probs >= np.array(thresholds)).astype(int)
    
    metrics = {}
    
    # Determine labels to evaluate
    eval_labels = [EVAL_LABEL] if USE_AGGREGATED_TOXIC else LABEL_COLS
    num_labels = y_true.shape[1]
    
    # Per-label metrics
    for i in range(num_labels):
        label = eval_labels[i] if i < len(eval_labels) else f"label_{i}"
        metrics[f"{label}_f1"] = f1_score(y_true[:, i], y_pred[:, i])
        metrics[f"{label}_roc_auc"] = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
        
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
        metrics[f"{label}_pr_auc"] = auc(recall, precision)
        
        metrics[f"{label}_ece"] = compute_ece(y_true[:, i], y_pred_probs[:, i])
        metrics[f"{label}_brier"] = brier_score_loss(y_true[:, i], y_pred_probs[:, i])
    
    # Macro metrics (average across all labels being evaluated)
    metrics["f1_macro"] = np.mean([metrics[f"{label}_f1"] for label in eval_labels[:num_labels]])
    
    # For binary classification (1 label), f1_micro = f1_macro
    if num_labels == 1:
        metrics["f1_micro"] = metrics["f1_macro"]
    else:
        metrics["f1_micro"] = f1_score(y_true, y_pred, average='micro')
    
    metrics["roc_auc_macro"] = np.mean([metrics[f"{label}_roc_auc"] for label in eval_labels[:num_labels]])
    metrics["pr_auc_macro"] = np.mean([metrics[f"{label}_pr_auc"] for label in eval_labels[:num_labels]])
    metrics["ece_macro"] = np.mean([metrics[f"{label}_ece"] for label in eval_labels[:num_labels]])
    metrics["brier_macro"] = np.mean([metrics[f"{label}_brier"] for label in eval_labels[:num_labels]])
    
    return metrics

def optimize_threshold(y_true, y_pred_probs, metric='f1'):
    """Optimize threshold per label"""
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    optimal_thresholds = []
    
    for i in range(y_true.shape[1]):
        best_thresh = 0.5
        best_score = 0.0
        
        for thresh in np.arange(0.1, 0.96, 0.05):
            y_pred = (y_pred_probs[:, i] >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true[:, i], y_pred)
            elif metric == 'ece':
                score = -compute_ece(y_true[:, i], y_pred_probs[:, i])  # Negative for minimization
            elif metric == 'brier':
                score = -brier_score_loss(y_true[:, i], y_pred_probs[:, i])
            
            if score > best_score:
                best_score = score
                best_thresh = thresh
        
        optimal_thresholds.append(best_thresh)
    
    return optimal_thresholds

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_label_prevalence(df, labels, split_name, save_path):
    """Plot label prevalence"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    prevalence = [df[label].mean() for label in labels]
    counts = [df[label].sum() for label in labels]
    
    bars = ax.bar(range(len(labels)), prevalence, color='steelblue', alpha=0.8)
    ax.set_xlabel('Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prevalence', fontsize=12, fontweight='bold')
    ax.set_title(f'Label Prevalence - {split_name} Set', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add counts on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({height*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")

def plot_cooccurrence_matrix(df, labels, save_path):
    """Plot label co-occurrence matrix"""
    cooc = np.zeros((len(labels), len(labels)))
    
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            cooc[i, j] = ((df[label1] == 1) & (df[label2] == 1)).sum()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cooc, annot=True, fmt='.0f', cmap='YlOrRd', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title('Label Co-occurrence Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")

def plot_roc_curves(y_true, y_pred_probs, labels, split_name, variant, save_path):
    """Plot ROC curves for all labels"""
    num_labels = len(labels)
    
    if num_labels == 1:
        # Single plot for binary classification
        fig, ax = plt.subplots(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_true[:, 0], y_pred_probs[:, 0])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=14)
        ax.set_title(f'ROC Curve - {split_name} - {variant}\n{labels[0].upper()}', 
                     fontsize=16, fontweight='bold')
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(alpha=0.3)
    else:
        # Multiple subplots for multi-label
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            axes[i].plot(fpr, tpr, color='darkorange', lw=2, 
                         label=f'ROC (AUC = {roc_auc:.3f})')
            axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.3)
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('False Positive Rate', fontweight='bold')
            axes[i].set_ylabel('True Positive Rate', fontweight='bold')
            axes[i].set_title(f'{label.upper()}', fontsize=12, fontweight='bold')
            axes[i].legend(loc="lower right")
            axes[i].grid(alpha=0.3)
        
        fig.suptitle(f'ROC Curves - {split_name} - {variant}', 
                     fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")

def plot_pr_curves(y_true, y_pred_probs, labels, split_name, variant, save_path):
    """Plot Precision-Recall curves for all labels"""
    num_labels = len(labels)
    
    if num_labels == 1:
        # Single plot for binary classification
        fig, ax = plt.subplots(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(y_true[:, 0], y_pred_probs[:, 0])
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, color='darkgreen', lw=3, 
                label=f'PR Curve (AUC = {pr_auc:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
        ax.set_title(f'Precision-Recall Curve - {split_name} - {variant}\n{labels[0].upper()}', 
                     fontsize=16, fontweight='bold')
        ax.legend(loc="best", fontsize=12)
        ax.grid(alpha=0.3)
    else:
        # Multiple subplots for multi-label
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, label in enumerate(labels):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
            pr_auc = auc(recall, precision)
            
            axes[i].plot(recall, precision, color='darkgreen', lw=2, 
                         label=f'PR (AUC = {pr_auc:.3f})')
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('Recall', fontweight='bold')
            axes[i].set_ylabel('Precision', fontweight='bold')
            axes[i].set_title(f'{label.upper()}', fontsize=12, fontweight='bold')
            axes[i].legend(loc="best")
            axes[i].grid(alpha=0.3)
        
        fig.suptitle(f'Precision-Recall Curves - {split_name} - {variant}', 
                     fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")

def plot_reliability_diagram(y_true, y_pred_probs, labels, split_name, variant, save_path, n_bins=15):
    """Plot reliability (calibration) diagrams"""
    num_labels = len(labels)
    
    if num_labels == 1:
        # Single plot for binary classification
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        y_t = y_true[:, 0]
        y_p = y_pred_probs[:, 0]
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        accuracies = []
        confidences = []
        counts = []
        
        for j in range(n_bins):
            mask = (y_p >= bin_boundaries[j]) & (y_p < bin_boundaries[j + 1])
            if mask.sum() > 0:
                accuracies.append(y_t[mask].mean())
                confidences.append(y_p[mask].mean())
                counts.append(mask.sum())
            else:
                accuracies.append(0)
                confidences.append(bin_centers[j])
                counts.append(0)
        
        ece = compute_ece(y_t, y_p, n_bins=n_bins)
        
        # Histogram
        ax1.bar(bin_centers, counts, width=1.0/n_bins, alpha=0.7, 
                color='steelblue', edgecolor='black')
        ax1.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Calibration plot
        ax2.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Perfect Calibration')
        ax2.plot(confidences, accuracies, 'ro-', lw=3, markersize=8, label='Model Calibration')
        ax2.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Probability (Accuracy)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Reliability Diagram - ECE={ece:.4f}', fontsize=14, fontweight='bold')
        ax2.legend(loc="upper left", fontsize=11)
        ax2.set_ylim([0, 1])
        ax2.set_xlim([0, 1])
        ax2.grid(alpha=0.3)
        
        fig.suptitle(f'{labels[0].upper()} - {split_name} - {variant}', 
                     fontsize=16, fontweight='bold')
    else:
        # Multiple subplots for multi-label
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, label in enumerate(labels):
            y_t = y_true[:, i]
            y_p = y_pred_probs[:, i]
            
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
            accuracies = []
            confidences = []
            counts = []
            
            for j in range(n_bins):
                mask = (y_p >= bin_boundaries[j]) & (y_p < bin_boundaries[j + 1])
                if mask.sum() > 0:
                    accuracies.append(y_t[mask].mean())
                    confidences.append(y_p[mask].mean())
                    counts.append(mask.sum())
                else:
                    accuracies.append(0)
                    confidences.append(bin_centers[j])
                    counts.append(0)
            
            # Plot
            axes[i].bar(bin_centers, counts, width=1.0/n_bins, alpha=0.3, 
                        color='lightblue', label='Count')
            ax2 = axes[i].twinx()
            ax2.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Perfect')
            ax2.plot(confidences, accuracies, 'ro-', lw=2, label='Model')
            
            ece = compute_ece(y_t, y_p, n_bins=n_bins)
            
            axes[i].set_xlabel('Predicted Probability', fontweight='bold')
            axes[i].set_ylabel('Count', fontweight='bold', color='blue')
            ax2.set_ylabel('True Probability', fontweight='bold', color='red')
            axes[i].set_title(f'{label.UPPER()} (ECE={ece:.4f})', 
                              fontsize=12, fontweight='bold')
            ax2.legend(loc="upper left")
            ax2.set_ylim([0, 1])
        
        fig.suptitle(f'Reliability Diagrams - {split_name} - {variant}', 
                     fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")

def plot_confusion_matrix_aggregate(y_true, y_pred, labels, split_name, variant, save_path):
    """Plot aggregated confusion matrix"""
    # Aggregate across all labels
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'], ax=ax)
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_title(f'Confusion Matrix (Aggregated) - {split_name} - {variant}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")

def plot_threshold_sweep(y_true, y_pred_probs, label_idx, label_name, save_path):
    """Plot F1 vs threshold for a single label"""
    thresholds = np.arange(0.05, 0.96, 0.01)
    f1_scores = []
    ece_scores = []
    brier_scores = []
    
    for thresh in thresholds:
        y_pred = (y_pred_probs[:, label_idx] >= thresh).astype(int)
        f1_scores.append(f1_score(y_true[:, label_idx], y_pred))
        ece_scores.append(compute_ece(y_true[:, label_idx], y_pred_probs[:, label_idx]))
        brier_scores.append(brier_score_loss(y_true[:, label_idx], y_pred_probs[:, label_idx]))
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(thresholds, f1_scores, 'b-', lw=2, label='F1 Score')
    ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(alpha=0.3)
    
    # Mark optimal F1
    best_f1_idx = np.argmax(f1_scores)
    ax1.axvline(thresholds[best_f1_idx], color='b', linestyle='--', alpha=0.5)
    ax1.scatter([thresholds[best_f1_idx]], [f1_scores[best_f1_idx]], 
                color='b', s=100, zorder=5, label=f'Optimal F1: {thresholds[best_f1_idx]:.2f}')
    
    ax2 = ax1.twinx()
    ax2.plot(thresholds, ece_scores, 'r-', lw=2, label='ECE')
    ax2.plot(thresholds, brier_scores, 'g-', lw=2, label='Brier')
    ax2.set_ylabel('ECE / Brier Score', fontsize=12, fontweight='bold', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.title(f'Threshold Optimization - {label_name.upper()}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main evaluation pipeline"""
    
    log_message("="*80)
    log_message("BERT TOXICITY MODEL EVALUATION PIPELINE")
    log_message("="*80)
    
    # Environment info
    log_message(f"Python: {sys.version}")
    log_message(f"PyTorch: {torch.__version__}")
    log_message(f"Device: {device}")
    log_message(f"Random Seed: {RANDOM_SEED}")
    log_message(f"Max Length: {MAX_LEN}")
    log_message(f"Batch Size: {BATCH_SIZE}")
    
    # ========================================================================
    # PHASE 1: DATA LOADING
    # ========================================================================
    
    log_message("\n" + "="*80)
    log_message("PHASE 1: DATA LOADING & PREPARATION")
    log_message("="*80)
    
    # Load validation data (train.csv)
    log_message("Loading validation data (train.csv)...")
    val_df = pd.read_csv(DATA_DIR / "train.csv")
    
    # Create aggregated toxic label (ANY toxic type = toxic)
    if USE_AGGREGATED_TOXIC:
        log_message("Creating aggregated 'any_toxic' label (severe_toxic, obscene, threat, insult, identity_hate → toxic)...")
        val_df[EVAL_LABEL] = (val_df[LABEL_COLS].sum(axis=1) > 0).astype(int)
        log_message(f"  Original 'toxic' prevalence: {val_df['toxic'].mean()*100:.2f}%")
        log_message(f"  Aggregated 'any_toxic' prevalence: {val_df[EVAL_LABEL].mean()*100:.2f}%")
    
    # Use subset for faster evaluation if enabled
    if USE_SUBSET and len(val_df) > SUBSET_SIZE:
        log_message(f"Using subset of {SUBSET_SIZE:,} samples for faster evaluation")
        val_df = val_df.sample(n=SUBSET_SIZE, random_state=RANDOM_SEED)
    
    log_message(f"Validation set: {len(val_df):,} samples")
    
    # Load test data
    log_message("Loading test data (test.csv + test_labels.csv)...")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_labels_df = pd.read_csv(DATA_DIR / "test_labels.csv")
    
    # Merge test data with labels
    test_df = test_df.merge(test_labels_df, on='id')
    
    # Filter out samples with -1 labels (unlabeled)
    test_df = test_df[(test_df[LABEL_COLS] != -1).all(axis=1)]
    
    # Create aggregated toxic label
    if USE_AGGREGATED_TOXIC:
        test_df[EVAL_LABEL] = (test_df[LABEL_COLS].sum(axis=1) > 0).astype(int)
        log_message(f"  Test 'any_toxic' prevalence: {test_df[EVAL_LABEL].mean()*100:.2f}%")
    
    # Use subset for faster evaluation if enabled
    if USE_SUBSET and len(test_df) > SUBSET_SIZE // 2:
        log_message(f"Using subset of {SUBSET_SIZE // 2:,} test samples for faster evaluation")
        test_df = test_df.sample(n=SUBSET_SIZE // 2, random_state=RANDOM_SEED)
    
    log_message(f"Test set: {len(test_df):,} samples (after filtering)")
    
    # Dataset statistics
    log_message("\nDataset Statistics:")
    log_message(f"  Validation: {len(val_df):,} samples")
    log_message(f"  Test: {len(test_df):,} samples")
    log_message(f"  Labels: {len(LABEL_COLS)}")
    
    # Label prevalence
    if USE_AGGREGATED_TOXIC:
        log_message("\nAggregated Label Prevalence (Validation):")
        count = val_df[EVAL_LABEL].sum()
        pct = val_df[EVAL_LABEL].mean() * 100
        log_message(f"  {EVAL_LABEL:15s}: {count:6,} ({pct:5.2f}%) <- ANY form of toxicity")
        
        log_message("\n  Breakdown by type:")
        for label in LABEL_COLS:
            count = val_df[label].sum()
            pct = val_df[label].mean() * 100
            log_message(f"    {label:15s}: {count:6,} ({pct:5.2f}%)")
        
        log_message("\nAggregated Label Prevalence (Test):")
        count = test_df[EVAL_LABEL].sum()
        pct = test_df[EVAL_LABEL].mean() * 100
        log_message(f"  {EVAL_LABEL:15s}: {count:6,} ({pct:5.2f}%) <- ANY form of toxicity")
        
        log_message("\n  Breakdown by type:")
        for label in LABEL_COLS:
            count = test_df[label].sum()
            pct = test_df[label].mean() * 100
            log_message(f"    {label:15s}: {count:6,} ({pct:5.2f}%)")
    else:
        log_message("\nLabel Prevalence (Validation):")
        for label in LABEL_COLS:
            count = val_df[label].sum()
            pct = val_df[label].mean() * 100
            log_message(f"  {label:15s}: {count:6,} ({pct:5.2f}%)")
        
        log_message("\nLabel Prevalence (Test):")
        for label in LABEL_COLS:
            count = test_df[label].sum()
            pct = test_df[label].mean() * 100
            log_message(f"  {label:15s}: {count:6,} ({pct:5.2f}%)")
    
    # Visualize label prevalence
    if USE_AGGREGATED_TOXIC:
        plot_label_prevalence(val_df, [EVAL_LABEL] + LABEL_COLS, "Validation", 
                              VIZ_DIR / "label_prevalence_val.png")
        plot_label_prevalence(test_df, [EVAL_LABEL] + LABEL_COLS, "Test", 
                              VIZ_DIR / "label_prevalence_test.png")
    else:
        plot_label_prevalence(val_df, LABEL_COLS, "Validation", 
                              VIZ_DIR / "label_prevalence_val.png")
        plot_label_prevalence(test_df, LABEL_COLS, "Test", 
                              VIZ_DIR / "label_prevalence_test.png")
    
    plot_cooccurrence_matrix(val_df, LABEL_COLS, 
                             VIZ_DIR / "class_imbalance_heatmap.png")
    
    # ========================================================================
    # PHASE 2: MODEL LOADING
    # ========================================================================
    
    log_message("\n" + "="*80)
    log_message("PHASE 2: MODEL & TOKENIZER LOADING")
    log_message("="*80)
    
    log_message(f"Loading model from: {MODEL_DIR}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    
    log_message(f"Model loaded: {model.__class__.__name__}")
    log_message(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # ========================================================================
    # PHASE 3: INFERENCE
    # ========================================================================
    
    log_message("\n" + "="*80)
    log_message("PHASE 3: UNCALIBRATED INFERENCE")
    log_message("="*80)
    
    def run_inference(df, split_name):
        """Run inference on dataset"""
        log_message(f"\nRunning inference on {split_name} set...")
        
        texts = df[TEXT_COL].tolist()
        
        # Use aggregated label if enabled
        if USE_AGGREGATED_TOXIC:
            labels = df[[EVAL_LABEL]].values  # Shape: (n, 1)
        else:
            labels = df[LABEL_COLS].values  # Shape: (n, 6)
        
        dataset = ToxicityDataset(texts, labels, tokenizer, MAX_LEN)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                               shuffle=False, num_workers=NUM_WORKERS, 
                               pin_memory=True if torch.cuda.is_available() else False)
        
        all_logits = []
        all_labels = []
        all_tokens_lengths = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Inference ({split_name})"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['labels'].cpu().numpy()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits_full = outputs.logits.cpu().numpy()  # Shape: (batch, 2)
                
                # For binary classification, extract positive class logit (class 1)
                logits = logits_full[:, 1:2]  # Shape: (batch, 1) - positive class only
                
                all_logits.append(logits)
                all_labels.append(labels_batch)
                
                # Track token lengths
                token_lengths = attention_mask.sum(dim=1).cpu().numpy()
                all_tokens_lengths.extend(token_lengths)
        
        all_logits = np.vstack(all_logits)
        all_labels = np.vstack(all_labels)
        all_probs = sigmoid(all_logits)
        
        log_message(f"  Logits shape: {all_logits.shape}")
        log_message(f"  Labels shape: {all_labels.shape}")
        log_message(f"  Avg token length: {np.mean(all_tokens_lengths):.1f}")
        
        return all_logits, all_probs, all_labels, all_tokens_lengths
    
    val_logits, val_probs, val_labels, val_token_lengths = run_inference(val_df, "Validation")
    test_logits, test_probs, test_labels, test_token_lengths = run_inference(test_df, "Test")
    
    # Define eval labels list for use throughout
    eval_labels_list = [EVAL_LABEL] if USE_AGGREGATED_TOXIC else LABEL_COLS
    
    # Save raw predictions
    np.save(EVAL_DIR / "val_logits.npy", val_logits)
    np.save(EVAL_DIR / "val_probs.npy", val_probs)
    np.save(EVAL_DIR / "val_labels.npy", val_labels)
    np.save(EVAL_DIR / "test_logits.npy", test_logits)
    np.save(EVAL_DIR / "test_probs.npy", test_probs)
    np.save(EVAL_DIR / "test_labels.npy", test_labels)
    log_message("Saved raw predictions to evaluation/")
    
    # ========================================================================
    # PHASE 4: BASELINE METRICS (UNCALIBRATED)
    # ========================================================================
    
    log_message("\n" + "="*80)
    log_message("PHASE 4: BASELINE METRICS (UNCALIBRATED)")
    log_message("="*80)
    
    # Compute metrics with default threshold (0.5)
    log_message("\nComputing baseline metrics (threshold=0.5)...")
    val_metrics_uncal = compute_all_metrics(val_labels, val_probs)
    test_metrics_uncal = compute_all_metrics(test_labels, test_probs)
    
    log_message("\n[VALIDATION - Uncalibrated]")
    log_message(f"  F1 (micro): {val_metrics_uncal['f1_micro']:.4f}")
    log_message(f"  F1 (macro): {val_metrics_uncal['f1_macro']:.4f}")
    log_message(f"  ROC-AUC (macro): {val_metrics_uncal['roc_auc_macro']:.4f}")
    log_message(f"  PR-AUC (macro): {val_metrics_uncal['pr_auc_macro']:.4f}")
    log_message(f"  ECE (macro): {val_metrics_uncal['ece_macro']:.4f}")
    log_message(f"  Brier (macro): {val_metrics_uncal['brier_macro']:.4f}")
    
    log_message("\n[TEST - Uncalibrated]")
    log_message(f"  F1 (micro): {test_metrics_uncal['f1_micro']:.4f}")
    log_message(f"  F1 (macro): {test_metrics_uncal['f1_macro']:.4f}")
    log_message(f"  ROC-AUC (macro): {test_metrics_uncal['roc_auc_macro']:.4f}")
    log_message(f"  PR-AUC (macro): {test_metrics_uncal['pr_auc_macro']:.4f}")
    log_message(f"  ECE (macro): {test_metrics_uncal['ece_macro']:.4f}")
    log_message(f"  Brier (macro): {test_metrics_uncal['brier_macro']:.4f}")
    
    # Optimize thresholds on validation
    log_message("\nOptimizing thresholds on validation set...")
    val_thresholds_f1 = optimize_threshold(val_labels, val_probs, metric='f1')
    val_thresholds_ece = optimize_threshold(val_labels, val_probs, metric='ece')
    val_thresholds_brier = optimize_threshold(val_labels, val_probs, metric='brier')
    
    log_message("\nOptimal Thresholds (F1):")
    for i, label in enumerate(eval_labels_list):
        log_message(f"  {label:15s}: {val_thresholds_f1[i]:.3f}")
    
    # Re-compute metrics with optimized thresholds
    val_metrics_uncal_opt = compute_all_metrics(val_labels, val_probs, val_thresholds_f1)
    test_metrics_uncal_opt = compute_all_metrics(test_labels, test_probs, val_thresholds_f1)
    
    log_message("\n[VALIDATION - Uncalibrated + Optimized Thresholds]")
    log_message(f"  F1 (micro): {val_metrics_uncal_opt['f1_micro']:.4f}")
    log_message(f"  F1 (macro): {val_metrics_uncal_opt['f1_macro']:.4f}")
    
    log_message("\n[TEST - Uncalibrated + Optimized Thresholds]")
    log_message(f"  F1 (micro): {test_metrics_uncal_opt['f1_micro']:.4f}")
    log_message(f"  F1 (macro): {test_metrics_uncal_opt['f1_macro']:.4f}")
    
    # Generate visualizations
    log_message("\nGenerating baseline visualizations...")
    
    eval_labels_list = [EVAL_LABEL] if USE_AGGREGATED_TOXIC else LABEL_COLS
    
    plot_roc_curves(val_labels, val_probs, eval_labels_list, "Validation", "Uncalibrated",
                    VIZ_DIR / "val_uncal_roc_perlabel.png")
    plot_roc_curves(test_labels, test_probs, eval_labels_list, "Test", "Uncalibrated",
                    VIZ_DIR / "test_uncal_roc_perlabel.png")
    
    plot_pr_curves(val_labels, val_probs, eval_labels_list, "Validation", "Uncalibrated",
                   VIZ_DIR / "val_uncal_pr_perlabel.png")
    plot_pr_curves(test_labels, test_probs, eval_labels_list, "Test", "Uncalibrated",
                   VIZ_DIR / "test_uncal_pr_perlabel.png")
    
    plot_reliability_diagram(val_labels, val_probs, eval_labels_list, "Validation", "Uncalibrated",
                            VIZ_DIR / "val_uncal_reliability.png")
    plot_reliability_diagram(test_labels, test_probs, eval_labels_list, "Test", "Uncalibrated",
                            VIZ_DIR / "test_uncal_reliability.png")
    
    val_pred_uncal = (val_probs >= 0.5).astype(int)
    test_pred_uncal = (test_probs >= 0.5).astype(int)
    plot_confusion_matrix_aggregate(val_labels, val_pred_uncal, eval_labels_list, "Validation", 
                                   "Uncalibrated", VIZ_DIR / "val_uncal_cm_aggregate.png")
    plot_confusion_matrix_aggregate(test_labels, test_pred_uncal, eval_labels_list, "Test", 
                                   "Uncalibrated", VIZ_DIR / "test_uncal_cm_aggregate.png")
    
    # Save metrics
    save_json({**val_metrics_uncal, "split": "val", "variant": "uncal"}, 
              EVAL_DIR / "metrics_val_uncal.json")
    save_json({**test_metrics_uncal, "split": "test", "variant": "uncal"}, 
              EVAL_DIR / "metrics_test_uncal.json")
    
    # ========================================================================
    # PHASE 5: CALIBRATION
    # ========================================================================
    
    log_message("\n" + "="*80)
    log_message("PHASE 5: POST-HOC CALIBRATION")
    log_message("="*80)
    
    calibration_results = {}
    
    # --- Temperature Scaling ---
    log_message("\n[1/3] Temperature Scaling...")
    temp_scaler = TemperatureScaling()
    temp_scaler.fit(val_logits, val_labels)
    
    val_probs_temp = temp_scaler.transform(val_logits)
    test_probs_temp = temp_scaler.transform(test_logits)
    
    val_thresholds_temp = optimize_threshold(val_labels, val_probs_temp, metric='f1')
    val_metrics_temp = compute_all_metrics(val_labels, val_probs_temp, val_thresholds_temp)
    test_metrics_temp = compute_all_metrics(test_labels, test_probs_temp, val_thresholds_temp)
    
    log_message(f"  VAL - F1 (macro): {val_metrics_temp['f1_macro']:.4f}, ECE: {val_metrics_temp['ece_macro']:.4f}")
    log_message(f"  TEST - F1 (macro): {test_metrics_temp['f1_macro']:.4f}, ECE: {test_metrics_temp['ece_macro']:.4f}")
    
    calibration_results['temperature'] = {
        'val_metrics': val_metrics_temp,
        'test_metrics': test_metrics_temp,
        'thresholds': val_thresholds_temp,
        'temperature': temp_scaler.temperature
    }
    
    # Generate visualizations
    plot_roc_curves(val_labels, val_probs_temp, eval_labels_list, "Validation", "Temperature",
                    VIZ_DIR / "val_temp_roc_perlabel.png")
    plot_roc_curves(test_labels, test_probs_temp, eval_labels_list, "Test", "Temperature",
                    VIZ_DIR / "test_temp_roc_perlabel.png")
    plot_pr_curves(val_labels, val_probs_temp, eval_labels_list, "Validation", "Temperature",
                   VIZ_DIR / "val_temp_pr_perlabel.png")
    plot_pr_curves(test_labels, test_probs_temp, eval_labels_list, "Test", "Temperature",
                   VIZ_DIR / "test_temp_pr_perlabel.png")
    plot_reliability_diagram(val_labels, val_probs_temp, eval_labels_list, "Validation", "Temperature",
                            VIZ_DIR / "val_temp_reliability.png")
    plot_reliability_diagram(test_labels, test_probs_temp, eval_labels_list, "Test", "Temperature",
                            VIZ_DIR / "test_temp_reliability.png")
    
    val_pred_temp = (val_probs_temp >= np.array(val_thresholds_temp)).astype(int)
    test_pred_temp = (test_probs_temp >= np.array(val_thresholds_temp)).astype(int)
    plot_confusion_matrix_aggregate(val_labels, val_pred_temp, eval_labels_list, "Validation", 
                                   "Temperature", VIZ_DIR / "val_temp_cm_aggregate.png")
    plot_confusion_matrix_aggregate(test_labels, test_pred_temp, eval_labels_list, "Test", 
                                   "Temperature", VIZ_DIR / "test_temp_cm_aggregate.png")
    
    # --- Platt Scaling ---
    log_message("\n[2/3] Platt Scaling...")
    platt_scaler = PlattScaling()
    platt_scaler.fit(val_logits, val_labels)
    
    val_probs_platt = platt_scaler.transform(val_logits)
    test_probs_platt = platt_scaler.transform(test_logits)
    
    val_thresholds_platt = optimize_threshold(val_labels, val_probs_platt, metric='f1')
    val_metrics_platt = compute_all_metrics(val_labels, val_probs_platt, val_thresholds_platt)
    test_metrics_platt = compute_all_metrics(test_labels, test_probs_platt, val_thresholds_platt)
    
    log_message(f"  VAL - F1 (macro): {val_metrics_platt['f1_macro']:.4f}, ECE: {val_metrics_platt['ece_macro']:.4f}")
    log_message(f"  TEST - F1 (macro): {test_metrics_platt['f1_macro']:.4f}, ECE: {test_metrics_platt['ece_macro']:.4f}")
    
    calibration_results['platt'] = {
        'val_metrics': val_metrics_platt,
        'test_metrics': test_metrics_platt,
        'thresholds': val_thresholds_platt,
        'coefficients': [(m.coef_[0][0], m.intercept_[0]) for m in platt_scaler.models]
    }
    
    # Generate visualizations
    plot_roc_curves(val_labels, val_probs_platt, eval_labels_list, "Validation", "Platt",
                    VIZ_DIR / "val_platt_roc_perlabel.png")
    plot_roc_curves(test_labels, test_probs_platt, eval_labels_list, "Test", "Platt",
                    VIZ_DIR / "test_platt_roc_perlabel.png")
    plot_pr_curves(val_labels, val_probs_platt, eval_labels_list, "Validation", "Platt",
                   VIZ_DIR / "val_platt_pr_perlabel.png")
    plot_pr_curves(test_labels, test_probs_platt, eval_labels_list, "Test", "Platt",
                   VIZ_DIR / "test_platt_pr_perlabel.png")
    plot_reliability_diagram(val_labels, val_probs_platt, eval_labels_list, "Validation", "Platt",
                            VIZ_DIR / "val_platt_reliability.png")
    plot_reliability_diagram(test_labels, test_probs_platt, eval_labels_list, "Test", "Platt",
                            VIZ_DIR / "test_platt_reliability.png")
    
    val_pred_platt = (val_probs_platt >= np.array(val_thresholds_platt)).astype(int)
    test_pred_platt = (test_probs_platt >= np.array(val_thresholds_platt)).astype(int)
    plot_confusion_matrix_aggregate(val_labels, val_pred_platt, eval_labels_list, "Validation", 
                                   "Platt", VIZ_DIR / "val_platt_cm_aggregate.png")
    plot_confusion_matrix_aggregate(test_labels, test_pred_platt, eval_labels_list, "Test", 
                                   "Platt", VIZ_DIR / "test_platt_cm_aggregate.png")
    
    # --- Isotonic Regression ---
    log_message("\n[3/3] Isotonic Regression...")
    isotonic_scaler = IsotonicCalibration()
    isotonic_scaler.fit(val_probs, val_labels)
    
    val_probs_isotonic = isotonic_scaler.transform(val_probs)
    test_probs_isotonic = isotonic_scaler.transform(test_probs)
    
    val_thresholds_isotonic = optimize_threshold(val_labels, val_probs_isotonic, metric='f1')
    val_metrics_isotonic = compute_all_metrics(val_labels, val_probs_isotonic, val_thresholds_isotonic)
    test_metrics_isotonic = compute_all_metrics(test_labels, test_probs_isotonic, val_thresholds_isotonic)
    
    log_message(f"  VAL - F1 (macro): {val_metrics_isotonic['f1_macro']:.4f}, ECE: {val_metrics_isotonic['ece_macro']:.4f}")
    log_message(f"  TEST - F1 (macro): {test_metrics_isotonic['f1_macro']:.4f}, ECE: {test_metrics_isotonic['ece_macro']:.4f}")
    
    calibration_results['isotonic'] = {
        'val_metrics': val_metrics_isotonic,
        'test_metrics': test_metrics_isotonic,
        'thresholds': val_thresholds_isotonic
    }
    
    # Generate visualizations
    plot_roc_curves(val_labels, val_probs_isotonic, eval_labels_list, "Validation", "Isotonic",
                    VIZ_DIR / "val_isotonic_roc_perlabel.png")
    plot_roc_curves(test_labels, test_probs_isotonic, eval_labels_list, "Test", "Isotonic",
                    VIZ_DIR / "test_isotonic_roc_perlabel.png")
    plot_pr_curves(val_labels, val_probs_isotonic, eval_labels_list, "Validation", "Isotonic",
                   VIZ_DIR / "val_isotonic_pr_perlabel.png")
    plot_pr_curves(test_labels, test_probs_isotonic, eval_labels_list, "Test", "Isotonic",
                   VIZ_DIR / "test_isotonic_pr_perlabel.png")
    plot_reliability_diagram(val_labels, val_probs_isotonic, eval_labels_list, "Validation", "Isotonic",
                            VIZ_DIR / "val_isotonic_reliability.png")
    plot_reliability_diagram(test_labels, test_probs_isotonic, eval_labels_list, "Test", "Isotonic",
                            VIZ_DIR / "test_isotonic_reliability.png")
    
    val_pred_isotonic = (val_probs_isotonic >= np.array(val_thresholds_isotonic)).astype(int)
    test_pred_isotonic = (test_probs_isotonic >= np.array(val_thresholds_isotonic)).astype(int)
    plot_confusion_matrix_aggregate(val_labels, val_pred_isotonic, eval_labels_list, "Validation", 
                                   "Isotonic", VIZ_DIR / "val_isotonic_cm_aggregate.png")
    plot_confusion_matrix_aggregate(test_labels, test_pred_isotonic, eval_labels_list, "Test", 
                                   "Isotonic", VIZ_DIR / "test_isotonic_cm_aggregate.png")
    
    # Save calibration parameters
    calibration_params = {
        'temperature': {
            'T': temp_scaler.temperature,
            'thresholds': val_thresholds_temp
        },
        'platt': {
            'coefficients': calibration_results['platt']['coefficients'],
            'thresholds': val_thresholds_platt
        },
        'isotonic': {
            'thresholds': val_thresholds_isotonic
        },
        'uncalibrated': {
            'thresholds_f1': val_thresholds_f1,
            'thresholds_ece': val_thresholds_ece,
            'thresholds_brier': val_thresholds_brier
        }
    }
    save_json(calibration_params, EVAL_DIR / "calibration_params.json")
    
    # ========================================================================
    # PHASE 6: THRESHOLD ANALYSIS
    # ========================================================================
    
    log_message("\n" + "="*80)
    log_message("PHASE 6: THRESHOLD ANALYSIS")
    log_message("="*80)
    
    log_message("\nGenerating threshold sweep plots...")
    for i, label in enumerate(eval_labels_list):
        plot_threshold_sweep(val_labels, val_probs, i, label, 
                            VIZ_DIR / f"threshold_sweep_{label}.png")
    
    # Create threshold comparison table
    threshold_data = []
    for i, label in enumerate(eval_labels_list):
        threshold_data.append({
            'label': label,
            'uncal_f1': val_thresholds_f1[i],
            'uncal_ece': val_thresholds_ece[i],
            'uncal_brier': val_thresholds_brier[i],
            'temp_f1': val_thresholds_temp[i],
            'platt_f1': val_thresholds_platt[i],
            'isotonic_f1': val_thresholds_isotonic[i]
        })
    
    threshold_df = pd.DataFrame(threshold_data)
    threshold_df.to_csv(EVAL_DIR / "thresholds_table.csv", index=False)
    log_message(f"Saved: {EVAL_DIR / 'thresholds_table.csv'}")
    
    # Threshold heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    threshold_matrix = threshold_df[['uncal_f1', 'temp_f1', 'platt_f1', 'isotonic_f1']].values.T
    sns.heatmap(threshold_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=eval_labels_list, 
                yticklabels=['Uncal (F1)', 'Temp (F1)', 'Platt (F1)', 'Isotonic (F1)'],
                ax=ax)
    ax.set_title('Optimal Thresholds Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "threshold_heatmap_all_methods.png", dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {VIZ_DIR / 'threshold_heatmap_all_methods.png'}")
    
    # ========================================================================
    # CONTINUED IN NEXT PART...
    # ========================================================================
    
    log_message("\n" + "="*80)
    log_message("Pipeline continues in tokenizer analysis...")
    log_message("="*80)
    
    return {
        'val_df': val_df,
        'test_df': test_df,
        'tokenizer': tokenizer,
        'val_token_lengths': val_token_lengths,
        'test_token_lengths': test_token_lengths,
        'calibration_results': calibration_results,
        'val_metrics_uncal': val_metrics_uncal,
        'test_metrics_uncal': test_metrics_uncal,
        'val_labels': val_labels,
        'test_labels': test_labels,
        'val_probs': val_probs,
        'test_probs': test_probs,
        'val_probs_temp': val_probs_temp,
        'test_probs_temp': test_probs_temp,
        'val_probs_platt': val_probs_platt,
        'test_probs_platt': test_probs_platt,
        'val_probs_isotonic': val_probs_isotonic,
        'test_probs_isotonic': test_probs_isotonic
    }

if __name__ == "__main__":
    results = main()
    log_message("\n" + "="*80)
    log_message("PHASE 1-6 COMPLETE - Proceeding to tokenizer & summary phases...")
    log_message("="*80)

