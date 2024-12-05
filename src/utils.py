# utils.py

from typing import List, Dict, Tuple
import numpy as np
import logging
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate_model(y_true: List[str], y_pred: List[str]) -> Dict:
    """Calculate accuracy, precision, recall, F1 score"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    return metrics

def create_confusion_matrix(y_true: List[str], y_pred: List[str]) -> np.ndarray:
    """Generate confusion matrix"""
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm

def save_results(results: Dict, filepath: str):
    """Save evaluation results to file"""
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=4)

def setup_logger() -> logging.Logger:
    """Configure logging"""
    logger = logging.getLogger('language_identifier')
    logger.setLevel(logging.INFO)
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add the handler to the logger if not already added
    if not logger.handlers:
        logger.addHandler(handler)
    return logger