"""
Utility functions for Insider Threat Detection System
Common helper functions used across multiple modules
"""

import logging
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
import pickle
import config

# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=getattr(logging, config.LOGGING['level']),
        format=config.LOGGING['format'],
        datefmt=config.LOGGING['date_format'],
        handlers=[
            logging.FileHandler(config.LOGGING['file']),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# DATA LOADING
# ============================================================================
def load_cert_dataset(data_dir: Path = config.RAW_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load CMU-CERT dataset files
    
    Args:
        data_dir: Directory containing the CERT dataset files
        
    Returns:
        Dictionary with dataframe for each log type
    """
    logger.info("Loading CMU-CERT dataset...")
    datasets = {}
    
    for log_type, filename in config.CERT_FILES.items():
        file_path = data_dir / filename
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                datasets[log_type] = df
                logger.info(f"Loaded {log_type}: {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading {log_type}: {e}")
        else:
            logger.warning(f"File not found: {file_path}")
    
    return datasets

def save_dataframe(df: pd.DataFrame, filename: str, directory: Path = config.PROCESSED_DATA_DIR):
    """Save dataframe to CSV"""
    filepath = directory / filename
    df.to_csv(filepath, index=False)
    logger.info(f"Saved dataframe to {filepath}")

def load_dataframe(filename: str, directory: Path = config.PROCESSED_DATA_DIR) -> pd.DataFrame:
    """Load dataframe from CSV"""
    filepath = directory / filename
    df = pd.read_csv(filepath)
    logger.info(f"Loaded dataframe from {filepath}")
    return df

# ============================================================================
# PSEUDONYMIZATION
# ============================================================================
def pseudonymize_identifier(identifier: str, salt: str = config.PRIVACY['salt']) -> str:
    """
    Create a pseudonymized version of an identifier using SHA-256 hashing
    
    Args:
        identifier: Original identifier (username, hostname, etc.)
        salt: Salt for hashing
        
    Returns:
        Pseudonymized identifier
    """
    if pd.isna(identifier):
        return 'UNKNOWN'
    
    combined = f"{salt}{identifier}"
    hashed = hashlib.sha256(combined.encode()).hexdigest()
    return f"USER_{hashed[:12]}"  # Use first 12 chars for readability

def pseudonymize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Pseudonymize a specific column in a dataframe
    
    Args:
        df: Input dataframe
        column: Column name to pseudonymize
        
    Returns:
        Dataframe with pseudonymized column
    """
    df = df.copy()
    df[f'{column}_pseudo'] = df[column].apply(pseudonymize_identifier)
    logger.info(f"Pseudonymized column: {column}")
    return df

# ============================================================================
# DATA CLEANING
# ============================================================================
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe: remove duplicates, handle missing values
    
    Args:
        df: Input dataframe
        
    Returns:
        Cleaned dataframe
    """
    logger.info(f"Cleaning dataframe with {len(df)} rows")
    initial_rows = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Log missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.info(f"Missing values:\n{missing[missing > 0]}")
    
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in dataframe
    
    Args:
        df: Input dataframe
        strategy: 'drop', 'fill_mean', 'fill_median', 'fill_mode'
        
    Returns:
        Dataframe with handled missing values
    """
    df = df.copy()
    
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'fill_mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'fill_median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'fill_mode':
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)
    
    logger.info(f"Handled missing values using strategy: {strategy}")
    return df

# ============================================================================
# DATETIME HANDLING
# ============================================================================
def parse_datetime(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """
    Parse datetime column and add temporal features
    
    Args:
        df: Input dataframe
        datetime_column: Name of datetime column
        
    Returns:
        Dataframe with parsed datetime and additional temporal features
    """
    df = df.copy()
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    
    # Add temporal features
    df['hour'] = df[datetime_column].dt.hour
    df['day_of_week'] = df[datetime_column].dt.dayofweek
    df['day_of_month'] = df[datetime_column].dt.day
    df['month'] = df[datetime_column].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    
    logger.info(f"Parsed datetime column: {datetime_column}")
    return df

# ============================================================================
# MODEL PERSISTENCE
# ============================================================================
def save_model(model: Any, filepath: Path):
    """Save model to disk"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {filepath}")

def load_model(filepath: Path) -> Any:
    """Load model from disk"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {filepath}")
    return model

# ============================================================================
# METRICS CALCULATION
# ============================================================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate performance metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores (for AUC)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    
    # Clean the data: ensure binary labels and handle NaN
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if y_scores is not None:
        y_scores = np.array(y_scores)[valid_mask]
    
    # Ensure binary (0 or 1 only)
    y_true = (y_true > 0).astype(int)
    y_pred = (y_pred > 0).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0, average='binary'),
        'recall': recall_score(y_true, y_pred, zero_division=0, average='binary'),
        'f1_score': f1_score(y_true, y_pred, zero_division=0, average='binary')
    }
    
    if y_scores is not None:
        try:
            # Ensure scores are also binary-compatible
            if len(np.unique(y_true)) > 1:  # Need both classes for AUC
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
            else:
                logger.warning("Only one class present in y_true, cannot calculate AUC-ROC")
                metrics['auc_roc'] = 0.0
        except ValueError as e:
            logger.warning(f"Could not calculate AUC-ROC: {e}")
            metrics['auc_roc'] = 0.0
    
    # Confusion matrix components
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        })
    except ValueError:
        # Handle case where confusion matrix can't be computed
        logger.warning("Could not compute confusion matrix")
        metrics.update({
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0
        })
    
    return metrics

def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """Print metrics in a formatted way"""
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} Performance Metrics")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    if 'auc_roc' in metrics:
        logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info(f"{'='*60}\n")

# ============================================================================
# DATA SPLITTING
# ============================================================================
def split_data(df: pd.DataFrame, 
               train_ratio: float = config.TRAIN_RATIO,
               val_ratio: float = config.VAL_RATIO,
               test_ratio: float = config.TEST_RATIO,
               random_state: int = config.RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets
    
    Args:
        df: Input dataframe
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df

# ============================================================================
# ANOMALY SCORE NORMALIZATION
# ============================================================================
def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize anomaly scores to [0, 1] range
    
    Args:
        scores: Raw anomaly scores
        
    Returns:
        Normalized scores
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score - min_score < 1e-10:
        return np.zeros_like(scores)
    
    return (scores - min_score) / (max_score - min_score)

# ============================================================================
# JSON SERIALIZATION
# ============================================================================
def save_json(data: Dict, filepath: Path):
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON to {filepath}")

def load_json(filepath: Path) -> Dict:
    """Load dictionary from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {filepath}")
    return data

# ============================================================================
# REPORTING
# ============================================================================
def generate_report_header(title: str) -> str:
    """Generate a formatted report header"""
    border = "=" * 80
    return f"\n{border}\n{title.center(80)}\n{border}\n"

def timestamp() -> str:
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

logger.info("Utility functions loaded successfully")