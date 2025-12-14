"""
Configuration file for Insider Threat Detection System
V2 - MULTI-DATASET VERSION

[--- V2.4 ---]
- Updated DATASET_SUBSET to ['r1', 'r2', 'r3.1', 'r3.2']
  to match available hardware and prevent OOM/storage crashes.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "all_data" 
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA SETTINGS
# ============================================================================

# (--- V2.4 FINAL SUBSET ---)
# We will ONLY process these 4 datasets.
DATASET_SUBSET = ['r1', 'r2', 'r3.1']
# ---------------------

# (--- V2 ---) Define all possible log and static files
LOG_FILENAMES = {
    'logon': 'logon.csv',
    'device': 'device.csv',
    'file': 'file.csv',
    'email': 'email.csv',
    'http': 'http.csv'
}

STATIC_FILENAMES = {
    'psychometric': 'psychometric.csv'
}

# (--- V2 ---) Canonical path for LDAP data
LDAP_PATH = Path("LDAP") # Relative to RAW_DATA_DIR

# Data split ratios (will be applied to *normal* data only)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# PREPROCESSING SETTINGS
# ============================================================================
PRIVACY = {
    'min_user_activity_threshold': 0 
}

# Feature engineering
TIME_WINDOWS = {
    'daily': '1D'
}

# Sequence settings for LSTM - REDUCED
SEQUENCE_LENGTH = 15
SEQUENCE_STRIDE = 10 

# ============================================================================
# ISOLATION FOREST PARAMETERS - OPTIMIZED
# ============================================================================
ISOLATION_FOREST = {
    'n_estimators': 50,
    'max_samples': 256, 
    'contamination': 'auto', 
    'max_features': 1.0,
    'bootstrap': False,
    'n_jobs': -1,
    'random_state': RANDOM_SEED,
    'verbose': 0
}

# ============================================================================
# DEEP CLUSTERING PARAMETERS - LIGHTWEIGHT
# ============================================================================
DEEP_CLUSTERING = {
    'n_clusters': 5,
    'input_dim': None,
    'encoding_dims': [64, 32],
    'activation': 'relu',
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 30,
    'lambda_param': 0.1,
    'patience': 5,
    'verbose': 1
}

# ============================================================================
# LSTM AUTOENCODER PARAMETERS - LIGHTWEIGHT
# ============================================================================
LSTM_AUTOENCODER = {
    'encoding_dim': 16,
    'lstm_units': [32, 16],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 128,
    'epochs': 20,
    'patience': 5,
    'verbose': 1,
    'reconstruction_error_threshold': None
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================
EVALUATION = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc_roc'],
    'threshold_steps': 100 
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
VISUALIZATION = {
    'figure_size': (10, 6),
    'dpi': 150,
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'Set2',
    'save_format': 'png'
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'file': LOGS_DIR / 'insider_threat_detection_v2.log'
}

# ============================================================================
# MODEL SAVE PATHS
# ============================================================================
MODEL_PATHS = {
    # V1 Models (from r2 baseline)
    'isolation_forest': MODELS_DIR / 'isolation_forest.pkl',
    'deep_clustering': MODELS_DIR / 'deep_clustering.pkl',
    'lstm_autoencoder': MODELS_DIR / 'lstm_autoencoder.keras',
    'scaler': MODELS_DIR / 'feature_scaler.pkl', # Deprecated
    'label_encoders': MODELS_DIR / 'label_encoders.pkl', # Deprecated

    # (--- V2 ---) New models and scalers for the full dataset
    'isolation_forest_v2': MODELS_DIR / 'isolation_forest_v2.pkl',
    'deep_clustering_v2': MODELS_DIR / 'deep_clustering_v2.pkl',
    'lstm_autoencoder_v2': MODELS_DIR / 'lstm_autoencoder_v2.keras',
    'static_scaler_v2': MODELS_DIR / 'static_feature_scaler_v2.pkl',
    'daily_scaler_v2': MODELS_DIR / 'daily_feature_scaler_v2.pkl',
}

# ============================================================================
# RESULT PATHS
# ============================================================================
RESULT_PATHS = {
    # V1 Results
    'evaluation_metrics': RESULTS_DIR / 'evaluation_metrics.csv',
    'anomaly_scores': RESULTS_DIR / 'anomaly_scores.csv',

    # (--- V2 ---) New results
    'evaluation_metrics_v2': RESULTS_DIR / 'evaluation_metrics_v2.csv',
    'optimal_evaluation_report_v2': RESULTS_DIR / 'optimal_evaluation_report_v2.csv',

    # Visualizations (shared)
    'confusion_matrices': RESULTS_DIR / 'confusion_matrices',
    'roc_curves': RESULTS_DIR / 'roc_curves',
    'visualizations': RESULTS_DIR / 'visualizations'
}

# Create result subdirectories
for key, path in RESULT_PATHS.items():
    if key in ['confusion_matrices', 'roc_curves', 'visualizations']:
        path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PERFORMANCE OPTIMIZATIONS
# ============================================================================
PERFORMANCE = {
    # Memory management
    'chunk_size': 2000000,
    
    # (--- V2.2 SAFETY FLAG ---)
    'FORCE_RERUN_PREPROCESSING': False,
    
    # (--- V2.2 SAFETY FLAG ---)
    'FORCE_RERUN_PASS1': False
}

print(f"Configuration V2 (Multi-Dataset) loaded.")
print(f"Raw data source: {RAW_DATA_DIR}")