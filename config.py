"""
Configuration file for Insider Threat Detection System
[--- UPDATED VERSION ---]
- Disables sampling and filtering to ensure insider (ONS0995) is processed
- Adds paths for psychometric and LDAP data
- Sets LSTM Autoencoder save path to .keras to fix Keras 3.x error
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
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
# CMU-CERT Dataset files
CERT_FILES = {
    'logon': 'logon.csv',
    'device': 'device.csv',
    # 'file': 'file.csv',  # Removed as it's not in your /raw folder
    'email': 'email.csv',
    'http': 'http.csv',
    'psychometric': 'psychometric.csv', # Added
    'ldap': 'LDAP/' # Added
}

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# --- FIX: SAMPLING MUST BE DISABLED TO CAPTURE THE INSIDER ---
MAX_ROWS = 100000
USE_SAMPLING = False # Set to False to process the entire dataset

# ============================================================================
# PREPROCESSING SETTINGS
# ============================================================================
# De-identification settings
PSEUDONYMIZE_USERS = True
PSEUDONYMIZE_HOSTS = True

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
    'contamination': 0.01,
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
# ENSEMBLE SETTINGS
# ============================================================================
ENSEMBLE = {
    'weights': {
        'isolation_forest': 0.4,
        'deep_clustering': 0.3,
        'lstm_autoencoder': 0.3
    },
    'voting_method': 'weighted',
    'final_threshold': 0.7
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================
EVALUATION = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc_roc'],
    'cross_validation_folds': 3,
    'threshold_steps': 50
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
    'file': LOGS_DIR / 'insider_threat_detection.log'
}

# ============================================================================
# ALERT SETTINGS
# ============================================================================
ALERTS = {
    'severity_levels': {
        'low': (0.5, 0.7),
        'medium': (0.7, 0.85),
        'high': (0.85, 0.95),
        'critical': (0.95, 1.0)
    },
    'max_alerts_per_day': 100,
    'explainability': True
}

# ============================================================================
# PRIVACY SETTINGS
# ============================================================================
PRIVACY = {
    'anonymization_method': 'hash',
    'salt': 'insider_threat_detection_2025',
    'keep_temporal_patterns': True,
    'aggregate_low_frequency_users': True,
    # --- FIX: FILTERING MUST BE DISABLED TO KEEP THE INSIDER ---
    'min_user_activity_threshold': 0 # Set to 0 to disable filtering
}

# ============================================================================
# FEATURE COLUMNS
# ============================================================================
FEATURE_COLUMNS = {
    'temporal': [],
    'behavioral': [],
    'statistical': [],
    'sequential': []
}

# ============================================================================
# MODEL SAVE PATHS
# ============================================================================
MODEL_PATHS = {
    'isolation_forest': MODELS_DIR / 'isolation_forest.pkl',
    'deep_clustering': MODELS_DIR / 'deep_clustering.pkl',
    # --- FIX: Keras models must be .keras or .h5 ---
    'lstm_autoencoder': MODELS_DIR / 'lstm_autoencoder.keras',
    
    # --- NEW: Separate scalers for static and daily data ---
    'static_scaler': MODELS_DIR / 'static_feature_scaler.pkl',
    'daily_scaler': MODELS_DIR / 'daily_feature_scaler.pkl',
    
    'label_encoders': MODELS_DIR / 'label_encoders.pkl'
}

# ============================================================================
# RESULT PATHS
# ============================================================================
RESULT_PATHS = {
    'evaluation_metrics': RESULTS_DIR / 'evaluation_metrics.csv',
    'confusion_matrices': RESULTS_DIR / 'confusion_matrices',
    'roc_curves': RESULTS_DIR / 'roc_curves',
    'anomaly_scores': RESULTS_DIR / 'anomaly_scores.csv',
    'alerts': RESULTS_DIR / 'generated_alerts.csv',
    'visualizations': RESULTS_DIR / 'visualizations'
}

# Create result subdirectories
for key, path in RESULT_PATHS.items():
    if key in ['confusion_matrices', 'roc_curves', 'visualizations']:
        path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PERFORMANCE OPTIMIZATIONS FOR MACBOOK
# ============================================================================
PERFORMANCE = {
    'chunk_size': 500000, # Increased chunk size
    'use_float32': True,
    'use_multiprocessing': True,
    'n_jobs': -1,
    'tf_memory_growth': True,
    'mixed_precision': False,
    'cache_preprocessed': True,
    'incremental_loading': True
}

# ============================================================================
# MACBOOK M4 PRO SPECIFIC SETTINGS
# ============================================================================
MACBOOK_OPTIMIZATIONS = {
    'use_mps': True,
    'max_memory_mb': 8000,
    'enable_batch_processing': True,
    'max_features': 50,
    'use_stratified_sampling': True
}

print(f"Configuration loaded (LIGHTWEIGHT MODE for MacBook M4 Pro)")
print(f"Base directory: {BASE_DIR}")
print(f"Performance optimizations enabled")