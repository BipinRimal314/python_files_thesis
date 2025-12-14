"""
Configuration file for Insider Threat Detection System
FAST TESTING MODE - Optimized for quick pipeline verification
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
LEGACY_DIR = BASE_DIR / "legacy"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, LEGACY_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FILE PATHS (INTERMEDIATE DATA)
# ============================================================================
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_unified_logs.csv"
DAILY_FEATURES_FILE = PROCESSED_DATA_DIR / "daily_features.parquet"
SEQUENCE_DATA_FILE = PROCESSED_DATA_DIR / "sequence_features.parquet" 
SEQUENCE_LABELS_FILE = PROCESSED_DATA_DIR / "sequence_labels.npy"

# ============================================================================
# DATA SETTINGS
# ============================================================================

# Datasets to process (Use only 'r1' for speed)
DATASET_SUBSET = ['r1', 'r2', 'r3.1', 'r3.2', 'r4.1'] 

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

LDAP_PATH = Path("LDAP") 

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ============================================================================
# PREPROCESSING SETTINGS
# ============================================================================
PRIVACY = {
    'min_user_activity_threshold': 0,
    'salt': 'salty_salty_salt' 
}

TIME_WINDOWS = {
    'daily': '1D'
}

# Sequence settings for LSTM
SEQUENCE_LENGTH = 15
SEQUENCE_STRIDE = 10 

# Set to None to use full dataset (was 5000 for debugging)
MAX_SEQUENCE_SAMPLES = None

# ============================================================================
# ISOLATION FOREST PARAMETERS (Fast Mode)
# ============================================================================
ISOLATION_FOREST = {
    'n_estimators': 50, # Low number of trees
    'max_samples': 256, 
    'contamination': 'auto', 
    'max_features': 1.0,
    'bootstrap': False,
    'n_jobs': -1,
    'random_state': RANDOM_SEED,
    'verbose': 0
}

# ============================================================================
# DEEP CLUSTERING PARAMETERS (Fast Mode)
# ============================================================================
DEEP_CLUSTERING = {
    'n_clusters': 5,
    'input_dim': None,
    'encoding_dims': [64, 32],
    'activation': 'relu',
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 10,
    'lambda_param': 0.1,
    'patience': 3,
    'verbose': 1
}

# ============================================================================
# LSTM AUTOENCODER PARAMETERS (Fast Mode)
# ============================================================================
LSTM_AUTOENCODER = {
    'encoding_dim': 16,
    'lstm_units': [32, 16],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 256,
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
# ENSEMBLE PARAMETERS
# ============================================================================
ENSEMBLE = {
    'weights': {
        'isolation_forest': 0.3,
        'lstm_autoencoder': 0.4,
        'deep_clustering': 0.3
    },
    'final_threshold': 0.7 
}

# ============================================================================
# ALERT SETTINGS
# ============================================================================
ALERTS = {
    'severity_levels': {
        'low': (0.7, 0.8),
        'medium': (0.8, 0.9),
        'high': (0.9, 0.98),
        'critical': (0.98, 1.0)
    }
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
# MODEL SAVE PATHS
# ============================================================================
MODEL_PATHS = {
    'isolation_forest': MODELS_DIR / 'isolation_forest_model.pkl',
    'isolation_forest_v2': MODELS_DIR / 'isolation_forest_model_v2.pkl',
    'lstm_autoencoder': MODELS_DIR / 'lstm_autoencoder_model.keras',
    'deep_clustering': MODELS_DIR / 'deep_clustering_model.pkl',
    'static_scaler': MODELS_DIR / 'static_scaler.pkl',
    'static_scaler_v2': MODELS_DIR / 'static_scaler_v2.pkl',
    'daily_scaler': MODELS_DIR / 'daily_scaler.pkl'
}

# ============================================================================
# RESULT PATHS
# ============================================================================
RESULT_PATHS = {
    'evaluation_metrics': RESULTS_DIR / 'evaluation_metrics.csv',
    'optimal_evaluation_report': RESULTS_DIR / 'optimal_evaluation_report.csv',
    'confusion_matrices': RESULTS_DIR / 'confusion_matrices',
    'roc_curves': RESULTS_DIR / 'roc_curves',
    'visualizations': RESULTS_DIR / 'visualizations',
    'alerts': RESULTS_DIR / 'alerts.csv'
}

for key, path in RESULT_PATHS.items():
    if key in ['confusion_matrices', 'roc_curves', 'visualizations']:
        path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PERFORMANCE OPTIMIZATIONS
# ============================================================================
PERFORMANCE = {
    'streaming_chunk_size': 50000, 
    'force_rerun': False
}

print(f"Configuration loaded. MAX_SEQUENCE_SAMPLES: {MAX_SEQUENCE_SAMPLES or 'Full dataset'}")
print(f"Raw data source: {RAW_DATA_DIR}")