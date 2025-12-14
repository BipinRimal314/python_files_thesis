"""
Isolation Forest for Insider Threat Detection (Anomaly Detection)
Unsupervised anomaly detection based on daily behavioral profiles.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib # Standardized on joblib
import config
import utils

logger = utils.logger

def load_data():
    """Load daily features for Isolation Forest"""
    file_path = config.DAILY_FEATURES_FILE
    
    if not file_path.exists():
        logger.error(f"Daily features not found at {file_path}")
        return None
    
    try:
        # Load Parquet
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded daily features: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading daily features: {e}")
        return None

def load_static_scaler():
    """Load the fitted scaler for static features using joblib"""
    scaler_path = config.MODEL_PATHS['static_scaler']
    if not scaler_path.exists():
        logger.error(f"Static scaler not found at {scaler_path}")
        return None
    
    try:
        # joblib.load is more robust for sklearn objects than pickle
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        return None

def main():
    logger.info(utils.generate_report_header("ISOLATION FOREST TRAINING (V2: MULTI-DATASET)"))
    
    # 1. Load Data
    df = load_data()
    if df is None:
        return None, None

    # 2. Preprocessing
    # Drop non-feature columns
    exclude_cols = ['user', 'day', 'is_anomaly', 'is_insider']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    
    # 3. Train Model
    model_config = config.ISOLATION_FOREST
    clf = IsolationForest(
        n_estimators=model_config['n_estimators'],
        max_samples=model_config['max_samples'],
        contamination=model_config['contamination'],
        max_features=model_config['max_features'],
        bootstrap=model_config['bootstrap'],
        n_jobs=model_config['n_jobs'],
        random_state=model_config['random_state'],
        verbose=model_config['verbose']
    )
    
    clf.fit(X)
    
    # 4. Predict / Generate Scores
    # decision_function returns negative values for anomalies, we invert so higher is more anomalous
    raw_scores = clf.decision_function(X)
    anomaly_scores = -raw_scores 
    
    # predict returns -1 for outlier, 1 for inlier
    predictions = clf.predict(X)
    # Convert to 0 (normal) and 1 (anomaly)
    binary_predictions = np.where(predictions == -1, 1, 0)
    
    # 5. Save Model
    try:
        joblib.dump(clf, config.MODEL_PATHS['isolation_forest'])
        logger.info(f"Model saved to {config.MODEL_PATHS['isolation_forest']}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # 6. Save Predictions
    results_df = df[['user', 'day']].copy()
    if 'is_anomaly' in df.columns:
        results_df['true_label'] = df['is_anomaly']
    else:
        results_df['true_label'] = 0
        
    results_df['prediction'] = binary_predictions
    results_df['anomaly_score'] = anomaly_scores
    
    output_path = config.RESULTS_DIR / 'isolation_forest_predictions.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # 7. Metrics
    metrics = utils.calculate_metrics(results_df['true_label'], binary_predictions, anomaly_scores)
    utils.print_metrics(metrics, "Isolation Forest")
    
    return clf, metrics

if __name__ == "__main__":
    main()