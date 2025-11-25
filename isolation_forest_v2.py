"""
Model 1: Isolation Forest (V2: Multi-Dataset)
Trains on the new, multi-dataset 'engineered_static_features_v2.csv' file
which includes all datasets and advanced Z-score features.

[--- V4.1 FINAL FIX ---]
- Dynamically builds the input/output filenames based on DATASET_SUBSET.
- Relies on the fixed `utils.calculate_metrics` to prevent crashes.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import joblib

# Use the V2 config
import config_v2 as config 
import utils

logger = utils.logger

class IsolationForestDetectorV2:
    
    def __init__(self, **kwargs):
        """
        Initialize the Isolation Forest model using V2 parameters.
        """
        # Load V2 params from config
        self.params = {**config.ISOLATION_FOREST, **kwargs}
        self.model = IsolationForest(**self.params)
        self.threshold = None
        
        logger.info(f"Initialized Isolation Forest (V2) with params: {self.params}")

    def train(self, X_train_normal: pd.DataFrame):
        """
        Train the Isolation Forest model *only* on normal data.
        
        Args:
            X_train_normal: DataFrame of normal user features for training.
        """
        logger.info(f"Training Isolation Forest on {len(X_train_normal)} normal samples...")
        self.model.fit(X_train_normal)
        logger.info("Isolation Forest training completed")

    def calibrate_threshold(self, X_val_normal: pd.DataFrame, target_fpr: float = 0.05):
        """
        Calibrate the anomaly threshold using a validation set of *normal* data.
        
        Args:
            X_val_normal: Validation features (normal data).
            target_fpr: The desired False Positive Rate to set the threshold.
        """
        logger.info(f"Calibrating threshold for target FPR={target_fpr}...")
        
        # Get anomaly scores for normal validation data
        # Note: score_samples returns the *opposite* of anomaly score
        # (higher is more normal). We must flip the sign.
        scores_normal = -self.model.score_samples(X_val_normal)
        
        # Find the score at the (1 - FPR) percentile
        self.threshold = np.percentile(scores_normal, 100 * (1 - target_fpr))
        logger.info(f"Calibrated threshold: {self.threshold:.4f}")
        return self.threshold

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies on new data.
        
        Args:
            X: DataFrame of features to predict on.
            
        Returns:
            Tuple of (predictions, anomaly_scores)
        """
        if self.threshold is None:
            logger.warning("Threshold not set. Calibrating with 99th percentile as default.")
            scores = -self.model.score_samples(X)
            self.threshold = np.percentile(scores, 99)
        
        # Get anomaly scores
        anomaly_scores = -self.model.score_samples(X)
        
        # Predict '1' (anomaly) if score is GREATER than the threshold
        predictions = (anomaly_scores > self.threshold).astype(int)
        
        # Normalize scores for reporting
        anomaly_scores_normalized = utils.normalize_scores(anomaly_scores)
        
        return predictions, anomaly_scores_normalized

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on a labeled test set.
        
        Args:
            X_test: Test features.
            y_test: Test labels (contains both 0s and 1s).
            
        Returns:
            Dictionary of performance metrics.
        """
        logger.info("Evaluating Isolation Forest (V2)...")
        predictions, scores = self.predict(X_test)
        
        if y_test.sum() > 0:
            logger.info(f"Test set contains {y_test.sum()} positive samples.")
            metrics = utils.calculate_metrics(y_test, predictions, scores)
        else:
            logger.warning("No positive samples in test set. Metrics (Precision, Recall, F1, AUC-ROC) will be 0 or NaN.")
            metrics = utils.calculate_metrics(y_test, predictions, scores)
        
        utils.print_metrics(metrics, "Isolation Forest (V2)")
        return metrics

    def save_model(self, filepath=None):
        """Save the trained model and scaler to disk."""
        if filepath is None:
            filepath = config.MODEL_PATHS['isolation_forest_v2']
        
        utils.save_model(self.model, filepath)
        logger.info(f"Saving Isolation Forest model to {filepath}")

    def load_model(self, filepath=None):
        """Load a trained model from disk."""
        if filepath is None:
            filepath = config.MODEL_PATHS['isolation_forest_v2']
        
        self.model = utils.load_model(filepath)
        logger.info(f"Loaded Isolation Forest model from {filepath}")


def main():
    """Main execution function"""
    logger.info(utils.generate_report_header("ISOLATION FOREST TRAINING (V2: MULTI-DATASET)"))

    # --- (V4.1 FILENAME FIX) ---
    # Dynamically build the filenames based on the config subset
    subset = config.DATASET_SUBSET if hasattr(config, 'DATASET_SUBSET') and config.DATASET_SUBSET else []
    subset_name = "_".join(subset) if subset else "ALL"
    
    static_features_file = config.PROCESSED_DATA_DIR / f'engineered_static_features_{subset_name}_v2.csv'
    static_scaler_file = config.MODEL_PATHS['static_scaler_v2'] # This path is already correct
    
    logger.info(f"Loading static features from: {static_features_file.name}")
    # --- (END V4.1 FIX) ---

    # --- Data Loading ---
    try:
        # Load the new V2 static features
        df = pd.read_csv(static_features_file)
    except Exception as e:
        logger.error(f"Failed to load '{static_features_file.name}'. Run feature_engineering_v2.py first. Error: {e}")
        return None, None
        
    # Load the new V2 static scaler
    try:
        scaler = utils.load_model(static_scaler_file)
    except Exception as e:
        logger.error(f"Failed to load '{static_scaler_file.name}'. Run feature_engineering_v2.py first. Error: {e}")
        return None, None
        
    # Define feature columns and label
    user_col = 'user_id'
    label_col = 'is_insider'
    feature_cols = [col for col in df.columns if col not in [user_col, label_col]]
    
    X = df[feature_cols]
    y = df[label_col] # y can be float here, your fixed utils.py will handle it
    
    logger.info(f"Data loaded: {len(X)} total samples, {y.sum()} insiders")
    
    # --- New Experimental Design ---
    # 1. Split data into "normal" (for training) and "full" (for testing)
    X_normal = X[y == 0]
    y_normal = y[y == 0]
    
    # Check if we have any normal data to train on
    if len(X_normal) == 0:
        logger.error("No normal data (is_insider == 0) found. Cannot train models.")
        return None, None

    # 2. Create a separate, smaller validation set *from the normal data*
    X_train_normal, X_val_normal, y_train_normal, y_val_normal = train_test_split(
        X_normal, y_normal, test_size=0.2, random_state=config.RANDOM_SEED
    )
    
    # 3. The Test Set is the *full, original* dataset
    X_test = X
    y_test = y
    
    logger.info(f"Training on: {len(X_train_normal)} normal samples")
    logger.info(f"Calibrating on: {len(X_val_normal)} normal samples")
    logger.info(f"Testing on: {len(X_test)} total samples (including {y_test.sum()} insiders)")

    # --- Model Training ---
    detector = IsolationForestDetectorV2()
    
    # 1. Train *only* on normal data
    detector.train(X_train_normal)
    
    # 2. Calibrate threshold *only* on normal validation data
    detector.calibrate_threshold(X_val_normal, target_fpr=0.05)
    
    # 3. Evaluate on the *full* test set (which includes the insiders)
    metrics = detector.evaluate(X_test, y_test)
    
    # --- Save Results ---
    detector.save_model()
    
    # Save metrics for model_evaluation_v2.py
    metrics_df = pd.DataFrame([metrics])
    metrics_df['model'] = 'isolation_forest'
    
    # (V4.1 FILENAME FIX)
    metrics_file = config.RESULTS_DIR / f'isolation_forest_metrics_{subset_name}_v2.csv'
    metrics_df.to_csv(metrics_file, index=False)
    
    # Save predictions for ensemble
    predictions, scores = detector.predict(X_test)
    results_df = pd.DataFrame({
        'user_id': df[user_col],
        'true_label': y_test,
        'prediction': predictions,
        'anomaly_score': scores
    })
    
    # (V4.1 FILENAME FIX)
    predictions_file = config.RESULTS_DIR / f'isolation_forest_predictions_{subset_name}_v2.csv'
    results_df.to_csv(predictions_file, index=False)
    
    logger.info("Isolation Forest (V2) training and evaluation completed!")
    
    return detector, metrics


if __name__ == "__main__":
    main()