"""
Model 3: Deep Clustering (Lightweight)
[--- UPDATED VERSION ---]
- Loads 'static_scaler.pkl'
- Loads 'engineered_static_features.csv'
- Uses 'stratify=y' on all splits
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from typing import Tuple, Dict
import config
import utils

logger = utils.logger

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available, using statistical fallback for Deep Clustering")

class DeepClusteringLite:
    
    def __init__(self, n_clusters: int, input_dim: int, **kwargs):
        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.params = config.DEEP_CLUSTERING.copy()
        self.params.update(kwargs)
        self.is_trained = False
        self.threshold = None
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, 
                                      random_state=config.RANDOM_SEED,
                                      batch_size=self.params['batch_size'],
                                      n_init=10)
        # --- NEW: Load the correct scaler ---
        try:
            self.scaler = utils.load_model(config.MODELS_DIR / 'static_scaler.pkl')
        except FileNotFoundError:
            logger.error("static_scaler.pkl not found! Run feature_engineering.py first.")
            self.scaler = None

        logger.info(f"Initialized Lightweight Deep Clustering (n_clusters={n_clusters})")
    
    def train(self, X_train: np.ndarray):
        logger.info("Training lightweight deep clustering model (K-Means)...")
        self.kmeans.fit(X_train)
        self.is_trained = True
        logger.info("Lightweight deep clustering training completed")
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting")
        
        distances = self.kmeans.transform(X)
        anomaly_scores = np.min(distances, axis=1)
        anomaly_scores = utils.normalize_scores(anomaly_scores)
        
        if self.threshold is not None:
            predictions = (anomaly_scores > self.threshold).astype(int)
        else:
            predictions = (anomaly_scores > 0.95).astype(int)
        
        return predictions, anomaly_scores
    
    def calibrate_threshold(self, X_val: np.ndarray, y_val: np.ndarray, 
                           target_fpr: float = 0.05) -> float:
        logger.info(f"Calibrating threshold for target FPR={target_fpr}")
        _, anomaly_scores = self.predict(X_val)
        
        normal_scores = anomaly_scores[y_val == 0]
        if len(normal_scores) == 0:
            logger.warning("No normal samples in validation set for calibration. Using default 0.95")
            self.threshold = 0.95
            return self.threshold
            
        normal_scores_sorted = np.sort(normal_scores)
        threshold_idx = int(len(normal_scores_sorted) * (1 - target_fpr))
        self.threshold = normal_scores_sorted[threshold_idx] if threshold_idx < len(normal_scores_sorted) else 0.95
        
        logger.info(f"Calibrated threshold: {self.threshold:.4f}")
        return self.threshold
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        logger.info("Evaluating lightweight deep clustering model...")
        predictions, scores = self.predict(X_test)
        metrics = utils.calculate_metrics(y_test, predictions, scores)
        utils.print_metrics(metrics, "Deep Clustering (Lightweight)")
        return metrics
    
    def save_model(self, filepath = None):
        if filepath is None:
            filepath = config.MODEL_PATHS['deep_clustering']
        utils.save_model(self.kmeans, filepath)
        logger.info(f"Lightweight deep clustering model saved to {filepath}")
    
    def load_model(self, filepath = None):
        if filepath is None:
            filepath = config.MODEL_PATHS['deep_clustering']
        self.kmeans = utils.load_model(filepath)
        self.is_trained = True
        logger.info(f"Lightweight deep clustering model loaded from {filepath}")


def main():
    logger.info(utils.generate_report_header("LIGHTWEIGHT DEEP CLUSTERING TRAINING"))
    
    try:
        df = utils.load_dataframe('engineered_static_features.csv')
    except:
        logger.error("engineered_static_features.csv not found. Run feature_engineering.py first.")
        return None, None
    
    user_col = 'user_pseudo' if 'user_pseudo' in df.columns else 'user'
    
    # --- FIX: Ensure feature names are loaded from the scaler ---
    detector = DeepClusteringLite(n_clusters=config.DEEP_CLUSTERING['n_clusters'], input_dim=0)
    if detector.scaler is None: return None, None
    
    feature_names = detector.scaler.get_feature_names_out()
    feature_names = [f for f in feature_names if f in df.columns]
    
    y = df['is_insider']
    X = df[feature_names] # Use only scaled features
    
    n_features = X.shape[1]
    detector.input_dim = n_features
    logger.info(f"Dataset: {len(X)} samples, {n_features} features")

    # Split data using stratify
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=(1 - config.TRAIN_RATIO), 
        random_state=config.RANDOM_SEED, 
        stratify=y if y.sum() > 1 else None
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=(config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)), 
        random_state=config.RANDOM_SEED,
        stratify=y_temp if y_temp.sum() > 1 else None
    )
    
    logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    logger.info(f"Insider labels split: Train={y_train.sum()}, Val={y_val.sum()}, Test={y_test.sum()}")
    
    # Train model
    import time
    start_time = time.time()
    detector.train(X_train.values) # K-Means prefers numpy arrays
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Calibrate threshold
    if y_val.sum() > 0 or len(y_val) > 0:
        detector.calibrate_threshold(X_val.values, y_val.values, target_fpr=0.05)
    else:
        logger.warning("No validation samples. Using default threshold.")
        detector.threshold = 0.95
    
    # Evaluate on test set
    metrics = detector.evaluate(X_test.values, y_test.values)
    
    detector.save_model()
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df['model'] = 'Deep Clustering (Lightweight)'
    metrics_df.to_csv(config.RESULTS_DIR / 'deep_clustering_metrics.csv', index=False)
    
    predictions, scores = detector.predict(X_test.values)
    results_df = pd.DataFrame({'true_label': y_test, 'prediction': predictions, 'anomaly_score': scores})
    results_df.to_csv(config.RESULTS_DIR / 'deep_clustering_predictions.csv', index=False)
    
    logger.info("Lightweight deep clustering training and evaluation completed!")
    logger.info(f"Total time: {training_time:.2f} seconds")
    
    return detector, metrics

if __name__ == "__main__":
    main()