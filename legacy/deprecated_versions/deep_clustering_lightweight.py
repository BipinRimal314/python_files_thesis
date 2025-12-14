"""
Model 3: Deep Clustering for Anomaly Detection - LIGHTWEIGHT VERSION
Trains on static, per-user statistical features.
This version uses a correct unsupervised evaluation methodology:
- Train on 100% normal data.
- Test on the full dataset (normal + anomalous).
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
import config
import utils

# Try to use TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger = utils.logger # Need to init logger if TF fails
    logger.warning("TensorFlow not available. Deep Clustering model will not function.")

if TF_AVAILABLE:
    logger = utils.logger

class DeepClusteringDetectorLite:
    """
    Lightweight Deep Clustering model for unsupervised anomaly detection.
    Trains an autoencoder and then a clustering layer on the latent space.
    """
    
    def __init__(self, n_clusters: int, input_dim: int, **kwargs):
        """
        Initialize the model.
        
        Args:
            n_clusters: Number of clusters (from config)
            input_dim: Number of features (e.g., 70)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for the DeepClustering model.")
            
        self.n_clusters = n_clusters
        self.input_dim = input_dim
        
        # Lightweight parameters from config
        self.params = {**config.DEEP_CLUSTERING, **kwargs}
        self.encoding_dims = self.params['encoding_dims']
        self.encoding_dim = self.encoding_dims[-1]  # Final bottleneck dimension
        
        self.autoencoder = None
        self.encoder = None
        self.model = None
        self.cluster_centers = None
        self.threshold = None
        self.scaler = None
        
        logger.info(f"Initialized Lightweight Deep Clustering (n_clusters={n_clusters}, encoding_dim={self.encoding_dim})")
        
        # Build the autoencoder
        self._build_autoencoder()

    def _build_autoencoder(self):
        """Builds the autoencoder and encoder models."""
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,), name='input')
        
        # Encoder
        encoded = input_layer
        for dim in self.encoding_dims:
            encoded = layers.Dense(dim, activation=self.params['activation'])(encoded)
            encoded = layers.Dropout(self.params['dropout_rate'])(encoded)
        
        # Latent space (bottleneck)
        latent_space = layers.Dense(self.encoding_dim, activation=self.params['activation'], name='encoder_output')(encoded)
        
        # Decoder
        decoded = latent_space
        for dim in reversed(self.encoding_dims[:-1]):
            decoded = layers.Dense(dim, activation=self.params['activation'])(decoded)
        
        # Output layer
        output_layer = layers.Dense(self.input_dim, activation='sigmoid', name='decoder_output')(decoded)
        
        # Autoencoder model
        self.autoencoder = Model(inputs=input_layer, outputs=output_layer, name='Autoencoder')
        
        # Encoder model
        self.encoder = Model(inputs=input_layer, outputs=latent_space, name='Encoder')
        
        # Compile autoencoder for pre-training
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss='mse'
        )

    def pretrain_autoencoder(self, X_train_normal: np.ndarray, X_val_normal: np.ndarray):
        """
        Pre-train the autoencoder on normal data.
        
        Args:
            X_train_normal: Training features (normal data).
            X_val_normal: Validation features (normal data).
        """
        logger.info("Pre-training autoencoder...")
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.params['patience'],
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        self.autoencoder.fit(
            X_train_normal, X_train_normal,
            validation_data=(X_val_normal, X_val_normal),
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            callbacks=callbacks,
            verbose=self.params['verbose']
        )
        logger.info("Autoencoder pre-training completed")

    def fit_clustering(self, X_train_normal: np.ndarray):
        """
        Fit K-Means to the latent space of the pre-trained autoencoder.
        
        Args:
            X_train_normal: Training features (normal data).
        """
        logger.info("Fitting MiniBatch K-Means clustering...")
        # Get latent space representations
        X_train_encoded = self.encoder.predict(X_train_normal, batch_size=self.params['batch_size'], verbose=0)
        
        # Fit K-Means
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=config.RANDOM_SEED, n_init=10)
        kmeans.fit(X_train_encoded)
        self.cluster_centers = kmeans.cluster_centers_
        logger.info("Lightweight deep clustering training completed")

    def train(self, X_train_normal: np.ndarray, X_val_normal: np.ndarray):
        """
        Train the full Deep Clustering model.
        
        Args:
            X_train_normal: Training features (normal data).
            X_val_normal: Validation features (normal data).
        """
        logger.info("Training lightweight deep clustering model...")
        
        # 1. Pre-train autoencoder
        self.pretrain_autoencoder(X_train_normal, X_val_normal)
        
        # 2. Fit K-Means
        self.fit_clustering(X_train_normal)

    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores as the distance to the nearest cluster center
        in the latent space.
        
        Args:
            X: Input features.
            
        Returns:
            Array of anomaly scores.
        """
        if self.encoder is None or self.cluster_centers is None:
            raise ValueError("Model must be trained first.")
            
        # Get latent space representations
        X_encoded = self.encoder.predict(X, batch_size=self.params['batch_size'], verbose=0)
        
        # Calculate distance to each cluster center
        distances = []
        for i in range(self.n_clusters):
            center = self.cluster_centers[i]
            dist = np.linalg.norm(X_encoded - center, axis=1)
            distances.append(dist)
            
        # Anomaly score is the minimum distance (distance to closest cluster)
        anomaly_scores = np.min(distances, axis=0)
        
        return anomaly_scores

    def calibrate_threshold(self, X_val_normal: np.ndarray, y_val_normal: pd.Series, target_fpr: float = 0.05):
        """
        Calibrate the anomaly threshold using a validation set of *normal* data.
        
        Args:
            X_val_normal: Validation features (normal data).
            y_val_normal: Validation labels (all 0s).
            target_fpr: The desired False Positive Rate to set the threshold.
        """
        logger.info(f"Calibrating threshold for target FPR={target_fpr}...")
        anomaly_scores = self.get_anomaly_scores(X_val_normal)
        
        # Find the score at the (1 - FPR) percentile
        self.threshold = np.percentile(anomaly_scores, 100 * (1 - target_fpr))
        logger.info(f"Calibrated threshold: {self.threshold:.4f}")
        return self.threshold

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies on new data.
        
        Args:
            X: DataFrame of features to predict on.
            
        Returns:
            Tuple of (predictions, anomaly_scores)
        """
        anomaly_scores = self.get_anomaly_scores(X)
        
        if self.threshold is None:
            logger.warning("Threshold not set, using 95th percentile as default.")
            self.threshold = np.percentile(anomaly_scores, 95)
            
        predictions = (anomaly_scores > self.threshold).astype(int)
        
        # Normalize scores for reporting
        anomaly_scores_normalized = utils.normalize_scores(anomaly_scores)
        
        return predictions, anomaly_scores_normalized

    def evaluate(self, X_test: np.ndarray, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on a labeled test set.
        
        Args:
            X_test: Test features.
            y_test: Test labels (contains both 0s and 1s).
            
        Returns:
            Dictionary of performance metrics.
        """
        logger.info("Evaluating lightweight deep clustering model...")
        predictions, scores = self.predict(X_test)
        
        # Check if there are any positive labels in the test set
        if y_test.sum() == 0:
            logger.warning("No positive samples in test set. Metrics (Precision, Recall, F1, AUC-ROC) will be 0 or NaN.")
            metrics = utils.calculate_metrics(y_test, predictions, scores)
        else:
            logger.info(f"Test set contains {y_test.sum()} positive samples.")
            # We can now calculate meaningful metrics
            # --- THIS IS THE FIX ---
            metrics = utils.calculate_metrics(y_test, predictions, scores)
            # --- END OF FIX ---
        
        utils.print_metrics(metrics, "Deep Clustering (Lightweight)")
        return metrics

    def save_model(self):
        """Save the trained model and scaler to disk."""
        # Save the K-Means cluster centers and threshold
        model_data = {
            'cluster_centers': self.cluster_centers,
            'threshold': self.threshold,
            'params': self.params
        }
        utils.save_model(model_data, config.MODEL_PATHS['deep_clustering'])
        
        # Save the encoder model (which is the valuable part)
        encoder_path = str(config.MODEL_PATHS['deep_clustering']).replace('.pkl', '_encoder.keras')
        self.encoder.save(encoder_path)
        logger.info(f"Encoder model saved to {encoder_path}")


def main():
    """Main execution function"""
    logger.info(utils.generate_report_header("LIGHTWEIGHT DEEP CLUSTERING TRAINING"))
    
    if not TF_AVAILABLE:
        logger.error("TensorFlow not found. Cannot run Deep Clustering model.")
        return None, {}

    # --- Data Loading ---
    try:
        df = utils.load_dataframe('engineered_static_features.csv')
    except Exception as e:
        logger.error(f"Failed to load 'engineered_static_features.csv'. Run feature_engineering.py first. Error: {e}")
        return None, None
        
    # Load the scaler used for this data
    try:
        scaler = utils.load_model(config.MODEL_PATHS['static_scaler'])
    except Exception as e:
        logger.error(f"Failed to load 'static_feature_scaler.pkl'. Run feature_engineering.py first. Error: {e}")
        return None, None
        
    # Define feature columns and label
    user_col = 'user_pseudo' if 'user_pseudo' in df.columns else 'user'
    label_col = 'is_insider'
    feature_cols = [col for col in df.columns if col not in [user_col, label_col]]
    
    X = df[feature_cols]
    y = df[label_col]
    
    # Scale data (Autoencoders are sensitive to scale, 0-1 is best)
    # We'll use the existing scaler but also apply a Min-Max scaler
    from sklearn.preprocessing import MinMaxScaler
    X_scaled = MinMaxScaler().fit_transform(X)
    
    logger.info(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
    
    # --- New Experimental Design ---
    # 1. Split data into "normal" (for training) and "full" (for testing)
    X_normal = X_scaled[y == 0]
    y_normal = y[y == 0]
    
    # 2. Create a separate, smaller validation set *from the normal data*
    X_train_normal, X_val_normal, y_train_normal, y_val_normal = train_test_split(
        X_normal, y_normal, test_size=0.2, random_state=config.RANDOM_SEED
    )
    
    # 3. The Test Set is the *full, original* dataset
    X_test = X_scaled
    y_test = y
    
    logger.info(f"Training on: {len(X_train_normal)} normal samples")
    logger.info(f"Calibrating on: {len(X_val_normal)} normal samples")
    logger.info(f"Testing on: {len(X_test)} total samples (including {y_test.sum()} insiders)")

    # --- Model Training ---
    detector = DeepClusteringDetectorLite(
        n_clusters=config.DEEP_CLUSTERING['n_clusters'],
        input_dim=len(feature_cols)
    )
    
    import time
    start_time = time.time()
    
    # 1. Train *only* on normal data
    detector.train(X_train_normal, X_val_normal)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # 2. Calibrate threshold *only* on normal validation data
    detector.calibrate_threshold(X_val_normal, y_val_normal, target_fpr=0.05)
    
    # 3. Evaluate on the *full* test set (which includes the insider)
    metrics = detector.evaluate(X_test, y_test)
    
    # --- Save Results ---
    detector.save_model()
    
    # Save metrics for model_evaluation.py
    metrics_df = pd.DataFrame([metrics])
    metrics_df['model'] = 'deep_clustering'
    metrics_df.to_csv(config.RESULTS_DIR / 'deep_clustering_metrics.csv', index=False)
    
    # Save predictions for ensemble
    predictions, scores = detector.predict(X_test)
    results_df = pd.DataFrame({
        'user': df[user_col],
        'true_label': y_test,
        'prediction': predictions,
        'anomaly_score': scores
    })
    results_df.to_csv(config.RESULTS_DIR / 'deep_clustering_predictions.csv', index=False)
    
    logger.info("Lightweight deep clustering training and evaluation completed!")
    logger.info(f"Total time: {training_time:.2f} seconds")
    
    return detector, metrics

if __name__ == "__main__":
    main()