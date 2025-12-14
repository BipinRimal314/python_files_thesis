"""
Model 3: LSTM Autoencoder (V2: Multi-Dataset)
Trains on the new, multi-dataset sequence files
which include all datasets and advanced Z-score features.

[--- V4.1 FINAL FIX ---]
- Dynamically builds the input filenames based on DATASET_SUBSET.
- Fixes `ValueError: Target is multiclass...` by converting the
  label array 'y' to .astype(int) after loading.
- Fixes `TypeError` in `calibrate_threshold` call.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
import joblib

# Use the V2 config
import config_v2 as config 
import utils

logger = utils.logger

# Try to use TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.error("TensorFlow not available. LSTM Autoencoder model will not function.")


class LSTMAutoencoderV2:
    """
    Lightweight LSTM Autoencoder for detecting sequential anomalies
    """
    
    def __init__(self, sequence_length: int, n_features: int, **kwargs):
        """
        Initialize Lightweight LSTM Autoencoder
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features per timestep
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for this model.")

        self.sequence_length = sequence_length
        self.n_features = n_features
        
        # Lightweight parameters from config
        self.params = {**config.LSTM_AUTOENCODER, **kwargs}
        self.lstm_units = self.params['lstm_units']
        self.encoding_dim = self.params['encoding_dim']
        self.dropout_rate = self.params['dropout_rate']
        self.learning_rate = self.params['learning_rate']
        self.batch_size = self.params['batch_size']
        self.epochs = self.params['epochs']
        self.patience = self.params['patience']
        
        self.model = None
        self.threshold = None
        self.is_trained = False
        
        logger.info(f"Initialized Lightweight LSTM Autoencoder (V2) ({sequence_length}, {n_features})")
    
    def build_model(self):
        """Build lightweight LSTM Autoencoder"""
        logger.info("Building lightweight LSTM Autoencoder (V2)...")
        
        # Input
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        
        # Encoder
        encoded = layers.LSTM(self.lstm_units[0], activation='tanh', return_sequences=True)(inputs)
        encoded = layers.Dropout(self.dropout_rate)(encoded)
        encoded = layers.LSTM(self.lstm_units[1], activation='tanh', return_sequences=False)(encoded)
        encoded = layers.Dropout(self.dropout_rate)(encoded)
        
        # Bottleneck
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.RepeatVector(self.sequence_length)(encoded)
        decoded = layers.LSTM(self.lstm_units[1], activation='tanh', return_sequences=True)(decoded)
        decoded = layers.Dropout(self.dropout_rate)(decoded)
        decoded = layers.LSTM(self.lstm_units[0], activation='tanh', return_sequences=True)(decoded)
        decoded = layers.Dropout(self.dropout_rate)(decoded)
        
        # Output
        outputs = layers.TimeDistributed(layers.Dense(self.n_features))(decoded)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='LSTM_Autoencoder_V2')
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Model built with {self.model.count_params():,} parameters")
        return self.model
    
    def train(self, X_train: np.ndarray, X_val: np.ndarray = None):
        """
        Train the LSTM Autoencoder
        
        Args:
            X_train: Training sequences (normal behavior only)
            X_val: Validation sequences
        """
        logger.info(f"Training lightweight LSTM Autoencoder on {len(X_train)} sequences...")
        
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("Training completed")
        
        return history
    
    def calculate_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for sequences
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating reconstruction error")
        
        X_reconstructed = self.model.predict(X, verbose=0, batch_size=256)
        reconstruction_errors = np.mean(np.square(X - X_reconstructed), axis=(1, 2))
        return reconstruction_errors
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies based on reconstruction error
        """
        reconstruction_errors = self.calculate_reconstruction_error(X)
        anomaly_scores = utils.normalize_scores(reconstruction_errors)
        
        if self.threshold is not None:
            predictions = (anomaly_scores > self.threshold).astype(int)
        else:
            logger.warning("Threshold not set, using 95th percentile as default.")
            default_threshold = np.percentile(anomaly_scores, 95)
            predictions = (anomaly_scores > default_threshold).astype(int)
        
        return predictions, anomaly_scores
    
    def calibrate_threshold(self, X_val_normal: np.ndarray, y_val_normal: np.ndarray, 
                           target_fpr: float = 0.05) -> float:
        """
        Calibrate decision threshold using a validation set of *normal* data.
        """
        logger.info(f"Calibrating threshold for target FPR={target_fpr}")
        
        reconstruction_errors = self.calculate_reconstruction_error(X_val_normal)
        anomaly_scores = utils.normalize_scores(reconstruction_errors)
        
        self.threshold = np.percentile(anomaly_scores, 100 * (1 - target_fpr))
        
        logger.info(f"Calibrated threshold: {self.threshold:.4f}")
        return self.threshold
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info("Evaluating lightweight LSTM Autoencoder (V2)...")
        
        predictions, scores = self.predict(X_test)
        
        if y_test.sum() > 0:
            logger.info(f"Test set contains {y_test.sum()} positive samples.")
            metrics = utils.calculate_metrics(y_test, predictions, scores)
        else:
            logger.warning("No positive samples in test set. Metrics (Precision, Recall, F1, AUC-ROC) will be 0 or NaN.")
            metrics = utils.calculate_metrics(y_test, predictions, scores)

        reconstruction_errors = self.calculate_reconstruction_error(X_test)
        metrics['mean_reconstruction_error'] = float(np.mean(reconstruction_errors))
        metrics['std_reconstruction_error'] = float(np.std(reconstruction_errors))
        
        utils.print_metrics(metrics, "LSTM Autoencoder (V2)")
        
        return metrics
    
    def save_model(self):
        """Save trained model"""
        filepath = config.MODEL_PATHS['lstm_autoencoder_v2']
        from pathlib import Path
        if isinstance(filepath, Path):
            filepath = str(filepath)
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self):
        """Load trained model"""
        filepath = config.MODEL_PATHS['lstm_autoencoder_v2']
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


def main():
    """Main execution function"""
    logger.info(utils.generate_report_header("LIGHTWEIGHT LSTM AUTOENCODER TRAINING (V2: MULTI-DATASET)"))
    
    if not TF_AVAILABLE:
        logger.error("TensorFlow not found. Cannot run LSTM Autoencoder model.")
        return None, {}

    # --- (V4.1 FILENAME FIX) ---
    subset = config.DATASET_SUBSET if hasattr(config, 'DATASET_SUBSET') and config.DATASET_SUBSET else []
    subset_name = "_".join(subset) if subset else "ALL"
    
    sequence_file = config.PROCESSED_DATA_DIR / f'sequences_{subset_name}_v2.npy'
    labels_file = config.PROCESSED_DATA_DIR / f'sequence_labels_{subset_name}_v2.npy'
    users_file = config.PROCESSED_DATA_DIR / f'sequence_users_{subset_name}_v2.npy'
    
    logger.info(f"Loading sequences from: {sequence_file.name}")
    # --- (END V4.1 FIX) ---

    # Load sequences
    try:
        X = np.load(sequence_file)
        # --- (V4.1 VALUEERROR FIX) ---
        # Convert labels to integer to prevent 'multiclass' error
        y = np.load(labels_file).astype(int)
        # --- (END V4.1 FIX) ---
        users = np.load(users_file)
    except Exception as e:
        logger.error(f"Sequence data not found. Run feature_engineering_v2.py first. Error: {e}")
        return None, {}
    
    logger.info(f"Loaded sequences: {X.shape}, with {y.sum()} insider sequences")
    
    sequence_length = X.shape[1]
    n_features = X.shape[2]
    
    # --- New Experimental Design ---
    # 1. Separate normal and anomalous data
    X_normal = X[y == 0]
    y_normal = y[y == 0]
    
    # 2. Split *only* the normal data for training and validation
    X_train_normal, X_val_normal, y_train_normal, y_val_normal = train_test_split(
        X_normal, y_normal, test_size=0.2, random_state=config.RANDOM_SEED
    )
    
    # 3. The Test Set is the *full, original* dataset
    X_test = X
    y_test = y
    
    logger.info(f"Training on: {len(X_train_normal)} normal sequences")
    logger.info(f"Calibrating on: {len(X_val_normal)} normal sequences")
    logger.info(f"Testing on: {len(X_test)} total sequences (including {y_test.sum()} insider sequences)")
    
    # Load the scaler
    try:
        scaler = utils.load_model(config.MODEL_PATHS['daily_scaler_v2'])
    except Exception as e:
        logger.error(f"Failed to load 'daily_scaler_v2.pkl'. Run feature_engineering_v2.py first. Error: {e}")
        return None, None
    
    # Initialize model
    autoencoder = LSTMAutoencoderV2(sequence_length, n_features)
    
    # --- Model Training ---
    import time
    start_time = time.time()
    
    # 1. Train *only* on normal data
    history = autoencoder.train(X_train_normal, X_val_normal)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # 2. Calibrate threshold *only* on normal validation data
    # --- (V4.1 TYPEERROR FIX) ---
    # Corrected the call to be purely positional
    autoencoder.calibrate_threshold(X_val_normal, y_val_normal)
    # --- (END V4.1 FIX) ---
    
    # 3. Evaluate on the *full* test set (which includes the insider)
    metrics = autoencoder.evaluate(X_test, y_test)
    
    # --- Save Results ---
    autoencoder.save_model()
    
    # Save metrics for model_evaluation_v2.py
    metrics_df = pd.DataFrame([metrics])
    metrics_df['model'] = 'lstm_autoencoder'
    
    # (V4.1 FILENAME FIX)
    metrics_file = config.RESULTS_DIR / f'lstm_autoencoder_metrics_{subset_name}_v2.csv'
    metrics_df.to_csv(metrics_file, index=False)
    
    # Save predictions for ensemble
    predictions, scores = autoencoder.predict(X_test)
    results_df = pd.DataFrame({
        'user_id': users, 
        'true_label': y_test,
        'prediction': predictions,
        'anomaly_score': scores
    })
    
    # (V4.1 FILENAME FIX)
    predictions_file = config.RESULTS_DIR / f'lstm_autoencoder_predictions_{subset_name}_v2.csv'
    results_df.to_csv(predictions_file, index=False)
    
    logger.info("Lightweight LSTM Autoencoder (V2) completed!")
    
    return autoencoder, metrics


if __name__ == "__main__":
    main()