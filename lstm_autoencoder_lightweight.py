"""
LSTM Autoencoder for Sequential Anomaly Detection - LIGHTWEIGHT VERSION
Optimized for MacBook M4 Pro with reduced model complexity
This version uses a correct unsupervised evaluation methodology:
- Train on 100% normal data.
- Test on the full dataset (normal + anomalous).
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
import config
import utils

logger = utils.logger

# Try to use TensorFlow, fall back to simpler method if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available, using statistical fallback method")

class LSTMAutoencoderLite:
    """
    Lightweight LSTM Autoencoder for detecting sequential anomalies
    Uses smaller architecture and reduced training for laptop efficiency
    """
    
    def __init__(self, sequence_length: int, n_features: int, **kwargs):
        """
        Initialize Lightweight LSTM Autoencoder
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features per timestep
            **kwargs: Override default parameters
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        
        # Lightweight parameters
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
        
        if not TF_AVAILABLE:
            # Fallback to statistical method
            self.mean_sequences = None
            self.std_sequences = None
        
        logger.info(f"Initialized Lightweight LSTM Autoencoder ({sequence_length}, {n_features})")
    
    def build_model(self):
        """Build lightweight LSTM Autoencoder"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, using statistical method")
            return None
        
        logger.info("Building lightweight LSTM Autoencoder...")
        
        # Input
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        
        # Encoder - simplified
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
        self.model = Model(inputs=inputs, outputs=outputs, name='LSTM_Autoencoder_Lite')
        
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
        logger.info("Training lightweight LSTM Autoencoder...")
        
        if not TF_AVAILABLE:
            # Statistical fallback - use mean and std
            logger.info("Using statistical fallback method...")
            self.mean_sequences = np.mean(X_train, axis=0)
            self.std_sequences = np.std(X_train, axis=0) + 1e-10
            self.is_trained = True
            logger.info("Statistical model training completed")
            return None
        
        if self.model is None:
            self.build_model()
        
        # Callbacks - reduced patience
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train
        logger.info(f"Training on {len(X_train)} sequences...")
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
        
        Args:
            X: Input sequences
            
        Returns:
            Array of reconstruction errors
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating reconstruction error")
        
        if not TF_AVAILABLE:
            # Statistical method - calculate deviation from mean
            deviations = np.abs(X - self.mean_sequences) / self.std_sequences
            reconstruction_errors = np.mean(deviations, axis=(1, 2))
        else:
            # Reconstruct sequences
            X_reconstructed = self.model.predict(X, verbose=0, batch_size=256)
            
            # Calculate MSE for each sequence
            reconstruction_errors = np.mean(np.square(X - X_reconstructed), axis=(1, 2))
        
        return reconstruction_errors
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies based on reconstruction error
        
        Args:
            X: Input sequences
            
        Returns:
            Tuple of (predictions, anomaly_scores)
        """
        # Get reconstruction errors
        reconstruction_errors = self.calculate_reconstruction_error(X)
        
        # Normalize to [0, 1]
        anomaly_scores = utils.normalize_scores(reconstruction_errors)
        
        # Apply threshold
        if self.threshold is not None:
            predictions = (anomaly_scores > self.threshold).astype(int)
        else:
            logger.warning("Threshold not calibrated, using 95th percentile as default.")
            default_threshold = np.percentile(anomaly_scores, 95)
            predictions = (anomaly_scores > default_threshold).astype(int)
        
        return predictions, anomaly_scores
    
    def calibrate_threshold(self, X_val_normal: np.ndarray, y_val_normal: np.ndarray, 
                           target_fpr: float = 0.05) -> float:
        """
        Calibrate decision threshold using a validation set of *normal* data.
        
        Args:
            X_val_normal: Validation features (normal data).
            y_val_normal: Validation labels (all 0s).
            target_fpr: The desired False Positive Rate to set the threshold.
        """
        logger.info(f"Calibrating threshold for target FPR={target_fpr}")
        
        reconstruction_errors = self.calculate_reconstruction_error(X_val_normal)
        anomaly_scores = utils.normalize_scores(reconstruction_errors)
        
        # Find the score at the (1 - FPR) percentile
        self.threshold = np.percentile(anomaly_scores, 100 * (1 - target_fpr))
        
        logger.info(f"Calibrated threshold: {self.threshold:.4f}")
        return self.threshold
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on a labeled test set.
        
        Args:
            X_test: Test features.
            y_test: Test labels (contains both 0s and 1s).
            
        Returns:
            Dictionary of performance metrics.
        """
        logger.info("Evaluating lightweight LSTM Autoencoder...")
        
        predictions, scores = self.predict(X_test)
        
        if y_test.sum() == 0:
            logger.warning("No positive samples in test set. Metrics (Precision, Recall, F1, AUC-ROC) will be 0 or NaN.")
            # (--- FIX ---) Corrected typo
            metrics = utils.calculate_metrics(y_test, predictions, scores)
        else:
            logger.info(f"Test set contains {y_test.sum()} positive samples.")
            # We can now calculate meaningful metrics
            # (--- FIX ---) Corrected typo
            metrics = utils.calculate_metrics(y_test, predictions, scores)

        reconstruction_errors = self.calculate_reconstruction_error(X_test)
        metrics['mean_reconstruction_error'] = float(np.mean(reconstruction_errors))
        metrics['std_reconstruction_error'] = float(np.std(reconstruction_errors))
        
        utils.print_metrics(metrics, "LSTM Autoencoder (Lightweight)")
        
        return metrics
    
    def save_model(self):
        """Save trained model"""
        filepath = config.MODEL_PATHS['lstm_autoencoder']
        
        # Convert Path to string
        from pathlib import Path
        if isinstance(filepath, Path):
            filepath = str(filepath)
        
        if TF_AVAILABLE and self.model is not None:
            # Save as .keras (fixes the ValueError)
            if not filepath.endswith(".keras"):
                logger.warning(f"Filepath in config is {filepath}, saving as .keras")
                filepath = str(Path(filepath).with_suffix('.keras'))
            
            self.model.save(filepath)
        else:
            # Save statistical model
            import pickle
            
            # (--- FIX ---) This is the line with the SyntaxError.
            # Replaced the `...` with the correct variable names.
            model_data = {
                'mean_sequences': self.mean_sequences,
                'std_sequences': self.std_sequences,
                'threshold': self.threshold,
                'sequence_length': self.sequence_length,
                'n_features': self.n_features
            }
            # (--- END FIX ---)
            
            # Ensure we save as .pkl if we are in fallback mode
            filepath = str(Path(filepath).with_suffix('.pkl'))
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self):
        """Load trained model"""
        filepath = config.MODEL_PATHS['lstm_autoencoder']
        
        # Convert Path to string
        from pathlib import Path
        if isinstance(filepath, Path):
            filepath = str(filepath)
        
        if TF_AVAILABLE:
            try:
                self.model = keras.models.load_model(filepath)
                self.is_trained = True
            except:
                # Try loading pickle file (fallback)
                try:
                    import pickle
                    pkl_path = str(Path(filepath).with_suffix('.pkl'))
                    with open(pkl_path, 'rb') as f:
                        model_data = pickle.load(f)
                    self.mean_sequences = model_data['mean_sequences']
                    self.std_sequences = model_data['std_sequences']
                    self.threshold = model_data.get('threshold')
                    self.is_trained = True
                    logger.warning(f"Loaded statistical fallback model from {pkl_path}")
                except Exception as e:
                    logger.error(f"Failed to load Keras model or PKL fallback: {e}")
        else:
            # Load statistical model
            import pickle
            pkl_path = str(Path(filepath).with_suffix('.pkl'))
            with open(pkl_path, 'rb') as f:
                model_data = pickle.load(f)
            self.mean_sequences = model_data['mean_sequences']
            self.std_sequences = model_data['std_sequences']
            self.threshold = model_data.get('threshold')
            self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")


def main():
    """Main execution function"""
    logger.info(utils.generate_report_header("LIGHTWEIGHT LSTM AUTOENCODER TRAINING"))
    
    # Load sequences
    try:
        X = np.load(config.PROCESSED_DATA_DIR / 'sequences.npy')
        y = np.load(config.PROCESSED_DATA_DIR / 'sequence_labels.npy')
    except Exception as e:
        logger.error(f"Sequence data not found. Run feature_engineering.py first. Error: {e}")
        return None, {}
    
    logger.info(f"Loaded sequences: {X.shape}, with {y.sum()} insider sequences")
    
    sequence_length = X.shape[1]
    n_features = X.shape[2]
    
    # --- New Experimental Design ---
    # 1. Separate normal and anomalous data
    X_normal = X[y == 0]
    y_normal = y[y == 0]
    X_insider = X[y == 1]
    y_insider = y[y == 1]
    
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
        scaler = utils.load_model(config.MODEL_PATHS['daily_scaler'])
    except Exception as e:
        logger.error(f"Failed to load 'daily_scaler.pkl'. Run feature_engineering.py first. Error: {e}")
        # Continue without scaler, but warn
        logger.warning("Proceeding without scaler, results may be poor.")
    
    # Initialize model
    autoencoder = LSTMAutoencoderLite(sequence_length, n_features)
    
    # --- Model Training ---
    import time
    start_time = time.time()
    
    # 1. Train *only* on normal data
    history = autoencoder.train(X_train_normal, X_val_normal)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # 2. Calibrate threshold *only* on normal validation data
    autoencoder.calibrate_threshold(X_val_normal, y_val_normal, target_fpr=0.05)
    
    # 3. Evaluate on the *full* test set (which includes the insider)
    metrics = autoencoder.evaluate(X_test, y_test)
    
    # --- Save Results ---
    autoencoder.save_model()
    
    # Save metrics for model_evaluation.py
    metrics_df = pd.DataFrame([metrics])
    metrics_df['model'] = 'lstm_autoencoder'
    metrics_df.to_csv(config.RESULTS_DIR / 'lstm_autoencoder_metrics.csv', index=False)
    
    # Get user IDs for predictions
    try:
        user_ids = np.load(config.PROCESSED_DATA_DIR / 'sequence_users.npy')
    except:
        logger.warning("Could not load 'sequence_users.npy'. Predictions will not be mapped to users.")
        # Create a placeholder array if file not found
        user_ids = ['unknown'] * len(y_test)
        if len(y_test) != len(user_ids): # This check is for the X_test slicing, not y_test
            user_ids = ['unknown'] * len(X_test)

    # Ensure user_ids array aligns with X_test (which is the full X)
    if len(user_ids) != len(X_test):
         logger.error(f"User IDs array length ({len(user_ids)}) does not match test set length ({len(X_test)}). Using placeholders.")
         user_ids = ['unknown'] * len(X_test)
    
    # Save predictions for ensemble
    predictions, scores = autoencoder.predict(X_test)
    results_df = pd.DataFrame({
        'user': user_ids, 
        'true_label': y_test,
        'prediction': predictions,
        'anomaly_score': scores
    })
    results_df.to_csv(config.RESULTS_DIR / 'lstm_autoencoder_predictions.csv', index=False)
    
    logger.info("Lightweight LSTM Autoencoder completed!")
    logger.info(f"Total time: {training_time:.2f} seconds")
    
    return autoencoder, metrics


if __name__ == "__main__":
    main()