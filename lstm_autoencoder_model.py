"""
LSTM Autoencoder for Sequential Anomaly Detection - LIGHTWEIGHT VERSION
[--- UPDATED VERSION ---]
- Fixes SyntaxError
- Fixes Keras save error by using .keras extension (from config)
- Uses stratify split to ensure insider sequences are in the test set
- Loads the correct 'daily_scaler.pkl'
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
import config
import utils

logger = utils.logger

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available, using statistical fallback method")

class LSTMAutoencoderLite:
    
    def __init__(self, sequence_length: int, n_features: int, **kwargs):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.params = config.LSTM_AUTOENCODER
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
        
        # --- NEW: Load the correct scaler ---
        try:
            self.scaler = utils.load_model(config.MODELS_DIR / 'daily_scaler.pkl')
        except FileNotFoundError:
            logger.error("daily_scaler.pkl not found! Run feature_engineering.py first.")
            self.scaler = None
        
        if not TF_AVAILABLE:
            self.mean_sequences = None
            self.std_sequences = None
        
        logger.info(f"Initialized Lightweight LSTM Autoencoder ({sequence_length}, {n_features})")
    
    def build_model(self):
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, using statistical method")
            return None
        
        logger.info("Building lightweight LSTM Autoencoder...")
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
        
        self.model = Model(inputs=inputs, outputs=outputs, name='LSTM_Autoencoder_Lite')
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        logger.info(f"Model built with {self.model.count_params():,} parameters")
        return self.model
    
    def train(self, X_train: np.ndarray, X_val: np.ndarray = None):
        logger.info("Training lightweight LSTM Autoencoder...")
        if not TF_AVAILABLE:
            logger.info("Using statistical fallback method...")
            self.mean_sequences = np.mean(X_train, axis=0)
            self.std_sequences = np.std(X_train, axis=0) + 1e-10
            self.is_trained = True
            logger.info("Statistical model training completed"); return None
        
        if self.model is None: self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
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
        logger.info("Training completed"); return history
    
    def calculate_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained")
        
        if not TF_AVAILABLE:
            deviations = np.abs(X - self.mean_sequences) / self.std_sequences
            reconstruction_errors = np.mean(deviations, axis=(1, 2))
        else:
            X_reconstructed = self.model.predict(X, verbose=0, batch_size=256)
            reconstruction_errors = np.mean(np.square(X - X_reconstructed), axis=(1, 2))
        
        return reconstruction_errors
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        reconstruction_errors = self.calculate_reconstruction_error(X)
        anomaly_scores = utils.normalize_scores(reconstruction_errors)
        
        if self.threshold is not None:
            predictions = (anomaly_scores > self.threshold).astype(int)
        else:
            predictions = (anomaly_scores > 0.95).astype(int)
        
        return predictions, anomaly_scores
    
    def calibrate_threshold(self, X_val: np.ndarray, y_val: np.ndarray, 
                           target_fpr: float = 0.05) -> float:
        logger.info(f"Calibrating threshold for target FPR={target_fpr}")
        reconstruction_errors = self.calculate_reconstruction_error(X_val)
        anomaly_scores = utils.normalize_scores(reconstruction_errors)
        
        normal_scores = anomaly_scores[y_val == 0]
        if len(normal_scores) == 0:
            logger.warning("No normal samples in validation set. Using default 0.95")
            self.threshold = 0.95
            return self.threshold

        normal_scores_sorted = np.sort(normal_scores)
        threshold_idx = int(len(normal_scores_sorted) * (1 - target_fpr))
        self.threshold = normal_scores_sorted[threshold_idx] if threshold_idx < len(normal_scores_sorted) else 0.95
        
        logger.info(f"Calibrated threshold: {self.threshold:.4f}")
        return self.threshold
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        logger.info("Evaluating lightweight LSTM Autoencoder...")
        predictions, scores = self.predict(X_test)
        metrics = utils.calculate_metrics(y_test, predictions, scores)
        reconstruction_errors = self.calculate_reconstruction_error(X_test)
        metrics['mean_reconstruction_error'] = float(np.mean(reconstruction_errors))
        metrics['std_reconstruction_error'] = float(np.std(reconstruction_errors))
        utils.print_metrics(metrics, "LSTM Autoencoder (Lightweight)")
        return metrics
    
    def save_model(self, filepath = None):
        if filepath is None:
            filepath = config.MODEL_PATHS['lstm_autoencoder']
        
        from pathlib import Path
        if isinstance(filepath, Path): filepath = str(filepath)
        
        # --- FIX: Keras models must end in .keras or .h5 ---
        if not (filepath.endswith('.keras') or filepath.endswith('.h5')):
            filepath = str(Path(filepath).with_suffix('.keras'))
            logger.warning(f"Filepath updated to {filepath} for Keras model")

        if TF_AVAILABLE and self.model is not None:
            self.model.save(filepath)
        else:
            # --- FIX: This is the fallback for the statistical method ---
            import pickle
            model_data = {
                'mean_sequences': self.mean_sequences,
                'std_sequences': self.std_sequences,
                'threshold': self.threshold,
                'sequence_length': self.sequence_length,
                'n_features': self.n_features
            }
            filepath = str(Path(filepath).with_suffix('.pkl'))
            with open(filepath, 'wb') as f: 
                pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath = None):
        if filepath is None:
            filepath = config.MODEL_PATHS['lstm_autoencoder']
        
        from pathlib import Path
        if isinstance(filepath, Path): filepath = str(filepath)
        
        if TF_AVAILABLE:
            try:
                # --- FIX: Ensure it loads the .keras file if it exists ---
                if not (filepath.endswith('.keras') or filepath.endswith('.h5')):
                    filepath = str(Path(filepath).with_suffix('.keras'))
                
                self.model = keras.models.load_model(filepath)
                self.is_trained = True
                logger.info(f"Keras model loaded from {filepath}")
            except Exception as e:
                logger.warning(f"Failed to load Keras model ({e}). Trying statistical fallback.")
                try:
                    import pickle
                    filepath = str(Path(filepath).with_suffix('.pkl'))
                    with open(filepath, 'rb') as f:
                        model_data = pickle.load(f)
                    self.mean_sequences = model_data['mean_sequences']
                    self.std_sequences = model_data['std_sequences']
                    self.threshold = model_data.get('threshold')
                    self.is_trained = True
                    logger.info(f"Statistical fallback model loaded from {filepath}")
                except Exception as e2:
                    logger.error(f"Failed to load any model from {filepath}: {e2}")
        else:
            import pickle
            filepath = str(Path(filepath).with_suffix('.pkl'))
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                self.mean_sequences = model_data['mean_sequences']
                self.std_sequences = model_data['std_sequences']
                self.threshold = model_data.get('threshold')
                self.is_trained = True
                logger.info(f"Statistical model loaded from {filepath}")
            except Exception as e:
                logger.error(f"Failed to load statistical model: {e}")


def main():
    logger.info(utils.generate_report_header("LIGHTWEIGHT LSTM AUTOENCODER TRAINING"))
    
    try:
        X = np.load(config.PROCESSED_DATA_DIR / 'sequences.npy')
        y = np.load(config.PROCESSED_DATA_DIR / 'sequence_labels.npy')
    except:
        logger.error("Sequence data not found. Run feature_engineering.py first.")
        return None, None
    
    logger.info(f"Loaded sequences: {X.shape}")
    
    sequence_length, n_features = X.shape[1], X.shape[2]
    
    # Split data using stratify
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=config.RANDOM_SEED, 
        stratify=y if y.sum() > 1 else None
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.5, 
        random_state=config.RANDOM_SEED, 
        stratify=y_temp if y_temp.sum() > 1 else None
    )
    
    logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    logger.info(f"Insider labels split: Train={y_train.sum()}, Val={y_val.sum()}, Test={y_test.sum()}")
    
    # Filter training data to normal behavior only
    X_train_normal = X_train[y_train == 0]
    X_val_normal = X_val[y_val == 0] if len(X_val[y_val == 0]) > 0 else X_val
    
    logger.info(f"Training on {len(X_train_normal)} normal sequences")
    
    # Initialize model
    autoencoder = LSTMAutoencoderLite(sequence_length, n_features)
    if autoencoder.scaler is None: return None, None # Stop if scaler failed to load
    
    # Train model
    import time
    start_time = time.time()
    history = autoencoder.train(X_train_normal, X_val_normal)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Calibrate threshold
    if y_val.sum() > 0 or len(y_val) > 0:
        autoencoder.calibrate_threshold(X_val, y_val, target_fpr=0.05)
    else:
        logger.warning("No validation samples. Using default threshold.")
        autoencoder.threshold = 0.95
    
    # Evaluate on test set
    metrics = autoencoder.evaluate(X_test, y_test)
    
    autoencoder.save_model()
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df['model'] = 'LSTM Autoencoder (Lightweight)'
    metrics_df.to_csv(config.RESULTS_DIR / 'lstm_autoencoder_metrics.csv', index=False)
    
    predictions, scores = autoencoder.predict(X_test)
    results_df = pd.DataFrame({'true_label': y_test, 'prediction': predictions, 'anomaly_score': scores})
    results_df.to_csv(config.RESULTS_DIR / 'lstm_autoencoder_predictions.csv', index=False)
    
    logger.info("Lightweight LSTM Autoencoder completed!")
    logger.info(f"Total time: {training_time:.2f} seconds")
    
    return autoencoder, metrics

if __name__ == "__main__":
    main()