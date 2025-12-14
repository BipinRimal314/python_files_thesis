"""
LSTM Autoencoder for Insider Threat Detection (Anomaly Detection)
Learns normal sequential patterns and flags deviations.
"""

import os
# Prevent TensorFlow from using Metal GPU which can cause hangs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU to avoid GPU issues

import numpy as np
import pandas as pd
import tensorflow as tf

# Disable GPU for stability on Mac
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, RepeatVector, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
import config
import utils

logger = utils.logger

def build_lstm_autoencoder(input_shape):
    """Build LSTM Autoencoder architecture"""
    # Encoder
    inputs = Input(shape=input_shape)
    encoded = LSTM(config.LSTM_AUTOENCODER['lstm_units'][0], activation='relu', return_sequences=True)(inputs)
    encoded = Dropout(config.LSTM_AUTOENCODER['dropout_rate'])(encoded)
    encoded = LSTM(config.LSTM_AUTOENCODER['lstm_units'][1], activation='relu', return_sequences=False)(encoded)
    
    # Latent space representation
    decoded = RepeatVector(input_shape[0])(encoded)
    
    # Decoder
    decoded = LSTM(config.LSTM_AUTOENCODER['lstm_units'][1], activation='relu', return_sequences=True)(decoded)
    decoded = Dropout(config.LSTM_AUTOENCODER['dropout_rate'])(decoded)
    decoded = LSTM(config.LSTM_AUTOENCODER['lstm_units'][0], activation='relu', return_sequences=True)(decoded)
    
    # Output
    output = TimeDistributed(Dense(input_shape[1]))(decoded)
    
    model = Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LSTM_AUTOENCODER['learning_rate']), 
                  loss='mse')
    
    return model

def load_data():
    """Load sequences and labels"""
    seq_path = config.PROCESSED_DATA_DIR / 'sequences.npy'
    label_path = config.SEQUENCE_LABELS_FILE
    
    if not seq_path.exists():
        logger.error(f"Sequence data not found at {seq_path}")
        return None, None
        
    try:
        X = np.load(seq_path)
        y = np.load(label_path) if label_path.exists() else np.zeros(len(X))
        
        # --- FAST DEBUGGING FIX ---
        # If dataset is massive, slice it for testing pipeline flow
        max_samples = getattr(config, 'MAX_SEQUENCE_SAMPLES', None)
        if max_samples and len(X) > max_samples:
            logger.warning(f"DEBUG MODE: Slicing data from {len(X)} to {max_samples} samples.")
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
            y = y[indices]
        # --------------------------
        
        logger.info(f"Loaded sequences: {X.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Error loading sequences: {e}")
        return None, None

def main():
    logger.info(utils.generate_report_header("LSTM AUTOENCODER TRAINING"))
    
    # 1. Load Data
    X, y = load_data()
    if X is None:
        raise FileNotFoundError("Training data not found.")

    # 2. Split Data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - config.TRAIN_RATIO), random_state=config.RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config.RANDOM_SEED
    )
    
    # Filter normal data for training
    X_train_normal = X_train[y_train == 0]
    X_val_normal = X_val[y_val == 0]
    
    logger.info(f"Training on {len(X_train_normal)} normal sequences")
    
    # 3. Build Model
    input_shape = (X.shape[1], X.shape[2]) 
    model = build_lstm_autoencoder(input_shape)
    
    if config.LSTM_AUTOENCODER['verbose']:
        model.summary(print_fn=logger.info)
    
    # 4. Train
    callbacks = [
        EarlyStopping(patience=config.LSTM_AUTOENCODER['patience'], restore_best_weights=True),
        ModelCheckpoint(filepath=str(config.MODEL_PATHS['lstm_autoencoder']), save_best_only=True)
    ]
    
    epochs = config.LSTM_AUTOENCODER['epochs']
    logger.info(f"Starting training for {epochs} epoch(s)...")
    
    history = model.fit(
        X_train_normal, X_train_normal,
        epochs=epochs,
        batch_size=config.LSTM_AUTOENCODER['batch_size'],
        validation_data=(X_val_normal, X_val_normal),
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Evaluate
    logger.info("Evaluating on Test Set...")
    X_test_pred = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
    
    anomaly_scores = np.mean(test_mae_loss, axis=1)
    threshold = np.percentile(anomaly_scores, 95)
    predictions = (anomaly_scores > threshold).astype(int)
    
    # Save Results
    results_df = pd.DataFrame({
        'true_label': y_test,
        'prediction': predictions,
        'anomaly_score': anomaly_scores
    })
    
    output_path = config.RESULTS_DIR / 'lstm_autoencoder_predictions.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # Metrics
    metrics = utils.calculate_metrics(y_test, predictions, anomaly_scores)
    utils.print_metrics(metrics, "LSTM Autoencoder")
    
    return model, metrics

if __name__ == "__main__":
    main()