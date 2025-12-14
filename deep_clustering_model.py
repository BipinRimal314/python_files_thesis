"""
Deep Clustering for Insider Threat Detection
Combines Autoencoder reconstruction loss with clustering distance.
"""

import os
# Prevent TensorFlow from using Metal GPU which can cause hangs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU to avoid GPU issues

import pandas as pd
import numpy as np

import tensorflow as tf
# Disable GPU for stability on Mac
tf.config.set_visible_devices([], 'GPU')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib # Standardized on joblib
import config
import utils

logger = utils.logger

def load_data():
    """Load daily features"""
    file_path = config.DAILY_FEATURES_FILE
    if not file_path.exists():
        logger.error(f"Daily features not found at {file_path}")
        return None
    return pd.read_parquet(file_path)

def load_static_scaler():
    """Load the fitted scaler for static features using joblib"""
    scaler_path = config.MODEL_PATHS['static_scaler']
    
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            return scaler
        except Exception as e:
            logger.warning(f"Could not load static scaler ({e}). Will fit new one.")
            return None
    return None

def main():
    logger.info(utils.generate_report_header("LIGHTWEIGHT DEEP CLUSTERING TRAINING"))
    
    # 1. Load Data
    df = load_data()
    if df is None:
        return None, None
        
    # Drop non-numeric
    exclude_cols = ['user', 'day', 'is_anomaly', 'is_insider']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values
    
    # 2. Scaling
    # We try to load the global scaler, but if dimensions mismatch (daily vs static), we fit a new one.
    scaler = load_static_scaler()
    
    try:
        X_scaled = scaler.transform(X) if scaler else StandardScaler().fit_transform(X)
    except:
        # Fallback if scaler expects different features
        X_scaled = StandardScaler().fit_transform(X)
        
    # 3. Clustering (Simplified Deep Clustering)
    input_dim = X_scaled.shape[1]
    
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded) # Latent space
    
    # Decoder
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    ae = Model(input_layer, decoded)
    ae.compile(optimizer='adam', loss='mse')
    
    # Train Autoencoder
    ae.fit(X_scaled, X_scaled, 
           epochs=10, 
           batch_size=256,
           shuffle=True,
           verbose=0)
    
    # Extract Latent Features
    encoder = Model(input_layer, encoded)
    X_latent = encoder.predict(X_scaled)
    
    # 3b. KMeans on Latent Space
    kmeans = KMeans(
        n_clusters=config.DEEP_CLUSTERING['n_clusters'],
        random_state=config.RANDOM_SEED,
        n_init=10
    )
    clusters = kmeans.fit_predict(X_latent)
    
    # 4. Anomaly Detection Logic
    centers = kmeans.cluster_centers_
    distances = np.linalg.norm(X_latent - centers[clusters], axis=1)
    
    X_recon = ae.predict(X_scaled)
    recon_error = np.mean(np.square(X_scaled - X_recon), axis=1)
    
    # Combine (Normalize both first)
    dist_norm = (distances - distances.min()) / (distances.max() - distances.min())
    recon_norm = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min())
    
    final_scores = 0.5 * dist_norm + 0.5 * recon_norm
    
    # Threshold
    threshold = np.percentile(final_scores, 95)
    predictions = (final_scores > threshold).astype(int)
    
    # 5. Save Results
    results_df = df[['user', 'day']].copy()
    if 'is_anomaly' in df.columns:
        results_df['true_label'] = df['is_anomaly']
    else:
        results_df['true_label'] = 0
        
    results_df['prediction'] = predictions
    results_df['anomaly_score'] = final_scores
    
    output_path = config.RESULTS_DIR / 'deep_clustering_predictions.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # Metrics
    metrics = utils.calculate_metrics(results_df['true_label'], predictions, final_scores)
    utils.print_metrics(metrics, "Deep Clustering")
    
    return kmeans, metrics

if __name__ == "__main__":
    main()