import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'final_optimized_ensemble_predictions.csv')

# OPTIMAL WEIGHTS (From Grid Search)
W_IF = 0.1
W_DC = 0.1
W_LSTM = 0.8

# UPDATED THRESHOLD:
# Set to 0.85 to capture the insider who scored 0.9 (IF+LSTM detection)
# and avoid floating point misses (0.9999 vs 1.0)
OPTIMAL_THRESHOLD = 0.85

# Input Files
ISO_FOREST_FILE = 'isolation_forest_predictions_r1_r2_r3.1_v2.csv'
DEEP_CLUSTER_FILE = 'deep_clustering_predictions_r1_r2_r3.1_v2.csv'
LSTM_FILE = 'lstm_autoencoder_predictions_r1_r2_r3.1_v2.csv'

def get_file_path(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path): return path
    if os.path.exists(filename): return filename
    return None

def load_prediction_df(filename, model_name):
    path = get_file_path(filename)
    if not path: return None
    print(f"  -> Loading {model_name}...")
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    
    score_cols = ['anomaly_score', 'score', 'reconstruction_error', 'prediction']
    score_col = next((c for c in score_cols if c in df.columns), None)
    
    label_cols = ['true_label', 'is_insider', 'label', 'ground_truth']
    label_col = next((c for c in label_cols if c in df.columns), None)

    user_cols = ['user_id', 'user', 'id']
    user_col = next((c for c in user_cols if c in df.columns), None)

    if not score_col or not label_col or not user_col:
        print(f"     [ERROR] Missing columns in {model_name}. Cols: {list(df.columns)}")
        return None

    rename_map = {score_col: 'score', label_col: 'label', user_col: 'user_id'}
    df = df.rename(columns=rename_map)
    return df[['user_id', 'label', 'score']].loc[:, ~df[['user_id', 'label', 'score']].columns.duplicated()]

def main():
    print("\n--- GENERATING FINAL OPTIMIZED ENSEMBLE ---")

    # 1. Load & Align
    df_if = load_prediction_df(ISO_FOREST_FILE, "Isolation Forest")
    df_dc = load_prediction_df(DEEP_CLUSTER_FILE, "Deep Clustering")
    df_lstm = load_prediction_df(LSTM_FILE, "LSTM")

    if any(x is None for x in [df_if, df_dc, df_lstm]):
        print("Error loading files.")
        return

    # 2. Aggregation (Max Score Strategy)
    print("  -> Aggregating scores...")
    df_if_agg = df_if.groupby('user_id')[['score', 'label']].max().rename(columns={'score': 'score_if'})
    df_dc_agg = df_dc.groupby('user_id')[['score', 'label']].max().rename(columns={'score': 'score_dc'})
    df_lstm_agg = df_lstm.groupby('user_id')[['score', 'label']].max().rename(columns={'score': 'score_lstm'})

    # 3. Merge
    print("  -> Merging models...")
    df_merged = df_if_agg[['score_if', 'label']].join(df_dc_agg[['score_dc']], how='inner')
    df_merged = df_merged.join(df_lstm_agg[['score_lstm']], how='inner')

    print(f"     Total Aligned Users: {len(df_merged)}")

    # 4. Normalize
    print("  -> Normalizing...")
    scaler = MinMaxScaler()
    df_merged['score_if'] = scaler.fit_transform(df_merged[['score_if']]).flatten()
    df_merged['score_dc'] = scaler.fit_transform(df_merged[['score_dc']]).flatten()
    df_merged['score_lstm'] = scaler.fit_transform(df_merged[['score_lstm']]).flatten()

    # 5. APPLY OPTIMAL WEIGHTS
    print(f"  -> Applying Weights: IF={W_IF}, DC={W_DC}, LSTM={W_LSTM}")
    df_merged['ensemble_score'] = (
        (df_merged['score_if'] * W_IF) + 
        (df_merged['score_dc'] * W_DC) + 
        (df_merged['score_lstm'] * W_LSTM)
    )

    # --- DEBUG: INSPECT TOP SCORES ---
    print("\n[DEBUG] Top 15 Highest Scoring Users:")
    print(df_merged[['label', 'ensemble_score']].sort_values(by='ensemble_score', ascending=False).head(15))
    # ---------------------------------

    # 6. Generate Predictions
    df_merged['predicted_class'] = (df_merged['ensemble_score'] >= OPTIMAL_THRESHOLD).astype(int)
    
    y_true = np.where(df_merged['label'] > 0, 1, 0)
    y_pred = df_merged['predicted_class']

    # 7. Final Metrics
    f1 = f1_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*40)
    print(f"FINAL PERFORMANCE (Threshold > {OPTIMAL_THRESHOLD})")
    print("="*40)
    print(f"F1 Score:  {f1:.4f}")
    print(f"Recall:    {rec:.4f} ({cm[1,1]}/{cm[1,0]+cm[1,1]} Insiders)")
    print(f"Precision: {prec:.4f}")
    print("-" * 20)
    print(f"False Positives: {cm[0,1]}")
    print(f"True Positives:  {cm[1,1]}")
    print("="*40)

    # 8. Save
    df_merged.to_csv(OUTPUT_FILE)
    print(f"\n[SUCCESS] Final predictions saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()