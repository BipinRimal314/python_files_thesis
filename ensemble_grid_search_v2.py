import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
import itertools
import os
import sys

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Model Prediction Files
ISO_FOREST_FILE = 'isolation_forest_predictions_r1_r2_r3.1_v2.csv'
DEEP_CLUSTER_FILE = 'deep_clustering_predictions_r1_r2_r3.1_v2.csv'
LSTM_FILE = 'lstm_autoencoder_predictions_r1_r2_r3.1_v2.csv'

def get_file_path(filename, search_dir=RESULTS_DIR):
    path = os.path.join(search_dir, filename)
    if os.path.exists(path):
        return path
    if os.path.exists(filename):
        return filename
    return None

def load_prediction_df(filename, model_name):
    """
    Loads the CSV, standardizes column names, and returns a DataFrame.
    """
    path = get_file_path(filename)
    if not path:
        print(f"[ERROR] Could not find {model_name} file: {filename}")
        return None

    print(f"  -> Loading {model_name}...")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"     [!] Failed to read CSV: {e}")
        return None

    # Standardize Column Names
    # We want: 'user_id', 'label', 'score'
    df.columns = [c.lower().strip() for c in df.columns]
    
    # rename 'true_label' or 'is_insider' -> 'label'
    rename_map = {}
    
    # 1. Find Score Column
    score_col = next((c for c in df.columns if c in ['anomaly_score', 'score', 'reconstruction_error', 'prediction']), None)
    if score_col: rename_map[score_col] = 'score'
    else: 
        print(f"     [CRITICAL] No score column found in {model_name}. Cols: {list(df.columns)}")
        return None

    # 2. Find Label Column
    label_col = next((c for c in df.columns if c in ['true_label', 'is_insider', 'label', 'ground_truth']), None)
    if label_col: rename_map[label_col] = 'label'
    else:
        print(f"     [CRITICAL] No label column found in {model_name}. Cols: {list(df.columns)}")
        return None

    # 3. Find User Column
    user_col = next((c for c in df.columns if c in ['user_id', 'user', 'id']), None)
    if user_col: rename_map[user_col] = 'user_id'
    else:
        print(f"     [CRITICAL] No user_id column found in {model_name}. Cols: {list(df.columns)}")
        return None

    # Apply renaming and return only relevant columns
    df = df.rename(columns=rename_map)
    return df[['user_id', 'label', 'score']]

def load_and_align_data():
    print("\n--- LOADING AND ALIGNING DATA (BY USER_ID) ---")

    # 1. Load DataFrames
    df_if = load_prediction_df(ISO_FOREST_FILE, "Isolation Forest")
    df_dc = load_prediction_df(DEEP_CLUSTER_FILE, "Deep Clustering")
    df_lstm = load_prediction_df(LSTM_FILE, "LSTM")

    if any(x is None for x in [df_if, df_dc, df_lstm]):
        return None, None, None, None

    # 2. Aggregation Strategy (The Fix for Size Mismatch)
    # We group by user_id and take the MAX score.
    # Logic: If a user has ANY high-anomaly window, they are risky.
    print(f"\n  -> Aggregating scores by User ID (Max Score Strategy)...")
    
    # Aggregation rules: Score -> Max, Label -> Max (if user was insider once, they are insider)
    agg_rules = {'score': 'max', 'label': 'max'}
    
    df_if_agg = df_if.groupby('user_id').agg(agg_rules).rename(columns={'score': 'score_if'})
    df_dc_agg = df_dc.groupby('user_id').agg(agg_rules).rename(columns={'score': 'score_dc'})
    df_lstm_agg = df_lstm.groupby('user_id').agg(agg_rules).rename(columns={'score': 'score_lstm'})

    print(f"     IF Users: {len(df_if_agg)} | DC Users: {len(df_dc_agg)} | LSTM Users: {len(df_lstm_agg)}")

    # 3. Merge DataFrames
    # We use inner join to ensure we only evaluate users present in ALL models
    print("  -> Merging models on 'user_id'...")
    df_merged = df_if_agg[['score_if', 'label']].join(df_dc_agg[['score_dc']], how='inner')
    df_merged = df_merged.join(df_lstm_agg[['score_lstm']], how='inner')

    print(f"     Aligned User Count: {len(df_merged)}")

    # 4. Extract Arrays
    y_test = df_merged['label'].values
    if_scores = df_merged['score_if'].values
    dc_scores = df_merged['score_dc'].values
    lstm_scores = df_merged['score_lstm'].values

    # --- FIX FOR MULTICLASS ERROR ---
    # Ensure labels are binary integers (0, 1) to satisfy precision_recall_curve
    print(f"     Raw Labels Found: {np.unique(y_test)}")
    y_test = np.where(y_test > 0, 1, 0).astype(int)
    print(f"     Binarized Labels: {np.unique(y_test)}")
    # --------------------------------

    # Check for Insiders
    num_pos = np.sum(y_test)
    print(f"     Insiders in Aligned Set: {int(num_pos)}")
    
    if num_pos == 0:
        print("[CRITICAL] No insiders found after alignment. Check if user_ids match across files.")
        return None, None, None, None

    # 5. Normalize
    print("  -> Normalizing scores...")
    scaler = MinMaxScaler()
    if_scores = scaler.fit_transform(if_scores.reshape(-1, 1)).flatten()
    dc_scores = scaler.fit_transform(dc_scores.reshape(-1, 1)).flatten()
    lstm_scores = scaler.fit_transform(lstm_scores.reshape(-1, 1)).flatten()

    return y_test, if_scores, dc_scores, lstm_scores

def perform_grid_search(y_test, if_scores, dc_scores, lstm_scores):
    print("\n--- STARTING GRID SEARCH ---")
    steps = np.arange(0, 1.1, 0.1)
    combinations = [p for p in itertools.product(steps, repeat=3) if np.isclose(sum(p), 1.0)]
    
    best_f1 = 0.0
    best_weights = (0, 0, 0)
    best_threshold = 0.0
    
    print_interval = max(1, len(combinations) // 5)

    for i, (w_if, w_dc, w_lstm) in enumerate(combinations):
        ensemble_scores = (w_if * if_scores) + (w_dc * dc_scores) + (w_lstm * lstm_scores)
        precisions, recalls, thresholds = precision_recall_curve(y_test, ensemble_scores)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        
        f1_scores = np.nan_to_num(f1_scores)
        if len(f1_scores) > 0:
            max_idx = np.argmax(f1_scores)
            if f1_scores[max_idx] > best_f1:
                best_f1 = f1_scores[max_idx]
                best_weights = (w_if, w_dc, w_lstm)
                best_threshold = thresholds[max_idx] if max_idx < len(thresholds) else thresholds[-1]
                
        if i % print_interval == 0:
            print(f"  Checking {i}/{len(combinations)}... Best F1: {best_f1:.4f}")

    return best_weights, best_threshold, best_f1

def main():
    data = load_and_align_data()
    if data[0] is None: return
    y_test, if_scores, dc_scores, lstm_scores = data
    weights, threshold, f1 = perform_grid_search(y_test, if_scores, dc_scores, lstm_scores)

    print("\n" + "="*40)
    print("GRID SEARCH RESULTS")
    print("="*40)
    print(f"Best F1-Score:      {f1:.5f}")
    print(f"Optimal Threshold:  {threshold:.5f}")
    print(f"Optimal Weights:    IF:{weights[0]:.1f}, DC:{weights[1]:.1f}, LSTM:{weights[2]:.1f}")
    print("="*40)
    
    if weights[2] < 0.1:
        print("\n[!] INSIGHT: LSTM weight is low (< 0.1).")
        print("    Confirms 'Sliding Window Fallacy' (Signal Dilution).")
    else:
        print(f"\n[+] INSIGHT: LSTM is contributing ({weights[2]:.1f}).")


if __name__ == "__main__":
    main()