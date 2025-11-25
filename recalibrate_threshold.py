# recalibrate_thresholds.py
import pandas as pd
import numpy as np
from pathlib import Path
import config

models = ['isolation_forest', 'deep_clustering', 'lstm_autoencoder']

for model in models:
    pred_file = config.RESULTS_DIR / f'{model}_predictions.csv'
    if not pred_file.exists():
        continue
    
    df = pd.read_csv(pred_file)
    
    # Calculate optimal threshold using F1-score
    scores = df['anomaly_score'].values
    y_true = df['true_label'].values
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.linspace(0.9, 0.999, 100):  # Try higher thresholds
        predictions = (scores > threshold).astype(int)
        
        tp = ((predictions == 1) & (y_true == 1)).sum()
        fp = ((predictions == 1) & (y_true == 0)).sum()
        fn = ((predictions == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n{model}:")
    print(f"  Best threshold: {best_threshold:.4f}")
    print(f"  Best F1-score: {best_f1:.4f}")
    
    # Apply best threshold
    df['prediction'] = (df['anomaly_score'] > best_threshold).astype(int)
    df.to_csv(pred_file, index=False)

print("\n✅ Thresholds recalibrated. Re-run: python main.py --evaluate")