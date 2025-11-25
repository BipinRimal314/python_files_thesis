import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score
import config_v2 as config
from pathlib import Path

def check_ensemble():
    print("--- CHECKING ENSEMBLE METRICS ---")
    
    # Load the file you just generated
    file_path = Path("results/ensemble_predictions_r1_r2_r3.1_v2.csv")
    
    if not file_path.exists():
        print("Error: Ensemble file not found!")
        return

    df = pd.read_csv(file_path)
    y_true = df['true_label']
    y_score = df['anomaly_score']
    
    # Calculate Optimal F1
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_f1_idx = f1_scores.argmax()
    
    best_f1 = f1_scores[best_f1_idx]
    best_prec = precision[best_f1_idx]
    best_rec = recall[best_f1_idx]
    
    print(f"Ensemble F1-Score: {best_f1:.4f}")
    print(f"Ensemble Precision: {best_prec:.4f}")
    print(f"Ensemble Recall:    {best_rec:.4f}")
    
    # Verification against draft
    if abs(best_f1 - 0.2222) < 0.001:
        print("\nVERDICT: The Draft Number (0.2222) is CORRECT.")
    else:
        print(f"\nVERDICT: Update Draft to {best_f1:.4f}")

if __name__ == "__main__":
    check_ensemble()