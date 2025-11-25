import pandas as pd
import config_v2 as config
from pathlib import Path

def calculate_improvement():
    print("--- VERIFYING THESIS MATH ---")
    
    # 1. Load Baseline (V1) Result
    # Adjust filename if your V1 results are named differently
    v1_path = Path("results/isolation_forest_metrics.csv") 
    
    if v1_path.exists():
        df_v1 = pd.read_csv(v1_path)
        # Assuming column names based on standard metric outputs
        # We want the max F1 score from the V1 experiment
        v1_f1 = df_v1['f1_score'].max()
        print(f"V1 Baseline Max F1: {v1_f1:.6f}")
    else:
        # Fallback to the number in your thesis draft if file missing
        v1_f1 = 0.0105
        print(f"V1 File not found. Using Draft Number: {v1_f1}")

    # 2. Load Advanced (V2) Ensemble Result
    # Based on your file list, this is likely in 'optimal_evaluation_report_v2.csv'
    # or 'ensemble_predictions_r1_r2_r3.1_v2.csv'
    v2_path = Path("results/optimal_evaluation_report_v2.csv")
    
    if v2_path.exists():
        df_v2 = pd.read_csv(v2_path)
        # Look for the Ensemble row
        ensemble_row = df_v2[df_v2['model'].str.contains("Ensemble", case=False)]
        if not ensemble_row.empty:
            v2_f1 = ensemble_row['f1_score'].values[0]
            print(f"V2 Ensemble Max F1: {v2_f1:.6f}")
        else:
             # Fallback
            v2_f1 = 0.2222
            print(f"Ensemble not found in report. Using Draft Number: {v2_f1}")
    else:
        # Fallback
        v2_f1 = 0.2222
        print(f"V2 Report not found. Using Draft Number: {v2_f1}")

    # 3. Calculate Improvement
    # Formula: ((New - Old) / Old) * 100
    improvement_pct = ((v2_f1 - v1_f1) / v1_f1) * 100
    
    print("-" * 30)
    print(f"Calculated Improvement: {improvement_pct:,.2f}%")
    print("-" * 30)
    
    if abs(improvement_pct - 2116) < 10:
        print("Verdict: The 2,116% in draft is ACCURATE (based on exact floats).")
    else:
        print(f"Verdict: The draft is INACCURATE. Update text to: {improvement_pct:,.0f}%")

if __name__ == "__main__":
    calculate_improvement()