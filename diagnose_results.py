"""
Diagnostic tool to investigate model predictions and data issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config

def diagnose_predictions():
    """Check what's happening with model predictions"""
    
    print("\n" + "="*80)
    print("DIAGNOSTIC REPORT: Model Predictions Analysis")
    print("="*80 + "\n")
    
    results_dir = config.RESULTS_DIR
    
    models = ['isolation_forest', 'lstm_autoencoder', 'deep_clustering']
    
    for model_name in models:
        pred_file = results_dir / f'{model_name}_predictions.csv'
        
        if not pred_file.exists():
            print(f"❌ {model_name}: No predictions file found")
            continue
        
        print(f"\n{'='*60}")
        print(f"📊 {model_name.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        
        df = pd.read_csv(pred_file)
        
        # Basic stats
        print(f"\nTotal samples: {len(df)}")
        
        # True labels distribution
        print(f"\n--- TRUE LABELS ---")
        true_label_counts = df['true_label'].value_counts()
        print(f"Normal (0):   {true_label_counts.get(0, 0):,} ({true_label_counts.get(0, 0)/len(df)*100:.2f}%)")
        print(f"Anomaly (1):  {true_label_counts.get(1, 0):,} ({true_label_counts.get(1, 0)/len(df)*100:.2f}%)")
        
        if true_label_counts.get(1, 0) == 0:
            print("⚠️  WARNING: No positive labels (anomalies) in the data!")
            print("   This is why AUC-ROC is NaN - need both classes for ROC curve")
        
        # Predictions distribution
        print(f"\n--- PREDICTIONS ---")
        pred_counts = df['prediction'].value_counts()
        print(f"Predicted Normal (0):   {pred_counts.get(0, 0):,} ({pred_counts.get(0, 0)/len(df)*100:.2f}%)")
        print(f"Predicted Anomaly (1):  {pred_counts.get(1, 0):,} ({pred_counts.get(1, 0)/len(df)*100:.2f}%)")
        
        if pred_counts.get(1, 0) == 0:
            print("⚠️  WARNING: Model didn't predict ANY anomalies!")
            print("   Threshold might be too high or model didn't learn properly")
        
        # Anomaly scores distribution
        print(f"\n--- ANOMALY SCORES ---")
        scores = df['anomaly_score'].values
        print(f"Min score:    {scores.min():.4f}")
        print(f"Max score:    {scores.max():.4f}")
        print(f"Mean score:   {scores.mean():.4f}")
        print(f"Median score: {np.median(scores):.4f}")
        print(f"Std dev:      {scores.std():.4f}")
        
        # Percentiles
        print(f"\nPercentiles:")
        for p in [90, 95, 99, 99.9]:
            print(f"  {p}th percentile: {np.percentile(scores, p):.4f}")
        
        # Check if scores are all the same
        if scores.std() < 0.01:
            print("⚠️  WARNING: All scores are nearly identical!")
            print("   Model may not have trained properly")
        
        # Confusion matrix if we have predictions
        if pred_counts.get(1, 0) > 0 and true_label_counts.get(1, 0) > 0:
            print(f"\n--- CONFUSION MATRIX ---")
            TP = ((df['prediction'] == 1) & (df['true_label'] == 1)).sum()
            FP = ((df['prediction'] == 1) & (df['true_label'] == 0)).sum()
            TN = ((df['prediction'] == 0) & (df['true_label'] == 0)).sum()
            FN = ((df['prediction'] == 0) & (df['true_label'] == 1)).sum()
            
            print(f"True Positives (TP):   {TP:,}")
            print(f"False Positives (FP):  {FP:,}")
            print(f"True Negatives (TN):   {TN:,}")
            print(f"False Negatives (FN):  {FN:,}")
    
    # Check preprocessed data
    print(f"\n{'='*80}")
    print("📁 CHECKING SOURCE DATA")
    print(f"{'='*80}\n")
    
    try:
        processed_file = config.PROCESSED_DATA_DIR / 'processed_unified_logs.csv'
        if processed_file.exists():
            df_source = pd.read_csv(processed_file, nrows=10000)  # Sample for speed
            
            if 'is_insider' in df_source.columns:
                insider_counts = df_source['is_insider'].value_counts()
                print(f"Source data (sampled 10k rows):")
                print(f"  Normal:   {insider_counts.get(0, 0):,}")
                print(f"  Insider:  {insider_counts.get(1, 0):,}")
                
                if insider_counts.get(1, 0) == 0:
                    print("\n❌ PROBLEM FOUND: Source data has NO insider labels!")
                    print("   Solutions:")
                    print("   1. Check if you provided an insider_list when preprocessing")
                    print("   2. Use CMU-CERT answers file to label insiders")
                    print("   3. Manually create synthetic labels for testing")
            else:
                print("⚠️  'is_insider' column not found in source data")
        else:
            print("Source data file not found")
    except Exception as e:
        print(f"Error reading source data: {e}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    # Check all prediction files
    all_have_labels = True
    all_have_predictions = True
    
    for model_name in models:
        pred_file = results_dir / f'{model_name}_predictions.csv'
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            if df['true_label'].sum() == 0:
                all_have_labels = False
            if df['prediction'].sum() == 0:
                all_have_predictions = False
    
    if not all_have_labels:
        print("🔴 CRITICAL: No ground truth labels (all true_label = 0)")
        print("\n   To fix this:")
        print("   1. Create a file 'insider_users.txt' with known malicious user IDs")
        print("   2. Run: python data_preprocessing.py --insider-list insider_users.txt")
        print("   3. Or add synthetic labels for testing:")
        print("      python create_synthetic_labels.py")
    
    if not all_have_predictions:
        print("\n🟡 WARNING: Models not predicting anomalies (all prediction = 0)")
        print("\n   To fix this:")
        print("   1. Lower the anomaly threshold in config.py:")
        print("      ENSEMBLE['final_threshold'] = 0.5  # From 0.7")
        print("   2. Check if models trained properly (look for errors in logs)")
        print("   3. Increase contamination for Isolation Forest:")
        print("      ISOLATION_FOREST['contamination'] = 0.05  # From 0.01")
    
    if all_have_labels and all_have_predictions:
        print("✅ Data looks good! Models are making predictions with ground truth.")
        print("   If metrics are still low, consider tuning hyperparameters.")
    
    print("\n" + "="*80 + "\n")


def create_synthetic_labels():
    """Create synthetic insider labels for testing purposes"""
    print("\n" + "="*80)
    print("CREATING SYNTHETIC INSIDER LABELS")
    print("="*80 + "\n")
    
    try:
        # Load processed data
        processed_file = config.PROCESSED_DATA_DIR / 'processed_unified_logs.csv'
        
        if not processed_file.exists():
            print("❌ Processed data not found. Run data_preprocessing.py first.")
            return
        
        print("Loading processed data...")
        df = pd.read_csv(processed_file)
        
        print(f"Total records: {len(df):,}")
        
        # Identify user column
        user_col = 'user_pseudo' if 'user_pseudo' in df.columns else 'user'
        
        if user_col not in df.columns:
            print(f"❌ User column not found")
            return
        
        # Get unique users
        users = df[user_col].unique()
        print(f"Total users: {len(users):,}")
        
        # Mark top 1% most active users as potential insiders (just for testing)
        user_activity = df[user_col].value_counts()
        n_insiders = max(1, int(len(users) * 0.01))  # 1% of users
        insider_users = user_activity.head(n_insiders).index.tolist()
        
        print(f"Marking {n_insiders} users as insiders (top 1% most active)")
        
        # Add labels
        if 'is_insider' not in df.columns:
            df['is_insider'] = 0
        
        df.loc[df[user_col].isin(insider_users), 'is_insider'] = 1
        
        # Save
        df.to_csv(processed_file, index=False)
        
        print(f"\n✅ Labels added:")
        print(f"   Normal:   {(df['is_insider'] == 0).sum():,}")
        print(f"   Insider:  {(df['is_insider'] == 1).sum():,}")
        print(f"\nSaved to: {processed_file}")
        print("\nNow re-run feature engineering and model training:")
        print("  python feature_engineering.py")
        print("  python main.py --train")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--create-labels':
        create_synthetic_labels()
    else:
        diagnose_predictions()