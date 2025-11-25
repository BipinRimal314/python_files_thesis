# check_data_distribution.py
import pandas as pd
import config

print("="*80)
print("DATA DISTRIBUTION CHECK")
print("="*80)

# Check test predictions
models = ['isolation_forest', 'deep_clustering', 'lstm_autoencoder']

for model in models:
    pred_file = config.RESULTS_DIR / f"{model}_predictions.csv"
    df = pd.read_csv(pred_file)
    
    total = len(df)
    insiders = df['true_label'].sum()
    normal = total - insiders
    
    print(f"\n{model.upper().replace('_', ' ')}:")
    print(f"  Total samples:   {total:,}")
    print(f"  Insider samples: {insiders} ({insiders/total*100:.4f}%)")
    print(f"  Normal samples:  {normal:,} ({normal/total*100:.2f}%)")
    print(f"  Imbalance ratio: 1:{normal/max(insiders,1):.0f}")

