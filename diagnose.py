# check_labels.py
import pandas as pd
import config

# Check processed data
df = pd.read_csv(config.PROCESSED_DATA_DIR / 'processed_unified_logs.csv', nrows=100000)

print("Label distribution in source data:")
print(df['is_insider'].value_counts())
print(f"\nInsider percentage: {df['is_insider'].mean()*100:.4f}%")

if 'user_pseudo' in df.columns:
    insider_users = df[df['is_insider']==1]['user_pseudo'].unique()
    print(f"\nUnique insider users found: {len(insider_users)}")
    print("Sample:", list(insider_users)[:5])