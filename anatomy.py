import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.dates as mdates

# --- CONFIGURATION ---
SUMMARY_FILE = os.path.join('results', 'ensemble_predictions_r1_r2_r3.1_v2.csv')
DAILY_FILE = os.path.join('data', 'processed', 'engineered_daily_features_r1_r2_r3.1_v2.csv')

def plot_behavior_timeline():
    # 1. LOAD SUMMARY TO FIND THE INSIDER
    if not os.path.exists(SUMMARY_FILE):
        print(f"CRITICAL: Could not find summary file at {SUMMARY_FILE}")
        return
        
    print(f"1. Loading summary from {SUMMARY_FILE}...")
    df_summary = pd.read_csv(SUMMARY_FILE)
    
    # Identify the insider with the highest anomaly score
    # Filter for true_label = 1 (Insider)
    insiders = df_summary[df_summary['true_label'] == 1]
    
    if len(insiders) == 0:
        print("No labeled insiders found. Picking top scoring user.")
        target_user = df_summary.sort_values('anomaly_score', ascending=False).iloc[0]['user_id']
    else:
        # Pick the 'worst' insider (highest score)
        target_user = insiders.sort_values('anomaly_score', ascending=False).iloc[0]['user_id']
        
    print(f"-> Target Insider Identified: {target_user}")

    # 2. LOAD DAILY DATA FOR TIMELINE
    if not os.path.exists(DAILY_FILE):
        print(f"CRITICAL: Could not find daily features at {DAILY_FILE}")
        return

    print(f"2. Loading daily timeline from {DAILY_FILE}...")
    df_daily = pd.read_csv(DAILY_FILE)
    
    # Filter for our target user
    user_timeline = df_daily[df_daily['user_id'] == target_user].copy()
    
    if user_timeline.empty:
        print(f"Error: User {target_user} not found in daily features file.")
        return

    # 3. HANDLE DATES
    if 'date' in user_timeline.columns:
        user_timeline['date'] = pd.to_datetime(user_timeline['date'])
        time_col = 'date'
    elif 'day' in user_timeline.columns:
        # Create synthetic dates if only 'day' index exists
        print("Generating dates from 'day' index...")
        user_timeline['date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(user_timeline['day'], unit='D')
        time_col = 'date'
    else:
        print("Error: No date/day column in daily features.")
        return
        
    user_timeline = user_timeline.sort_values(time_col)

    # 4. SELECT FEATURE TO PLOT
    # We want to plot the Z-Score if available, as that drives the anomaly score
    zscore_cols = [c for c in user_timeline.columns if 'zscore' in c and 'self' in c]
    
    if zscore_cols:
        # Plot the feature with the highest variance/max value (most anomalous)
        best_col = user_timeline[zscore_cols].max().idxmax()
        print(f"-> Visualizing Feature: {best_col}")
        y_data = user_timeline[best_col]
        y_label = 'Contextual Z-Score (Deviation)'
        title_suffix = 'Behavioral Deviation (Z-Score)'
    else:
        # Fallback to a volume column
        print("-> No Z-Score columns found. Plotting 'count' or 'sum'.")
        vol_cols = [c for c in user_timeline.columns if 'sum' in c or 'count' in c]
        best_col = vol_cols[0] if vol_cols else user_timeline.columns[2]
        y_data = user_timeline[best_col]
        y_label = 'Daily Volume'
        title_suffix = 'Daily Activity Volume'

    # 5. PLOTTING
    plt.figure(figsize=(14, 7))
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.plot(user_timeline[time_col], y_data, 
             color='#2c3e50', linewidth=2, label=best_col)

    # Highlight Spikes (Dynamic Threshold)
    # Highlight top 10% of activity as "Anomalous"
    threshold = y_data.quantile(0.90)
    plt.fill_between(user_timeline[time_col], y_data, 
                     where=(y_data > threshold), 
                     interpolate=True, color='#e74c3c', alpha=0.3, label='Abnormal Behavior')

    plt.title(f'Anatomy of an Attack: User {target_user} - {title_suffix}', fontsize=16, fontweight='bold')
    plt.ylabel(y_label)
    plt.xlabel('Timeline')
    plt.legend(loc='upper left')

    # Format Dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    
    # Save
    save_dir = os.path.join('results', 'v2', 'visualizations')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'thesis_anatomy_{target_user}.png')
    
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_behavior_timeline()