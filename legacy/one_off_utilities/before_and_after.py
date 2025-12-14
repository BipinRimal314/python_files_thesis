import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# --- FILE PATH ---
FILE_PATH = os.path.join('data', 'processed', 'engineered_static_features_r1_r2_r3.1_v2.csv')

def find_best_feature_and_plot():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    print(f"Loading data from {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)
    
    # 1. ROBUST LABELING (Fixing the 0 vs 2,3,4 issue)
    # Check for label column
    label_col = 'is_insider' if 'is_insider' in df.columns else 'label'
    if label_col not in df.columns:
        print(f"Error: Could not find label column. Available: {df.columns[:5]}...")
        return
        
    # Create readable labels: Any ID > 0 is an Insider
    df['Label'] = df[label_col].apply(lambda x: 'Insider' if x > 0 else 'Benign')
    
    # Check if we have insiders
    if len(df[df['Label'] == 'Insider']) == 0:
        print("CRITICAL: No insiders found even with x > 0 logic!")
        return

    # 2. AUTO-DISCOVER THE BEST FEATURE
    # We want the feature where Insiders have the highest average Z-Score
    print("Scanning for the most anomalous feature...")
    
    # Filter for Z-Score columns only
    zscore_cols = [c for c in df.columns if 'zscore_self' in c]
    
    # Calculate mean Z-score for Insiders per column
    insider_data = df[df['Label'] == 'Insider'][zscore_cols]
    # We use absolute value because a massive negative deviation is also an anomaly
    best_v2_feature = insider_data.abs().mean().idxmax()
    
    print(f"-> Best Discriminator Found: {best_v2_feature}")
    
    # 3. ATTEMPT TO MAP BACK TO V1 (RAW VOLUME)
    # Try to guess the raw feature name by stripping the zscore suffix
    # Example: 'email_size_kb_mean_zscore_self_max' -> 'email_size_kb_sum_sum'
    # This is heuristic; if it fails, we pick a generic activity count
    base_name = best_v2_feature.split('_zscore')[0]
    
    # Look for a 'sum' or 'count' feature that matches the base name
    v1_candidates = [c for c in df.columns if base_name in c and ('sum' in c or 'count' in c) and 'zscore' not in c]
    
    if v1_candidates:
        best_v1_feature = v1_candidates[0] # Pick the first match
    else:
        print("Could not map V2 feature to V1 raw count. Using generic activity count.")
        best_v1_feature = 'activity_count' if 'activity_count' in df.columns else df.columns[0]

    print(f"-> Corresponding V1 Feature: {best_v1_feature}")

    # 4. PLOTTING
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Define Palette
    palette = {'Benign': '#95a5a6', 'Insider': '#e74c3c'}

    # Plot 1: V1 (Raw Volume)
    # We set alpha=0.5 so we can see if the red is hiding under the grey
    sns.kdeplot(data=df, x=best_v1_feature, hue='Label', fill=True, 
                common_norm=False, palette=palette, ax=axes[0], 
                log_scale=True, alpha=0.5, linewidth=1.5)
                
    axes[0].set_title(f'V1 Baseline: Raw Volume\n({best_v1_feature})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Volume (Log Scale)')
    
    # Plot 2: V2 (Z-Scores)
    sns.kdeplot(data=df, x=best_v2_feature, hue='Label', fill=True, 
                common_norm=False, palette=palette, ax=axes[1], 
                alpha=0.5, linewidth=1.5)

    axes[1].set_title(f'V2 Advanced: Contextual Entropy\n({best_v2_feature})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Deviation from Self (Z-Score)')
    
    # Add Threshold Line
    axes[1].axvline(x=3, color='black', linestyle='--', alpha=0.7)
    axes[1].text(3.1, axes[1].get_ylim()[1]*0.7, 'Anomaly Threshold (3σ)', rotation=90)

    # Force X-axis to show the separation if outliers are extreme
    # axes[1].set_xlim(left=-5, right=10) # Optional: Enable if insiders are TOO far right

    plt.suptitle('Resolution of the Accuracy Paradox: From Volume to Entropy', fontsize=18)
    plt.tight_layout()
    
    # Save
    save_path = os.path.join('results', 'v2', 'visualizations', 'thesis_contextual_entropy_proof_SMART.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    find_best_feature_and_plot()