import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
FILE_PATH = os.path.join('data', 'processed', 'engineered_static_features_r1_r2_r3.1_v2.csv')

def generate_3d_separation():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    print(f"Loading data from {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)

    # 1. ROBUST LABELING (The Fix)
    # Check for label column
    label_col = 'is_insider' if 'is_insider' in df.columns else 'label'
    if label_col not in df.columns:
        print("Error: Label column not found.")
        return

    # Create string labels for plotting
    # CRITICAL FIX: Any ID > 0 is an Insider
    df['Label_Str'] = df[label_col].apply(lambda x: 'Insider' if x > 0 else 'Normal')
    
    insider_count = len(df[df['Label_Str'] == 'Insider'])
    print(f"Insiders identified for plotting: {insider_count}")

    # 2. PREPARE FEATURES FOR PCA
    # Drop non-feature columns
    drop_cols = ['user_id', 'date', 'day', 'label', 'is_insider', 'Label_Str', 'role', 'dataset']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Use only numeric columns
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    
    # Standardize (Critical for PCA)
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. DIMENSIONALITY REDUCTION (PCA)
    print("Reducing to 3 Dimensions...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # Add PCA results back to dataframe
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    df['PCA3'] = X_pca[:, 2]

    # 4. PLOTTING (Layered for Visibility)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Split Data
    normal_df = df[df['Label_Str'] == 'Normal']
    insider_df = df[df['Label_Str'] == 'Insider']

    # Plot Normal (Blue, Transparent, Background)
    # We downsample normal to 5000 points to avoid overcrowding if dataset is huge
    if len(normal_df) > 5000:
        normal_df = normal_df.sample(5000, random_state=42)
        
    ax.scatter(normal_df['PCA1'], normal_df['PCA2'], normal_df['PCA3'], 
               c='#4a90e2', alpha=0.1, s=10, label='Normal Behavior')

    # Plot Insider (Red, Opaque, On Top, Larger)
    ax.scatter(insider_df['PCA1'], insider_df['PCA2'], insider_df['PCA3'], 
               c='#e74c3c', alpha=1.0, s=50, edgecolors='black', linewidth=0.5, 
               label='Contextual Anomaly (Insider)', zorder=10)

    # Formatting
    ax.set_title("Figure 8: The 'Weird-but-Normal' Separation\n(Contextual Z-Scores Isolate Anomalies)", fontsize=14, fontweight='bold')
    ax.set_xlabel('Context Dimension 1 (PCA)')
    ax.set_ylabel('Context Dimension 2 (PCA)')
    ax.set_zlabel('Context Dimension 3 (PCA)')
    ax.legend()

    # Save
    save_path = os.path.join('results', 'v2', 'visualizations', 'thesis_3d_separation_FIXED.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    generate_3d_separation()