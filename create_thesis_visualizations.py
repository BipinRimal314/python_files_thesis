import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = 'thesis_plots'
DPI = 300  # High resolution for print
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# FIGURE 1: ENSEMBLE WEIGHTS (The "Breakthrough")
# ==========================================
def plot_ensemble_weights():
    print("Generating Figure 1: Ensemble Weights...")
    
    models = ['Isolation Forest', 'Deep Clustering', 'LSTM Autoencoder']
    weights = [0.1, 0.1, 0.8]
    colors = ['#95a5a6', '#95a5a6', '#e74c3c']  # Grey for static, Red for Sequential

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, weights, color=colors, edgecolor='black', linewidth=1)
    
    # Add labels on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.title('Optimal Ensemble Weights (Grid Search Results)', fontsize=16, pad=20)
    plt.ylabel('Weight Contribution (0.0 - 1.0)', fontsize=12)
    plt.ylim(0, 1.0)
    
    # Add explanatory text inside the plot
    plt.text(2, 0.5, "Dominant Signal\n(Sequential)", ha='center', color='#c0392b', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_X_Ensemble_Weights.png'), dpi=DPI)
    plt.close()

# ==========================================
# FIGURE 2: THE FORENSIC HEATMAP (Who caught whom?)
# ==========================================
def plot_forensic_heatmap():
    print("Generating Figure 2: Forensic Detection Matrix...")
    
    # Data from your findings
    data = np.array([
        [1.0, 1.0, 1.0],  # User 616
        [1.0, 0.0, 1.0],  # User 724
        [0.0, 0.0, 0.0]   # User 3908
    ])
    
    insiders = ['User 616\n(Exfiltration)', 'User 724\n(Sabotage)', 'User 3908\n(Stealth/LotL)']
    models = ['Isolation\nForest', 'Deep\nClustering', 'LSTM\n(Sequential)']

    plt.figure(figsize=(8, 6))
    
    # Create Heatmap
    # cmap: Red (Missed) to Green (Detected)
    ax = sns.heatmap(data, annot=True, cmap='RdYlGn', vmin=0, vmax=1, 
                     linewidths=1, linecolor='black', cbar=False,
                     xticklabels=models, yticklabels=insiders,
                     annot_kws={"size": 14, "weight": "bold"})

    plt.title('Forensic Analysis of Insider Detection', fontsize=16, pad=20)
    
    # Highlight the Stealth Gap
    plt.axhline(y=2, color='blue', linewidth=3)
    plt.text(1.5, 2.5, "THE STEALTH GAP (Ontological Ceiling)", 
             ha='center', va='center', color='white', fontweight='bold', 
             bbox=dict(facecolor='black', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_Y_Forensic_Heatmap.png'), dpi=DPI)
    plt.close()

# ==========================================
# FIGURE 3: CONFUSION MATRIX (The Operational Reality)
# ==========================================
def plot_confusion_matrix():
    print("Generating Figure 3: Confusion Matrix...")
    
    # Based on F1 = 0.05, Recall = 0.66 (2/3)
    # This implies Precision approx 0.026 -> ~75 False Positives
    tp = 2
    fn = 1
    fp = 74
    tn = 4921 # Total 4998 - 3 insiders - 74 fp
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    labels = ['Benign', 'Malicious']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 16})
    
    plt.title('Operational Confusion Matrix (Aligned Test Set)', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add metrics text box
    stats_text = (
        f"Recall: {(tp/(tp+fn)):.2%}\n"
        f"Precision: {(tp/(tp+fp)):.2%}\n"
        f"False Positives: {fp}\n"
        f"Alert Ratio: 1:{int(fp/tp) if tp>0 else 0}"
    )
    plt.text(1.5, 0.5, stats_text, fontsize=12, 
             bbox=dict(boxstyle="round", alpha=0.1, color='black'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_Z_Confusion_Matrix.png'), dpi=DPI)
    plt.close()

# ==========================================
# FIGURE 4: THE "ONTOLOGICAL CEILING" (Comparison)
# ==========================================
def plot_stealth_gap():
    print("Generating Figure 4: The Stealth Gap...")
    
    # Scores
    insider_scores = [1.0, 1.0, 0.0]
    labels = ['User 616', 'User 724', 'User 3908']
    colors = ['#2ecc71', '#2ecc71', '#e74c3c'] # Green detected, Red missed
    
    # Add average benign score for context
    avg_benign = 0.06
    
    plt.figure(figsize=(10, 6))
    
    # Plot baseline
    plt.axhline(y=avg_benign, color='grey', linestyle='--', label='Avg Benign Score (Noise Floor)')
    
    # Plot Insiders
    bars = plt.bar(labels, insider_scores, color=colors, alpha=0.8, edgecolor='black')
    
    plt.title('The "Ontological Ceiling": Visible vs. Invisible Threats', fontsize=16, pad=20)
    plt.ylabel('Ensemble Anomaly Score', fontsize=12)
    plt.ylim(0, 1.1)
    
    plt.legend()
    
    # Annotations
    plt.text(0, 0.5, "DETECTED\n(Score >> Noise)", ha='center', color='white', fontweight='bold')
    plt.text(1, 0.5, "DETECTED\n(Score >> Noise)", ha='center', color='white', fontweight='bold')
    plt.text(2, 0.1, "UNDETECTED\n(Indistinguishable from Noise)", ha='center', color='black')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_W_Stealth_Gap.png'), dpi=DPI)
    plt.close()

def main():
    plot_ensemble_weights()
    plot_forensic_heatmap()
    plot_confusion_matrix()
    plot_stealth_gap()
    print(f"\nAll plots saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()