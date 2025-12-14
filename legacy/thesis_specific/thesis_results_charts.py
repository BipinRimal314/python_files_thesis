# thesis_results_chart.py
import matplotlib.pyplot as plt
import numpy as np

models = ['Isolation\nForest', 'Deep\nClustering', 'LSTM\nAutoencoder']
auc_scores = [0.821, 0.741, 0.985]
rankings = [18, 26, 1.7]  # percentile

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# AUC-ROC comparison
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars1 = ax1.bar(models, auc_scores, color=colors, alpha=0.8, edgecolor='black')
ax1.axhline(y=0.9, color='green', linestyle='--', label='Excellent (>0.9)', alpha=0.5)
ax1.axhline(y=0.8, color='orange', linestyle='--', label='Good (>0.8)', alpha=0.5)
ax1.set_ylabel('AUC-ROC Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Discriminative Ability', fontsize=14, fontweight='bold')
ax1.set_ylim(0.6, 1.0)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add values on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# Insider ranking
bars2 = ax2.bar(models, rankings, color=colors, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Insider Ranking Percentile (%)', fontsize=12, fontweight='bold')
ax2.set_title('Operational Efficiency\n(Lower is Better)', fontsize=14, fontweight='bold')
ax2.invert_yaxis()  # Lower is better
ax2.grid(axis='y', alpha=0.3)

# Add values
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='top' if height < 10 else 'bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('thesis_results_comparison.png', dpi=300, bbox_inches='tight')
print("Chart saved: thesis_results_comparison.png")