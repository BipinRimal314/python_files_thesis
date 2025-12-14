"""
Proper evaluation metrics for highly imbalanced insider threat detection.

With 1 insider in 1000 samples, traditional metrics are meaningless.
We use metrics that are practical for SOC operations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import config
import utils

logger = utils.logger

class ImbalancedEvaluator:
    """
    Evaluator designed for extreme class imbalance (0.01-0.1% positive class)
    """
    
    def __init__(self):
        self.models = ['isolation_forest', 'deep_clustering', 'lstm_autoencoder']
        self.results = {}
        
    def load_predictions(self, model_name: str):
        """Load predictions for a model"""
        pred_file = config.RESULTS_DIR / f"{model_name}_predictions.csv"
        df = pd.read_csv(pred_file)
        logger.info(f"Loaded {model_name}: {len(df)} samples, {df['true_label'].sum()} insiders")
        return df
    
    def calculate_metrics_at_k(self, y_true, y_scores, k_values=[10, 50, 100, 500]):
        """
        Precision at K: Of the top K highest-scoring samples, how many are insiders?
        This is what SOC analysts care about - they investigate top alerts first.
        """
        metrics = {}
        
        # Sort by scores (highest first)
        sorted_indices = np.argsort(y_scores)[::-1]
        
        for k in k_values:
            if k > len(y_true):
                continue
                
            # Get top K predictions
            top_k_labels = y_true[sorted_indices[:k]]
            
            # How many insiders in top K?
            precision_at_k = top_k_labels.sum() / k
            
            # Did we catch any insiders? (recall)
            total_insiders = y_true.sum()
            if total_insiders > 0:
                recall_at_k = top_k_labels.sum() / total_insiders
            else:
                recall_at_k = 0
            
            metrics[f'precision_at_{k}'] = precision_at_k
            metrics[f'recall_at_{k}'] = recall_at_k
        
        return metrics
    
    def calculate_recall_at_fpr(self, y_true, y_scores, fpr_thresholds=[0.01, 0.05, 0.10]):
        """
        Recall at fixed False Positive Rate.
        "If we accept X% false positives, what % of insiders do we catch?"
        """
        metrics = {}
        
        if y_true.sum() == 0:
            return metrics
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        for target_fpr in fpr_thresholds:
            # Find the recall (TPR) at this FPR
            idx = np.where(fpr <= target_fpr)[0]
            if len(idx) > 0:
                recall = tpr[idx[-1]]  # Last point where FPR <= target
            else:
                recall = 0
            
            metrics[f'recall_at_{int(target_fpr*100)}pct_fpr'] = recall
        
        return metrics
    
    def evaluate_model(self, model_name: str):
        """Comprehensive evaluation for one model"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating: {model_name.replace('_', ' ').title()}")
        logger.info(f"{'='*80}")
        
        df = self.load_predictions(model_name)
        y_true = df['true_label'].values
        y_scores = df['anomaly_score'].values
        
        total_samples = len(y_true)
        n_insiders = y_true.sum()
        
        logger.info(f"\nDataset: {total_samples:,} samples, {n_insiders} insiders ({n_insiders/total_samples*100:.4f}%)")
        
        if n_insiders == 0:
            logger.warning("No insiders in test set - cannot evaluate!")
            return None
        
        results = {
            'model': model_name.replace('_', ' ').title(),
            'n_samples': total_samples,
            'n_insiders': n_insiders,
            'imbalance_ratio': f"1:{int(total_samples/n_insiders)}"
        }
        
        # AUC-ROC (works well for imbalanced data)
        try:
            auc = roc_auc_score(y_true, y_scores)
            results['auc_roc'] = auc
            logger.info(f"AUC-ROC: {auc:.4f}")
        except:
            results['auc_roc'] = np.nan
            logger.warning("Could not calculate AUC-ROC")
        
        # Precision at K
        logger.info("\nPrecision at K (Top K alerts):")
        k_metrics = self.calculate_metrics_at_k(y_true, y_scores)
        for key, value in k_metrics.items():
            results[key] = value
            if 'precision' in key:
                k = key.split('_')[-1]
                logger.info(f"  Top {k}: {value*100:.2f}% of alerts are real threats")
        
        # Recall at FPR
        logger.info("\nRecall at Fixed False Positive Rate:")
        fpr_metrics = self.calculate_recall_at_fpr(y_true, y_scores)
        for key, value in fpr_metrics.items():
            results[key] = value
            fpr_pct = key.split('_')[2].replace('pct', '')
            logger.info(f"  At {fpr_pct}% FPR: Catch {value*100:.1f}% of insiders")
        
        # Rank of insiders (where do they appear in sorted list?)
        logger.info("\nInsider Rankings (lower is better):")
        sorted_indices = np.argsort(y_scores)[::-1]
        insider_indices = np.where(y_true == 1)[0]
        
        insider_ranks = []
        for insider_idx in insider_indices:
            rank = np.where(sorted_indices == insider_idx)[0][0] + 1  # 1-indexed
            insider_ranks.append(rank)
            logger.info(f"  Insider #{len(insider_ranks)} ranked: #{rank} / {total_samples} (top {rank/total_samples*100:.2f}%)")
        
        results['best_insider_rank'] = min(insider_ranks)
        results['worst_insider_rank'] = max(insider_ranks)
        results['avg_insider_rank'] = np.mean(insider_ranks)
        
        return results
    
    def create_comparison_table(self, all_results):
        """Create comparison table focusing on practical metrics"""
        df = pd.DataFrame(all_results)
        
        # Select most important columns
        important_cols = [
            'model', 'n_insiders', 'imbalance_ratio', 'auc_roc',
            'precision_at_50', 'precision_at_100',
            'recall_at_5pct_fpr', 'recall_at_10pct_fpr',
            'best_insider_rank', 'avg_insider_rank'
        ]
        
        display_cols = [col for col in important_cols if col in df.columns]
        df_display = df[display_cols].copy()
        
        return df_display
    
    def plot_ranking_analysis(self, all_results):
        """Plot where insiders appear in ranked lists"""
        fig, axes = plt.subplots(1, len(all_results), figsize=(15, 5))
        
        if len(all_results) == 1:
            axes = [axes]
        
        for idx, (model_name, ax) in enumerate(zip([r['model'] for r in all_results], axes)):
            df = self.load_predictions(model_name.lower().replace(' ', '_'))
            y_true = df['true_label'].values
            y_scores = df['anomaly_score'].values
            
            # Sort by score
            sorted_indices = np.argsort(y_scores)[::-1]
            insider_indices = np.where(y_true == 1)[0]
            
            # Find ranks
            ranks = []
            for insider_idx in insider_indices:
                rank = np.where(sorted_indices == insider_idx)[0][0]
                ranks.append(rank)
            
            # Plot
            total = len(y_true)
            ax.scatter(ranks, [1]*len(ranks), s=200, c='red', marker='*', 
                      label=f'Insiders (n={len(ranks)})', zorder=3)
            ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
            
            # Mark zones
            ax.axvspan(0, total*0.05, alpha=0.2, color='green', label='Top 5%')
            ax.axvspan(total*0.05, total*0.10, alpha=0.1, color='yellow', label='Top 10%')
            
            ax.set_xlim(-10, total)
            ax.set_ylim(0.5, 1.5)
            ax.set_xlabel('Rank Position')
            ax.set_title(model_name)
            ax.set_yticks([])
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = config.RESULT_PATHS['visualizations'] / 'insider_ranking_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"\nRanking analysis saved to {save_path}")
    
    def run_full_evaluation(self):
        """Run complete evaluation for all models"""
        logger.info(utils.generate_report_header("IMBALANCED DATA EVALUATION"))
        
        all_results = []
        
        for model_name in self.models:
            try:
                results = self.evaluate_model(model_name)
                if results:
                    all_results.append(results)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        if not all_results:
            logger.error("No results to compare!")
            return
        
        # Create comparison table
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE COMPARISON")
        logger.info("="*80 + "\n")
        
        comparison_df = self.create_comparison_table(all_results)
        print(comparison_df.to_string(index=False))
        
        # Save
        comparison_df.to_csv(config.RESULTS_DIR / 'proper_evaluation_metrics.csv', index=False)
        logger.info(f"\nSaved to: {config.RESULTS_DIR / 'proper_evaluation_metrics.csv'}")
        
        # Plot ranking analysis
        self.plot_ranking_analysis(all_results)
        
        # Recommendations
        self.print_recommendations(comparison_df)
    
    def print_recommendations(self, df):
        """Print interpretation and thesis recommendations"""
        print("\n" + "="*80)
        print("📊 INTERPRETATION FOR YOUR THESIS")
        print("="*80)
        
        if 'auc_roc' in df.columns:
            best_auc_idx = df['auc_roc'].idxmax()
            best_model = df.loc[best_auc_idx, 'model']
            best_auc = df.loc[best_auc_idx, 'auc_roc']
            
            print(f"\n✅ Best Model (AUC-ROC): {best_model} ({best_auc:.4f})")
            
            if best_auc > 0.85:
                print("   Interpretation: EXCELLENT discriminative ability")
            elif best_auc > 0.75:
                print("   Interpretation: GOOD discriminative ability")
            else:
                print("   Interpretation: MODERATE discriminative ability")
        
        if 'precision_at_100' in df.columns:
            print("\n📈 Precision at Top 100:")
            for _, row in df.iterrows():
                p100 = row['precision_at_100'] * 100
                print(f"   {row['model']}: {p100:.1f}% of top 100 alerts are real")
                if p100 > 20:
                    print("      → EXCELLENT for SOC operations")
                elif p100 > 10:
                    print("      → GOOD for SOC operations")
                elif p100 > 5:
                    print("      → ACCEPTABLE for SOC operations")
                else:
                    print("      → Needs improvement")
        
        print("\n💡 FOR YOUR THESIS, EMPHASIZE:")
        print("   1. AUC-ROC (most reliable for imbalanced data)")
        print("   2. Precision at K (practical for SOC analysts)")
        print("   3. Recall at 5% FPR (catches threats with manageable alerts)")
        print("   4. Insider rankings (shows model finds them early)")
        
        print("\n⚠️  DON'T report traditional F1-Score/Precision")
        print("   With 1 insider in 1000 samples, these metrics are meaningless!")
        print("   Even perfect models would have ~0.1% precision.")
        
        print("\n" + "="*80 + "\n")


def main():
    evaluator = ImbalancedEvaluator()
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()