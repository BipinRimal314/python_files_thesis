"""
Model Evaluation and Comparison
Comprehensive performance analysis of all detection models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from pathlib import Path
import config
import utils

logger = utils.logger

class ModelEvaluator:
    """
    Evaluates and compares multiple anomaly detection models
    """
    
    def __init__(self):
        """Initialize the model evaluator"""
        self.results = {}
        self.metrics_summary = None
        
        # Set plotting style
        plt.style.use(config.VISUALIZATION['style'])
        sns.set_palette(config.VISUALIZATION['color_palette'])
    
    def load_model_results(self, model_name: str) -> dict:
        """
        Load predictions and scores for a model
        
        Args:
            model_name: Name of the model ('isolation_forest', 'lstm_autoencoder', 'deep_clustering')
            
        Returns:
            Dictionary with predictions and scores
        """
        predictions_file = config.RESULTS_DIR / f"{model_name}_predictions.csv"
        metrics_file = config.RESULTS_DIR / f"{model_name}_metrics.csv"
        
        results = {}
        
        if predictions_file.exists():
            df = pd.read_csv(predictions_file)
            results['predictions'] = df['prediction'].values
            results['scores'] = df['anomaly_score'].values
            results['true_labels'] = df['true_label'].values
        
        if metrics_file.exists():
            metrics_df = pd.read_csv(metrics_file)
            results['metrics'] = metrics_df.iloc[0].to_dict()
        
        return results
    
    def load_all_results(self):
        """Load results for all models"""
        logger.info("Loading results for all models...")
        
        model_names = ['isolation_forest', 'lstm_autoencoder', 'deep_clustering']
        
        for model_name in model_names:
            results = self.load_model_results(model_name)
            if results:
                self.results[model_name] = results
                logger.info(f"Loaded results for {model_name}")
        
        logger.info(f"Loaded results for {len(self.results)} models")
    
    def create_metrics_comparison_table(self) -> pd.DataFrame:
        """
        Create a comparison table of all model metrics
        
        Returns:
            DataFrame with metrics for all models
        """
        logger.info("Creating metrics comparison table...")
        
        metrics_list = []
        
        for model_name, results in self.results.items():
            if 'metrics' in results:
                metrics = results['metrics'].copy()
                metrics['model'] = model_name.replace('_', ' ').title()
                metrics_list.append(metrics)
        
        if not metrics_list:
            logger.warning("No metrics found to compare")
            return pd.DataFrame()
        
        df = pd.DataFrame(metrics_list)
        
        # Select key metrics for display
        display_cols = ['model', 'accuracy', 'precision', 'recall', 'f1_score']
        if 'auc_roc' in df.columns:
            display_cols.append('auc_roc')
        
        df_display = df[display_cols].copy()
        
        # Round values
        for col in display_cols[1:]:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(4)
        
        self.metrics_summary = df_display
        
        logger.info("Metrics comparison table created")
        return df_display
    
    def plot_confusion_matrices(self):
        """Create confusion matrix plots for all models"""
        logger.info("Creating confusion matrix plots...")
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            if 'predictions' not in results or 'true_labels' not in results:
                continue
            
            cm = confusion_matrix(results['true_labels'], results['predictions'])
            
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=axes[idx],
                cbar=True
            )
            
            axes[idx].set_title(f"{model_name.replace('_', ' ').title()}")
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        save_path = config.RESULT_PATHS['confusion_matrices'] / 'all_models.png'
        plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
        logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curves(self):
        """Create ROC curve comparison plot"""
        logger.info("Creating ROC curves...")
        
        plt.figure(figsize=config.VISUALIZATION['figure_size'])
        
        for model_name, results in self.results.items():
            if 'scores' not in results or 'true_labels' not in results:
                continue
            
            fpr, tpr, _ = roc_curve(results['true_labels'], results['scores'])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                label=f"{model_name.replace('_', ' ').title()} (AUC = {roc_auc:.3f})",
                linewidth=2
            )
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        
        save_path = config.RESULT_PATHS['roc_curves'] / 'roc_comparison.png'
        plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curves(self):
        """Create Precision-Recall curve comparison plot"""
        logger.info("Creating Precision-Recall curves...")
        
        plt.figure(figsize=config.VISUALIZATION['figure_size'])
        
        for model_name, results in self.results.items():
            if 'scores' not in results or 'true_labels' not in results:
                continue
            
            precision, recall, _ = precision_recall_curve(
                results['true_labels'],
                results['scores']
            )
            avg_precision = average_precision_score(
                results['true_labels'],
                results['scores']
            )
            
            plt.plot(
                recall, precision,
                label=f"{model_name.replace('_', ' ').title()} (AP = {avg_precision:.3f})",
                linewidth=2
            )
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(alpha=0.3)
        
        save_path = config.RESULT_PATHS['visualizations'] / 'precision_recall_comparison.png'
        plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
        logger.info(f"Precision-Recall curves saved to {save_path}")
        
        plt.close()
    
    def plot_metrics_comparison_bar(self):
        """Create bar chart comparing key metrics"""
        logger.info("Creating metrics comparison bar chart...")
        
        if self.metrics_summary is None or len(self.metrics_summary) == 0:
            logger.warning("No metrics summary available")
            return
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        available_metrics = [m for m in metrics_to_plot if m in self.metrics_summary.columns]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            data = self.metrics_summary[['model', metric]].sort_values(metric, ascending=False)
            
            bars = ax.bar(data['model'], data[metric], alpha=0.7, edgecolor='black')
            
            # Color bars
            colors = plt.cm.Set2(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9
                )
            
            ax.tick_params(axis='x', rotation=15)
        
        # Remove extra subplot if odd number of metrics
        if len(available_metrics) < 4:
            fig.delaxes(axes[3])
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        save_path = config.RESULT_PATHS['visualizations'] / 'metrics_comparison_bar.png'
        plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
        logger.info(f"Metrics comparison bar chart saved to {save_path}")
        
        plt.close()
    
    def plot_score_distributions(self):
        """Plot anomaly score distributions for each model"""
        logger.info("Creating anomaly score distribution plots...")
        
        n_models = len(self.results)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4*n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            if 'scores' not in results or 'true_labels' not in results:
                continue
            
            normal_scores = results['scores'][results['true_labels'] == 0]
            anomaly_scores = results['scores'][results['true_labels'] == 1]
            
            axes[idx].hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='green')
            axes[idx].hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red')
            
            axes[idx].set_xlabel('Anomaly Score')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f"{model_name.replace('_', ' ').title()} - Score Distribution")
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = config.RESULT_PATHS['visualizations'] / 'score_distributions.png'
        plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
        logger.info(f"Score distributions saved to {save_path}")
        
        plt.close()
    
    def generate_evaluation_report(self) -> str:
        """
        Generate a comprehensive evaluation report
        
        Returns:
            Report as string
        """
        logger.info("Generating evaluation report...")
        
        report = []
        report.append(utils.generate_report_header("MODEL EVALUATION REPORT"))
        report.append(f"Generated: {utils.timestamp()}\n")
        
        # Summary table
        if self.metrics_summary is not None:
            report.append("\n" + "="*80)
            report.append("PERFORMANCE METRICS COMPARISON")
            report.append("="*80 + "\n")
            report.append(self.metrics_summary.to_string(index=False))
            report.append("\n")
        
        # Individual model details
        for model_name, results in self.results.items():
            if 'metrics' not in results:
                continue
            
            report.append("\n" + "-"*80)
            report.append(f"{model_name.replace('_', ' ').upper()} - DETAILED METRICS")
            report.append("-"*80 + "\n")
            
            metrics = results['metrics']
            for key, value in metrics.items():
                if key != 'model' and not key.startswith('Unnamed'):
                    if isinstance(value, (int, float)):
                        report.append(f"{key.replace('_', ' ').title()}: {value:.4f}")
                    else:
                        report.append(f"{key.replace('_', ' ').title()}: {value}")
            report.append("")
        
        # Best model
        if self.metrics_summary is not None and 'f1_score' in self.metrics_summary.columns:
            best_model = self.metrics_summary.loc[self.metrics_summary['f1_score'].idxmax()]
            report.append("\n" + "="*80)
            report.append("BEST PERFORMING MODEL")
            report.append("="*80)
            report.append(f"Model: {best_model['model']}")
            report.append(f"F1-Score: {best_model['f1_score']:.4f}")
            if 'auc_roc' in best_model:
                report.append(f"AUC-ROC: {best_model['auc_roc']:.4f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = config.RESULTS_DIR / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return report_text
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        logger.info(utils.generate_report_header("COMPREHENSIVE MODEL EVALUATION"))
        
        # Load all results
        self.load_all_results()
        
        if not self.results:
            logger.error("No model results found. Train models first.")
            return
        
        # Create metrics comparison
        self.create_metrics_comparison_table()
        
        # Create all visualizations
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_metrics_comparison_bar()
        self.plot_score_distributions()
        
        # Generate report
        report = self.generate_evaluation_report()
        
        # Print report
        print(report)
        
        logger.info("Comprehensive evaluation completed!")


def main():
    """Main execution function"""
    evaluator = ModelEvaluator()
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()