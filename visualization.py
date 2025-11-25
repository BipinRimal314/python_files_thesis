"""
Visualization Dashboard for Insider Threat Detection Results
Creates comprehensive visualizations for analysis and reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import config
import utils

logger = utils.logger

class VisualizationDashboard:
    """
    Creates comprehensive visualizations for insider threat detection results
    """
    
    def __init__(self):
        """Initialize the visualization dashboard"""
        plt.style.use(config.VISUALIZATION['style'])
        sns.set_palette(config.VISUALIZATION['color_palette'])
        self.save_dir = config.RESULT_PATHS['visualizations']
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_alert_severity_distribution(self):
        """Plot distribution of alert severities"""
        logger.info("Creating alert severity distribution plot...")
        
        try:
            alerts_df = pd.read_csv(config.RESULT_PATHS['alerts'])
        except:
            logger.warning("Alerts file not found")
            return
        
        plt.figure(figsize=(10, 6))
        
        severity_counts = alerts_df['severity'].value_counts()
        colors = {'critical': 'darkred', 'high': 'red', 'medium': 'orange', 'low': 'yellow'}
        
        bars = plt.bar(
            severity_counts.index,
            severity_counts.values,
            color=[colors.get(s, 'gray') for s in severity_counts.index]
        )
        
        plt.xlabel('Severity Level', fontsize=12)
        plt.ylabel('Number of Alerts', fontsize=12)
        plt.title('Alert Severity Distribution', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10
            )
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'alert_severity_distribution.png', dpi=300)
        plt.close()
        
        logger.info("Alert severity distribution plot saved")
    
    def plot_model_agreement(self):
        """Plot how often models agree on anomalies"""
        logger.info("Creating model agreement plot...")
        
        try:
            alerts_df = pd.read_csv(config.RESULT_PATHS['alerts'])
        except:
            logger.warning("Alerts file not found")
            return
        
        plt.figure(figsize=(10, 6))
        
        agreement_counts = alerts_df['num_models_agree'].value_counts().sort_index()
        
        plt.bar(agreement_counts.index, agreement_counts.values, color='steelblue', alpha=0.7)
        plt.xlabel('Number of Models in Agreement', fontsize=12)
        plt.ylabel('Number of Alerts', fontsize=12)
        plt.title('Model Agreement on Anomalies', fontsize=14, fontweight='bold')
        plt.xticks(range(1, 4))
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'model_agreement.png', dpi=300)
        plt.close()
        
        logger.info("Model agreement plot saved")
    
    def plot_anomaly_score_timeline(self, model_name: str = 'ensemble_weighted'):
        """Plot anomaly scores over time (if temporal data available)"""
        logger.info(f"Creating anomaly score timeline for {model_name}...")
        
        try:
            predictions_df = pd.read_csv(config.RESULTS_DIR / f'{model_name}_predictions.csv')
        except:
            logger.warning(f"Predictions file not found for {model_name}")
            return
        
        plt.figure(figsize=(14, 6))
        
        # Plot scores
        x = range(len(predictions_df))
        plt.plot(x, predictions_df['anomaly_score'], alpha=0.6, linewidth=0.5, color='blue')
        
        # Highlight actual anomalies
        anomalies = predictions_df[predictions_df['true_label'] == 1]
        plt.scatter(
            anomalies.index,
            anomalies['anomaly_score'],
            color='red',
            s=50,
            alpha=0.8,
            label='True Anomalies',
            zorder=5
        )
        
        # Add threshold line
        threshold = config.ENSEMBLE['final_threshold']
        plt.axhline(y=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Anomaly Score', fontsize=12)
        plt.title(f'Anomaly Score Timeline - {model_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'anomaly_score_timeline_{model_name}.png', dpi=300)
        plt.close()
        
        logger.info(f"Anomaly score timeline saved for {model_name}")
    
    def plot_feature_importance_heatmap(self):
        """Plot feature importance if available"""
        logger.info("Creating feature importance visualization...")
        
        # This is a placeholder - would need actual feature importance data
        # For demonstration, creating a synthetic example
        
        features = ['login_count', 'after_hours_ratio', 'file_access', 'email_volume', 
                   'usb_usage', 'weekend_activity', 'sensitive_files', 'data_transfer']
        models = ['Isolation Forest', 'Deep Clustering', 'LSTM Autoencoder']
        
        # Synthetic importance scores
        importance_matrix = np.random.rand(len(models), len(features))
        importance_matrix = importance_matrix / importance_matrix.sum(axis=1, keepdims=True)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            importance_matrix,
            xticklabels=features,
            yticklabels=models,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Importance Score'}
        )
        
        plt.title('Feature Importance by Model', fontsize=14, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Models', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'feature_importance_heatmap.png', dpi=300)
        plt.close()
        
        logger.info("Feature importance heatmap saved")
    
    def create_executive_summary_chart(self):
        """Create a single comprehensive summary chart"""
        logger.info("Creating executive summary chart...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Model Performance Comparison
        ax1 = fig.add_subplot(gs[0, :2])
        try:
            comparison_df = pd.read_csv(config.RESULTS_DIR / 'ensemble_weighted_comparison.csv')
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            x = np.arange(len(comparison_df))
            width = 0.2
            
            for i, metric in enumerate(metrics):
                if metric in comparison_df.columns:
                    ax1.bar(x + i*width, comparison_df[metric], width, label=metric.replace('_', ' ').title())
            
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Score')
            ax1.set_title('Model Performance Comparison')
            ax1.set_xticks(x + width * 1.5)
            ax1.set_xticklabels(comparison_df['model'], rotation=15, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
        except Exception as e:
            logger.warning(f"Could not create performance comparison: {e}")
        
        # 2. Alert Severity Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        try:
            alerts_df = pd.read_csv(config.RESULT_PATHS['alerts'])
            severity_counts = alerts_df['severity'].value_counts()
            colors_pie = ['darkred', 'red', 'orange', 'yellow']
            ax2.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%', colors=colors_pie)
            ax2.set_title('Alert Severity Distribution')
        except Exception as e:
            logger.warning(f"Could not create severity pie chart: {e}")
        
        # 3. ROC Curves (if available)
        ax3 = fig.add_subplot(gs[1, :])
        try:
            from sklearn.metrics import roc_curve, auc
            
            model_files = ['isolation_forest', 'lstm_autoencoder', 'deep_clustering', 'ensemble_weighted']
            for model_name in model_files:
                try:
                    pred_df = pd.read_csv(config.RESULTS_DIR / f'{model_name}_predictions.csv')
                    fpr, tpr, _ = roc_curve(pred_df['true_label'], pred_df['anomaly_score'])
                    roc_auc = auc(fpr, tpr)
                    ax3.plot(fpr, tpr, label=f"{model_name.replace('_', ' ').title()} (AUC={roc_auc:.3f})")
                except:
                    continue
            
            ax3.plot([0, 1], [0, 1], 'k--', label='Random')
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curves Comparison')
            ax3.legend()
            ax3.grid(alpha=0.3)
        except Exception as e:
            logger.warning(f"Could not create ROC curves: {e}")
        
        # 4. Top Anomalies
        ax4 = fig.add_subplot(gs[2, :])
        try:
            alerts_df = pd.read_csv(config.RESULT_PATHS['alerts'])
            top_alerts = alerts_df.head(10)
            
            colors_bar = ['darkred' if s == 'critical' else 'red' if s == 'high' else 'orange' if s == 'medium' else 'yellow' 
                         for s in top_alerts['severity']]
            
            ax4.barh(range(len(top_alerts)), top_alerts['anomaly_score'], color=colors_bar)
            ax4.set_yticks(range(len(top_alerts)))
            ax4.set_yticklabels([f"Alert {i+1}" for i in range(len(top_alerts))])
            ax4.set_xlabel('Anomaly Score')
            ax4.set_title('Top 10 Alerts by Anomaly Score')
            ax4.grid(axis='x', alpha=0.3)
        except Exception as e:
            logger.warning(f"Could not create top alerts chart: {e}")
        
        plt.suptitle('Insider Threat Detection - Executive Summary', fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(self.save_dir / 'executive_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Executive summary chart saved")
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        logger.info(utils.generate_report_header("GENERATING VISUALIZATIONS"))
        
        self.plot_alert_severity_distribution()
        self.plot_model_agreement()
        self.plot_anomaly_score_timeline('ensemble_weighted')
        self.plot_feature_importance_heatmap()
        self.create_executive_summary_chart()
        
        logger.info(f"All visualizations saved to {self.save_dir}")
        logger.info("Visualization generation completed!")


def main():
    """Main execution function"""
    dashboard = VisualizationDashboard()
    dashboard.generate_all_visualizations()


if __name__ == "__main__":
    main()