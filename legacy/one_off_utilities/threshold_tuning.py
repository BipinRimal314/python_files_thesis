"""
Diagnose model results and find the optimal decision threshold
to maximize F1-Score.

This is the final step: tuning the model for real-world performance
by converting anomaly scores into optimal predictions.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config
import utils

logger = utils.logger

class ResultDiagnoser:
    
    def __init__(self, models: list = ['isolation_forest', 'deep_clustering', 'lstm_autoencoder']):
        """
        Initialize the diagnoser.
        
        Args:
            models: List of model names to diagnose.
        """
        self.models = models
        self.results_dir = config.RESULTS_DIR
        self.best_thresholds = {}
        sns.set_style(config.VISUALIZATION['style'])
        plt.rcParams['figure.figsize'] = config.VISUALIZATION['figure_size']
        plt.rcParams['figure.dpi'] = config.VISUALIZATION['dpi']

    def load_predictions(self, model_name: str) -> pd.DataFrame:
        """Loads the prediction file for a given model."""
        try:
            pred_file = self.results_dir / f"{model_name}_predictions.csv"
            df = pd.read_csv(pred_file)
            logger.info(f"Loaded {len(df)} predictions for {model_name}")
            return df
        except Exception as e:
            logger.error(f"Could not load predictions for {model_name}: {e}")
            return None

    def find_optimal_threshold(self, y_true: np.ndarray, anomaly_scores: np.ndarray) -> dict:
        """
        Iterate through potential thresholds to find the one that
        maximizes the F1-Score for the positive (insider) class.
        """
        best_f1 = 0
        best_threshold = 0
        best_metrics = {}
        
        # We test 100 threshold levels
        thresholds = np.linspace(np.min(anomaly_scores), np.max(anomaly_scores), 100)
        
        for threshold in thresholds:
            # Predict '1' (anomaly) if score is GREATER than the threshold
            predictions = (anomaly_scores > threshold).astype(int)
            
            # Calculate F1 for the positive class (label=1)
            f1 = f1_score(y_true, predictions, pos_label=1, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'f1_score': f1,
                    'precision': precision_score(y_true, predictions, pos_label=1, zero_division=0),
                    'recall': recall_score(y_true, predictions, pos_label=1, zero_division=0)
                }
        
        return best_metrics

    def plot_threshold_analysis(self, model_name: str, y_true: np.ndarray, anomaly_scores: np.ndarray, best_metrics: dict):
        """
        Plots Precision, Recall, and F1-Score vs. Threshold.
        """
        thresholds = np.linspace(np.min(anomaly_scores), np.max(anomaly_scores), 100)
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for threshold in thresholds:
            predictions = (anomaly_scores > threshold).astype(int)
            precision_scores.append(precision_score(y_true, predictions, pos_label=1, zero_division=0))
            recall_scores.append(recall_score(y_true, predictions, pos_label=1, zero_division=0))
            f1_scores.append(f1_score(y_true, predictions, pos_label=1, zero_division=0))

        plt.figure()
        plt.plot(thresholds, precision_scores, label='Precision', color='blue')
        plt.plot(thresholds, recall_scores, label='Recall', color='green')
        plt.plot(thresholds, f1_scores, label='F1-Score', color='red', linestyle='--', linewidth=2)
        
        # Mark the best threshold
        plt.axvline(
            x=best_metrics['threshold'], 
            color='purple', 
            linestyle=':', 
            label=f"Best F1 ({best_metrics['f1_score']:.2f}) at t={best_metrics['threshold']:.2f}"
        )
        
        plt.title(f"{model_name.replace('_', ' ').title()} - Threshold Analysis")
        plt.xlabel("Anomaly Score Threshold")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        save_path = config.RESULT_PATHS['visualizations'] / f"{model_name}_threshold_analysis.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Threshold analysis plot saved to {save_path}")

    def run_diagnosis(self):
        """
        Run the full diagnosis and tuning pipeline for all models.
        """
        logger.info(utils.generate_report_header("MODEL DIAGNOSIS & THRESHOLD TUNING"))
        
        final_report = []

        for model_name in self.models:
            logger.info("\n" + "="*80)
            logger.info(f"Diagnosing: {model_name.replace('_', ' ').title()}")
            logger.info("="*80)
            
            df = self.load_predictions(model_name)
            if df is None:
                continue
                
            y_true = df['true_label']
            anomaly_scores = df['anomaly_score']

            if y_true.sum() == 0:
                logger.error("No true positive labels found in prediction file. Cannot diagnose.")
                continue

            # Find the best threshold
            best_metrics = self.find_optimal_threshold(y_true, anomaly_scores)
            self.best_thresholds[model_name] = best_metrics
            
            logger.info("--- Optimal Threshold Found ---")
            logger.info(f"Best F1-Score: {best_metrics['f1_score']:.4f}")
            logger.info(f"Precision:     {best_metrics['precision']:.4f}")
            logger.info(f"Recall:        {best_metrics['recall']:.4f}")
            logger.info(f"Threshold:     {best_metrics['threshold']:.4f}")
            
            # Plot the analysis
            self.plot_threshold_analysis(model_name, y_true, anomaly_scores, best_metrics)
            
            final_report.append({
                "model": model_name.replace('_', ' ').title(),
                **best_metrics
            })
        
        # Print final summary
        logger.info("\n" + "="*80)
        logger.info("OPTIMAL PERFORMANCE SUMMARY")
        logger.info("="*80)
        report_df = pd.DataFrame(final_report).set_index('model')
        report_df = report_df[['f1_score', 'precision', 'recall', 'threshold']]
        print(report_df.to_markdown(floatfmt=".4f"))
        
        # Save to file
        report_df.to_csv(config.RESULTS_DIR / 'optimal_evaluation_report.csv')
        logger.info(f"\nOptimal report saved to {config.RESULTS_DIR / 'optimal_evaluation_report.csv'}")


def main():
    diagnoser = ResultDiagnoser()
    diagnoser.run_diagnosis()

if __name__ == "__main__":
    main()
