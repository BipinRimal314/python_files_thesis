"""
Diagnose V2 model results and find the optimal decision threshold
to maximize F1-Score.

[--- V4.3 FINAL FIX ---]
- Added 'ensemble' to the models list.
- Ensures 'ensemble' looks for the correct file pattern.
- Handles binarization robustly to avoid multiclass errors.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from pathlib import Path

# Use the V2 config
import config_v2 as config
import utils

logger = utils.logger

class ResultDiagnoserV2:
    
    def __init__(self):
        # --- UPDATED MODEL LIST TO INCLUDE ENSEMBLE ---
        self.models = ['isolation_forest', 'deep_clustering', 'lstm_autoencoder', 'ensemble']
        self.results_dir = config.RESULTS_DIR

        subset = config.DATASET_SUBSET if hasattr(config, 'DATASET_SUBSET') and config.DATASET_SUBSET else []
        self.subset_name = "_".join(subset) if subset else "ALL"

        self.best_thresholds = {}
        plt.style.use(config.VISUALIZATION['style'])
        plt.rcParams['figure.figsize'] = config.VISUALIZATION['figure_size']
        plt.rcParams['figure.dpi'] = config.VISUALIZATION['dpi']

    def load_predictions(self, model_name: str) -> pd.DataFrame:
        try:
            # Ensemble has a slightly different naming pattern in some contexts, 
            # but your ensemble_v2.py saves it as: ensemble_predictions_{subset}_v2.csv
            # This matches the pattern for other models perfectly.
            pred_file = self.results_dir / f"{model_name}_predictions_{self.subset_name}_v2.csv"
            
            if not pred_file.exists():
                logger.warning(f"File not found: {pred_file}")
                return None
                
            df = pd.read_csv(pred_file)
            logger.info(f"Loaded {len(df)} predictions for {model_name}")
            return df
        except Exception as e:
            logger.error(f"Could not load predictions for {model_name}: {e}")
            return None

    def binarize_labels(self, y):
        """Force all non-zero values to 1 for binary metrics."""
        y = pd.Series(y).fillna(0)
        # Ensure we handle floats like 1.0 or strings like "1"
        y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
        y = np.where(y > 0, 1, 0)
        return y

    def find_optimal_threshold(self, y_true: np.ndarray, anomaly_scores: np.ndarray) -> dict:
        y_true = self.binarize_labels(y_true)
        
        # Handle NaN scores
        anomaly_scores = pd.Series(anomaly_scores).fillna(0).values

        best_f1, best_threshold = 0, 0
        best_metrics = {
            'threshold': 0, 'f1_score': 0, 'precision': 0, 'recall': 0
        }

        if len(np.unique(anomaly_scores)) > 1000:
            # Optimization: if too many unique scores, sample 1000 quantiles
            thresholds = np.unique(np.quantile(anomaly_scores, np.linspace(0, 1, 1000)))
        else:
            thresholds = np.unique(anomaly_scores)

        for threshold in thresholds:
            predictions = (anomaly_scores > threshold).astype(int)
            
            # Manual calc to be safe against sklearn warnings
            tp = np.sum((predictions == 1) & (y_true == 1))
            fp = np.sum((predictions == 1) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall
                }

        return best_metrics

    def plot_threshold_analysis(self, model_name: str, y_true: np.ndarray, anomaly_scores: np.ndarray, best_metrics: dict):
        y_true = self.binarize_labels(y_true)
        anomaly_scores = pd.Series(anomaly_scores).fillna(0).values

        thresholds = np.linspace(np.min(anomaly_scores), np.max(anomaly_scores), 100)
        precision_scores, recall_scores, f1_scores = [], [], []

        for threshold in thresholds:
            predictions = (anomaly_scores > threshold).astype(int)
            
            tp = np.sum((predictions == 1) & (y_true == 1))
            fp = np.sum((predictions == 1) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            precision_scores.append(prec)
            recall_scores.append(rec)
            f1_scores.append(f1)

        plt.figure()
        plt.plot(thresholds, precision_scores, label='Precision', color='blue')
        plt.plot(thresholds, recall_scores, label='Recall', color='green')
        plt.plot(thresholds, f1_scores, label='F1-Score', color='red', linestyle='--', linewidth=2)

        plt.axvline(x=best_metrics['threshold'], color='purple', linestyle=':',
                    label=f"Best F1 ({best_metrics['f1_score']:.4f})")

        plt.title(f"{model_name.replace('_', ' ').title()} - Threshold Analysis")
        plt.xlabel("Anomaly Score Threshold")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)

        save_path = config.RESULT_PATHS['visualizations'] / f"{model_name}_threshold_analysis_{self.subset_name}_v2.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Threshold analysis plot saved to {save_path}")

    def run_diagnosis(self):
        logger.info(utils.generate_report_header(f"MODEL DIAGNOSIS & THRESHOLD TUNING (V2: {self.subset_name})"))
        final_report = []

        for model_name in self.models:
            logger.info("\n" + "="*80)
            logger.info(f"Diagnosing: {model_name.replace('_', ' ').title()}")
            logger.info("="*80)

            df = self.load_predictions(model_name)
            if df is None:
                continue

            y_true = self.binarize_labels(df['true_label'])
            anomaly_scores = df['anomaly_score']

            if y_true.sum() == 0:
                logger.error("No positive labels found. Skipping diagnosis.")
                continue

            best_metrics = self.find_optimal_threshold(y_true, anomaly_scores)
            self.best_thresholds[model_name] = best_metrics

            logger.info("--- Optimal Threshold Found ---")
            logger.info(f"Best F1-Score: {best_metrics['f1_score']:.4f}")
            logger.info(f"Precision:     {best_metrics['precision']:.4f}")
            logger.info(f"Recall:        {best_metrics['recall']:.4f}")
            logger.info(f"Threshold:     {best_metrics['threshold']:.4f}")

            self.plot_threshold_analysis(model_name, y_true, anomaly_scores, best_metrics)

            final_report.append({
                "model": model_name.replace('_', ' ').title(),
                **best_metrics
            })

        if final_report:
            report_df = pd.DataFrame(final_report).set_index('model')
            report_df = report_df[['f1_score', 'precision', 'recall', 'threshold']]

            print("\n" + report_df.to_markdown(floatfmt=".4f"))
            report_file = config.RESULTS_DIR / f'optimal_evaluation_report_{self.subset_name}_v2.csv'
            report_df.to_csv(report_file)
            logger.info(f"\nOptimal report saved to {report_file}")
        else:
            logger.error("No valid reports generated.")


def main():
    diagnoser = ResultDiagnoserV2()
    diagnoser.run_diagnosis()

if __name__ == "__main__":
    main()