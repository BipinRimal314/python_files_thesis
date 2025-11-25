"""
Ensemble System for Insider Threat Detection
Combines multiple models for improved detection accuracy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import config
import utils

logger = utils.logger

class EnsembleDetector:
    """
    Ensemble system combining multiple anomaly detection models
    """
    
    def __init__(self, method: str = 'weighted'):
        """
        Initialize ensemble detector
        
        Args:
            method: Ensemble method ('weighted', 'majority', 'cascade')
        """
        self.method = method
        self.weights = config.ENSEMBLE['weights']
        self.model_results = {}
        self.ensemble_scores = None
        self.ensemble_predictions = None
        
        logger.info(f"Initialized Ensemble Detector with method: {method}")
    
    def load_model_predictions(self):
        """Load predictions from all individual models"""
        logger.info("Loading predictions from individual models...")
        
        model_names = ['isolation_forest', 'lstm_autoencoder', 'deep_clustering']
        
        for model_name in model_names:
            predictions_file = config.RESULTS_DIR / f"{model_name}_predictions.csv"
            
            if predictions_file.exists():
                df = pd.read_csv(predictions_file)
                self.model_results[model_name] = {
                    'predictions': df['prediction'].values,
                    'scores': df['anomaly_score'].values,
                    'true_labels': df['true_label'].values
                }
                logger.info(f"Loaded {model_name} predictions: {len(df)} samples")
            else:
                logger.warning(f"Predictions file not found for {model_name}")
        
        if not self.model_results:
            raise ValueError("No model predictions found. Train models first.")
        
        # Check if all models have same number of samples
        sample_counts = {name: len(results['predictions']) 
                        for name, results in self.model_results.items()}
        
        if len(set(sample_counts.values())) > 1:
            logger.warning(f"Models have different sample counts: {sample_counts}")
            logger.info("Will use aggregation strategy to align predictions...")
            self._align_predictions()
        
        logger.info(f"Loaded predictions from {len(self.model_results)} models")
    
    def _align_predictions(self):
        """
        Align predictions from models with different sample counts.
        For LSTM (sequence-based), aggregate to match feature-based models.
        """
        logger.info("Aligning predictions across models...")
        
        # Find the model with fewest samples (likely the aggregated feature models)
        min_samples = min(len(results['predictions']) for results in self.model_results.values())
        reference_model = None
        
        for model_name, results in self.model_results.items():
            if len(results['predictions']) == min_samples:
                reference_model = model_name
                break
        
        logger.info(f"Using {reference_model} as reference ({min_samples} samples)")
        
        # Align all models to reference size
        aligned_results = {}
        
        for model_name, results in self.model_results.items():
            n_samples = len(results['predictions'])
            
            if n_samples == min_samples:
                # Already aligned
                aligned_results[model_name] = results
            else:
                # Need to aggregate (downsample)
                logger.info(f"Aggregating {model_name} from {n_samples} to {min_samples} samples")
                
                # Strategy: Take every nth sample to downsample evenly
                indices = np.linspace(0, n_samples - 1, min_samples, dtype=int)
                
                aligned_results[model_name] = {
                    'predictions': results['predictions'][indices],
                    'scores': results['scores'][indices],
                    'true_labels': results['true_labels'][indices]
                }
        
        self.model_results = aligned_results
        logger.info("Predictions aligned successfully")
    
    def weighted_ensemble(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted voting ensemble
        Combines anomaly scores using predefined weights
        
        Returns:
            Tuple of (ensemble_predictions, ensemble_scores)
        """
        logger.info("Computing weighted ensemble...")
        
        # Initialize ensemble scores
        n_samples = len(next(iter(self.model_results.values()))['scores'])
        ensemble_scores = np.zeros(n_samples)
        
        # Weighted average of scores
        total_weight = 0
        for model_name, results in self.model_results.items():
            weight = self.weights.get(model_name, 1.0 / len(self.model_results))
            ensemble_scores += weight * results['scores']
            total_weight += weight
        
        # Normalize
        ensemble_scores /= total_weight
        
        # Apply threshold
        threshold = config.ENSEMBLE['final_threshold']
        ensemble_predictions = (ensemble_scores > threshold).astype(int)
        
        logger.info(f"Weighted ensemble: {ensemble_predictions.sum()} anomalies detected")
        
        return ensemble_predictions, ensemble_scores
    
    def majority_voting_ensemble(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Majority voting ensemble
        An instance is anomalous if majority of models agree
        
        Returns:
            Tuple of (ensemble_predictions, ensemble_scores)
        """
        logger.info("Computing majority voting ensemble...")
        
        # Stack predictions from all models
        all_predictions = np.array([
            results['predictions'] for results in self.model_results.values()
        ])
        
        # Majority vote
        ensemble_predictions = (np.mean(all_predictions, axis=0) > 0.5).astype(int)
        
        # Average scores for confidence
        all_scores = np.array([
            results['scores'] for results in self.model_results.values()
        ])
        ensemble_scores = np.mean(all_scores, axis=0)
        
        logger.info(f"Majority voting: {ensemble_predictions.sum()} anomalies detected")
        
        return ensemble_predictions, ensemble_scores
    
    def cascade_ensemble(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cascade ensemble
        Models are applied in sequence, with each filtering candidates
        
        Returns:
            Tuple of (ensemble_predictions, ensemble_scores)
        """
        logger.info("Computing cascade ensemble...")
        
        # Start with Isolation Forest (fast screening)
        if 'isolation_forest' in self.model_results:
            candidates = self.model_results['isolation_forest']['predictions'] == 1
            ensemble_scores = self.model_results['isolation_forest']['scores'].copy()
        else:
            # Fallback to first available model
            first_model = next(iter(self.model_results.values()))
            candidates = first_model['predictions'] == 1
            ensemble_scores = first_model['scores'].copy()
        
        # Refine with other models
        for model_name in ['deep_clustering', 'lstm_autoencoder']:
            if model_name in self.model_results:
                model_predictions = self.model_results[model_name]['predictions']
                model_scores = self.model_results[model_name]['scores']
                
                # Keep only candidates confirmed by this model
                candidates = candidates & (model_predictions == 1)
                
                # Update scores (take maximum)
                ensemble_scores = np.maximum(ensemble_scores, model_scores)
        
        ensemble_predictions = candidates.astype(int)
        
        logger.info(f"Cascade ensemble: {ensemble_predictions.sum()} anomalies detected")
        
        return ensemble_predictions, ensemble_scores
    
    def compute_ensemble(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ensemble predictions based on selected method
        
        Returns:
            Tuple of (ensemble_predictions, ensemble_scores)
        """
        if self.method == 'weighted':
            return self.weighted_ensemble()
        elif self.method == 'majority':
            return self.majority_voting_ensemble()
        elif self.method == 'cascade':
            return self.cascade_ensemble()
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def evaluate_ensemble(self, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble performance
        
        Args:
            y_true: True labels
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Evaluating ensemble performance...")
        
        if self.ensemble_predictions is None or self.ensemble_scores is None:
            raise ValueError("Compute ensemble predictions first")
        
        metrics = utils.calculate_metrics(
            y_true,
            self.ensemble_predictions,
            self.ensemble_scores
        )
        
        utils.print_metrics(metrics, f"Ensemble ({self.method})")
        
        return metrics
    
    def compare_with_individual_models(self, y_true: np.ndarray) -> pd.DataFrame:
        """
        Compare ensemble with individual models
        
        Args:
            y_true: True labels
            
        Returns:
            DataFrame with comparison metrics
        """
        logger.info("Comparing ensemble with individual models...")
        
        comparison_data = []
        
        # Individual models
        for model_name, results in self.model_results.items():
            metrics = utils.calculate_metrics(
                y_true,
                results['predictions'],
                results['scores']
            )
            metrics['model'] = model_name.replace('_', ' ').title()
            comparison_data.append(metrics)
        
        # Ensemble
        ensemble_metrics = self.evaluate_ensemble(y_true)
        ensemble_metrics['model'] = f"Ensemble ({self.method})"
        comparison_data.append(ensemble_metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Select key metrics
        display_cols = ['model', 'accuracy', 'precision', 'recall', 'f1_score']
        if 'auc_roc' in comparison_df.columns:
            display_cols.append('auc_roc')
        
        comparison_df = comparison_df[display_cols]
        
        # Round values
        for col in display_cols[1:]:
            comparison_df[col] = comparison_df[col].round(4)
        
        logger.info("\nComparison Results:")
        logger.info("\n" + comparison_df.to_string(index=False))
        
        return comparison_df
    
    def generate_alerts(self, anomaly_threshold: float = None) -> pd.DataFrame:
        """
        Generate actionable alerts from ensemble predictions
        
        Args:
            anomaly_threshold: Override default threshold
            
        Returns:
            DataFrame with alerts and contextual information
        """
        logger.info("Generating alerts...")
        
        if self.ensemble_scores is None:
            raise ValueError("Compute ensemble predictions first")
        
        if anomaly_threshold is None:
            anomaly_threshold = config.ENSEMBLE['final_threshold']
        
        # Filter for anomalies
        anomaly_indices = np.where(self.ensemble_scores > anomaly_threshold)[0]
        
        alerts = []
        for idx in anomaly_indices:
            score = self.ensemble_scores[idx]
            
            # Determine severity
            severity = 'low'
            for level, (min_score, max_score) in config.ALERTS['severity_levels'].items():
                if min_score <= score <= max_score:
                    severity = level
                    break
            
            # Get contributing models
            contributing_models = []
            for model_name, results in self.model_results.items():
                if results['predictions'][idx] == 1:
                    contributing_models.append(model_name)
            
            alert = {
                'alert_id': f"ALERT_{idx:06d}",
                'index': idx,
                'anomaly_score': score,
                'severity': severity,
                'contributing_models': ', '.join(contributing_models),
                'num_models_agree': len(contributing_models)
            }
            alerts.append(alert)
        
        alerts_df = pd.DataFrame(alerts)
        
        # Sort by severity and score
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        alerts_df['severity_rank'] = alerts_df['severity'].map(severity_order)
        alerts_df = alerts_df.sort_values(
            ['severity_rank', 'anomaly_score'],
            ascending=[False, False]
        ).drop('severity_rank', axis=1)
        
        logger.info(f"Generated {len(alerts_df)} alerts")
        logger.info(f"Severity distribution:\n{alerts_df['severity'].value_counts()}")
        
        return alerts_df
    
    def save_results(self):
        """Save ensemble results and alerts"""
        # Save predictions
        if self.ensemble_predictions is not None:
            y_true = next(iter(self.model_results.values()))['true_labels']
            
            results_df = pd.DataFrame({
                'true_label': y_true,
                'prediction': self.ensemble_predictions,
                'anomaly_score': self.ensemble_scores
            })
            results_df.to_csv(
                config.RESULTS_DIR / f'ensemble_{self.method}_predictions.csv',
                index=False
            )
            logger.info(f"Ensemble predictions saved")
        
        # Save alerts
        alerts_df = self.generate_alerts()
        alerts_df.to_csv(config.RESULT_PATHS['alerts'], index=False)
        logger.info(f"Alerts saved to {config.RESULT_PATHS['alerts']}")
    
    def run_ensemble_pipeline(self) -> Dict:
        """
        Run complete ensemble pipeline
        
        Returns:
            Dictionary with results and metrics
        """
        logger.info(utils.generate_report_header("ENSEMBLE SYSTEM PIPELINE"))
        
        # Load predictions
        self.load_model_predictions()
        
        # Compute ensemble
        self.ensemble_predictions, self.ensemble_scores = self.compute_ensemble()
        
        # Get true labels
        y_true = next(iter(self.model_results.values()))['true_labels']
        
        # Evaluate
        metrics = self.evaluate_ensemble(y_true)
        
        # Compare with individual models
        comparison_df = self.compare_with_individual_models(y_true)
        
        # Save results
        self.save_results()
        
        # Save comparison
        comparison_df.to_csv(
            config.RESULTS_DIR / f'ensemble_{self.method}_comparison.csv',
            index=False
        )
        
        logger.info("Ensemble pipeline completed!")
        
        return {
            'metrics': metrics,
            'comparison': comparison_df,
            'predictions': self.ensemble_predictions,
            'scores': self.ensemble_scores
        }


def main():
    """Main execution function"""
    # Try different ensemble methods
    methods = ['weighted', 'majority', 'cascade']
    
    results = {}
    
    for method in methods:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing ensemble method: {method.upper()}")
        logger.info(f"{'='*80}\n")
        
        try:
            ensemble = EnsembleDetector(method=method)
            results[method] = ensemble.run_ensemble_pipeline()
        except Exception as e:
            logger.error(f"Error with {method} ensemble: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("ENSEMBLE METHODS COMPARISON")
    print("="*80 + "\n")
    
    for method, result in results.items():
        if 'metrics' in result:
            print(f"\n{method.upper()} Ensemble:")
            print(f"  Accuracy:  {result['metrics']['accuracy']:.4f}")
            print(f"  Precision: {result['metrics']['precision']:.4f}")
            print(f"  Recall:    {result['metrics']['recall']:.4f}")
            print(f"  F1-Score:  {result['metrics']['f1_score']:.4f}")
            if 'auc_roc' in result['metrics']:
                print(f"  AUC-ROC:   {result['metrics']['auc_roc']:.4f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()