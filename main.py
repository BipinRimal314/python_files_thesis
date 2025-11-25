"""
Main Execution Pipeline for Insider Threat Detection System
Orchestrates the complete workflow from data preprocessing to final reporting
"""

import sys
import argparse
from pathlib import Path
import config
import utils

# Import all modules
import data_preprocessing
import feature_engineering
import isolation_forest_model
import lstm_autoencoder_model
import deep_clustering_model
import model_evaluation
import ensemble_system
import visualization

logger = utils.logger

class InsiderThreatDetectionPipeline:
    """
    Complete pipeline for insider threat detection
    """
    
    def __init__(self):
        """Initialize the pipeline"""
        self.preprocessor = None
        self.feature_engineer = None
        self.models = {}
        self.results = {}
        
        logger.info("="*80)
        logger.info("INSIDER THREAT DETECTION SYSTEM")
        logger.info("Unsupervised Behavioral Profiling Using Time-Series and Anomaly Detection")
        logger.info("="*80)
    
    def stage_1_data_preprocessing(self, insider_list=None):
        """
        Stage 1: Data Preprocessing
        Load, clean, and prepare raw CMU-CERT dataset
        
        Args:
            insider_list: Optional list of known insider users
        """
        logger.info(utils.generate_report_header("STAGE 1: DATA PREPROCESSING"))
        
        self.preprocessor = data_preprocessing.DataPreprocessor()
        processed_data = self.preprocessor.run_full_pipeline(insider_list=insider_list)
        
        logger.info(f"✓ Stage 1 completed: {len(processed_data)} records processed")
        
        return processed_data
    
    def stage_2_feature_engineering(self, df=None):
        """
        Stage 2: Feature Engineering
        Create temporal and behavioral features
        
        Args:
            df: Preprocessed dataframe (loads from disk if None)
        """
        logger.info(utils.generate_report_header("STAGE 2: FEATURE ENGINEERING"))
        
        if df is None:
            try:
                df = utils.load_dataframe('processed_unified_logs.csv')
            except:
                logger.error("No preprocessed data found. Run stage 1 first.")
                return None
        
        self.feature_engineer = feature_engineering.FeatureEngineer()
        
        # Create aggregated features
        features_df = self.feature_engineer.run_full_pipeline(df, time_window='1D')
        
        # Create sequences for LSTM
        sequences, labels, user_ids = self.feature_engineer.create_sequence_features(df)
        
        logger.info(f"✓ Stage 2 completed: {len(features_df)} feature vectors, {len(sequences)} sequences")
        
        return features_df, sequences, labels
    
    def stage_3_model_training(self):
        """
        Stage 3: Model Training
        Train all three models: Isolation Forest, LSTM Autoencoder, Deep Clustering
        """
        logger.info(utils.generate_report_header("STAGE 3: MODEL TRAINING"))
        
        # Train Isolation Forest
        logger.info("\n>>> Training Isolation Forest...")
        try:
            if_detector, if_metrics = isolation_forest_model.main()
            self.models['isolation_forest'] = if_detector
            self.results['isolation_forest'] = if_metrics
            logger.info("✓ Isolation Forest training completed")
        except Exception as e:
            logger.error(f"✗ Isolation Forest training failed: {e}")
        
        # Train LSTM Autoencoder
        logger.info("\n>>> Training LSTM Autoencoder...")
        try:
            lstm_model, lstm_metrics = lstm_autoencoder_model.main()
            self.models['lstm_autoencoder'] = lstm_model
            self.results['lstm_autoencoder'] = lstm_metrics
            logger.info("✓ LSTM Autoencoder training completed")
        except Exception as e:
            logger.error(f"✗ LSTM Autoencoder training failed: {e}")
        
        # Train Deep Clustering
        logger.info("\n>>> Training Deep Clustering...")
        try:
            dc_model, dc_metrics = deep_clustering_model.main()
            self.models['deep_clustering'] = dc_model
            self.results['deep_clustering'] = dc_metrics
            logger.info("✓ Deep Clustering training completed")
        except Exception as e:
            logger.error(f"✗ Deep Clustering training failed: {e}")
        
        logger.info(f"\n✓ Stage 3 completed: {len(self.models)} models trained")
    
    def stage_4_model_evaluation(self):
        """
        Stage 4: Model Evaluation
        Compare and evaluate all models
        """
        logger.info(utils.generate_report_header("STAGE 4: MODEL EVALUATION"))
        
        evaluator = model_evaluation.ModelEvaluator()
        evaluator.run_full_evaluation()
        
        logger.info("✓ Stage 4 completed: Comprehensive evaluation report generated")
    
    def stage_5_ensemble_integration(self):
        """
        Stage 5: Ensemble Integration
        Combine models and generate final alerts
        """
        logger.info(utils.generate_report_header("STAGE 5: ENSEMBLE INTEGRATION"))
        
        # Test all ensemble methods
        methods = ['weighted', 'majority', 'cascade']
        best_method = None
        best_f1 = 0
        
        for method in methods:
            try:
                ensemble = ensemble_system.EnsembleDetector(method=method)
                results = ensemble.run_ensemble_pipeline()
                
                if results['metrics']['f1_score'] > best_f1:
                    best_f1 = results['metrics']['f1_score']
                    best_method = method
            except Exception as e:
                logger.error(f"Error with {method} ensemble: {e}")
        
        logger.info(f"\n✓ Stage 5 completed: Best ensemble method = {best_method} (F1={best_f1:.4f})")
        
        return best_method
    
    def stage_6_visualization(self):
        """
        Stage 6: Results Visualization
        Generate comprehensive visualizations and dashboards
        """
        logger.info(utils.generate_report_header("STAGE 6: VISUALIZATION"))
        
        dashboard = visualization.VisualizationDashboard()
        dashboard.generate_all_visualizations()
        
        logger.info("✓ Stage 6 completed: All visualizations generated")
    
    def run_full_pipeline(self, insider_list=None, skip_stages=None):
        """
        Run the complete pipeline from start to finish
        
        Args:
            insider_list: Optional list of known insider users for labeling
            skip_stages: List of stage numbers to skip (e.g., [1, 2] to skip preprocessing and feature engineering)
        """
        if skip_stages is None:
            skip_stages = []
        
        logger.info("\n" + "="*80)
        logger.info("STARTING FULL PIPELINE EXECUTION")
        logger.info("="*80 + "\n")
        
        try:
            # Stage 1: Data Preprocessing
            if 1 not in skip_stages:
                processed_data = self.stage_1_data_preprocessing(insider_list)
            else:
                logger.info("Skipping Stage 1: Data Preprocessing")
                processed_data = None
            
            # Stage 2: Feature Engineering
            if 2 not in skip_stages:
                features_df, sequences, labels = self.stage_2_feature_engineering(processed_data)
            else:
                logger.info("Skipping Stage 2: Feature Engineering")
            
            # Stage 3: Model Training
            if 3 not in skip_stages:
                self.stage_3_model_training()
            else:
                logger.info("Skipping Stage 3: Model Training")
            
            # Stage 4: Model Evaluation
            if 4 not in skip_stages:
                self.stage_4_model_evaluation()
            else:
                logger.info("Skipping Stage 4: Model Evaluation")
            
            # Stage 5: Ensemble Integration
            if 5 not in skip_stages:
                best_ensemble = self.stage_5_ensemble_integration()
            else:
                logger.info("Skipping Stage 5: Ensemble Integration")
            
            # Stage 6: Visualization
            if 6 not in skip_stages:
                self.stage_6_visualization()
            else:
                logger.info("Skipping Stage 6: Visualization")
            
            # Final Summary
            self.print_final_summary()
            
            logger.info("\n" + "="*80)
            logger.info("✓ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            logger.info("="*80 + "\n")
            
        except Exception as e:
            logger.error(f"\n✗ PIPELINE EXECUTION FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    def print_final_summary(self):
        """Print final summary of results"""
        logger.info("\n" + "="*80)
        logger.info("FINAL SUMMARY")
        logger.info("="*80)
        
        # Model performance
        logger.info("\nModel Performance:")
        for model_name, metrics in self.results.items():
            logger.info(f"\n{model_name.replace('_', ' ').title()}:")
            logger.info(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
            logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
            logger.info(f"  Recall:    {metrics.get('recall', 0):.4f}")
            logger.info(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}")
            if 'auc_roc' in metrics:
                logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        # Output files
        logger.info("\nGenerated Output Files:")
        logger.info(f"  - Processed Data: {config.PROCESSED_DATA_DIR}")
        logger.info(f"  - Trained Models: {config.MODELS_DIR}")
        logger.info(f"  - Results & Metrics: {config.RESULTS_DIR}")
        logger.info(f"  - Visualizations: {config.RESULT_PATHS['visualizations']}")
        logger.info(f"  - Alerts: {config.RESULT_PATHS['alerts']}")
        logger.info(f"  - Logs: {config.LOGGING['file']}")
        
        logger.info("\n" + "="*80 + "\n")


def main():
    """Main entry point with command-line argument support"""
    parser = argparse.ArgumentParser(
        description='Insider Threat Detection System - Main Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --full
  
  # Run only specific stages
  python main.py --stages 1 2 3
  
  # Skip certain stages (e.g., if data already preprocessed)
  python main.py --full --skip 1 2
  
  # Run individual stages
  python main.py --preprocess
  python main.py --train
  python main.py --evaluate
        """
    )
    
    # Pipeline options
    parser.add_argument('--full', action='store_true', help='Run complete pipeline')
    parser.add_argument('--stages', nargs='+', type=int, choices=range(1, 7),
                       help='Run specific stages (1-6)')
    parser.add_argument('--skip', nargs='+', type=int, choices=range(1, 7),
                       help='Skip specific stages')
    
    # Individual stage options
    parser.add_argument('--preprocess', action='store_true', help='Run only preprocessing')
    parser.add_argument('--feature-eng', action='store_true', help='Run only feature engineering')
    parser.add_argument('--train', action='store_true', help='Run only model training')
    parser.add_argument('--evaluate', action='store_true', help='Run only evaluation')
    parser.add_argument('--ensemble', action='store_true', help='Run only ensemble')
    parser.add_argument('--visualize', action='store_true', help='Run only visualization')
    
    # Other options
    parser.add_argument('--insider-list', type=str, help='Path to file with insider user IDs')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = InsiderThreatDetectionPipeline()
    
    # Load insider list if provided
    insider_list = None
    if args.insider_list:
        try:
            with open(args.insider_list, 'r') as f:
                insider_list = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(insider_list)} insider user IDs")
        except Exception as e:
            logger.error(f"Could not load insider list: {e}")
    
    # Run based on arguments
    if args.full:
        pipeline.run_full_pipeline(insider_list=insider_list, skip_stages=args.skip)
    
    elif args.stages:
        skip_stages = [i for i in range(1, 7) if i not in args.stages]
        pipeline.run_full_pipeline(insider_list=insider_list, skip_stages=skip_stages)
    
    elif args.preprocess:
        pipeline.stage_1_data_preprocessing(insider_list)
    
    elif args.feature_eng:
        pipeline.stage_2_feature_engineering()
    
    elif args.train:
        pipeline.stage_3_model_training()
    
    elif args.evaluate:
        pipeline.stage_4_model_evaluation()
    
    elif args.ensemble:
        pipeline.stage_5_ensemble_integration()
    
    elif args.visualize:
        pipeline.stage_6_visualization()
    
    else:
        # No arguments provided, run full pipeline
        logger.info("No arguments provided. Running full pipeline...")
        pipeline.run_full_pipeline(insider_list=insider_list)


if __name__ == "__main__":
    main()