"""
Main Execution Pipeline for Insider Threat Detection System
Orchestrates the complete workflow from data preprocessing to final reporting
"""

# CRITICAL: Must be set BEFORE any TensorFlow imports (including via other modules)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import argparse
from pathlib import Path

# Disable GPU before TensorFlow loads
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import config
import utils

# Import modules
import data_preprocessing_polars as data_preprocessing
import feature_engineering_polars as feature_engineering
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
        self.models = {}
        self.results = {}
        
        logger.info("="*80)
        logger.info("INSIDER THREAT DETECTION SYSTEM")
        logger.info("Unsupervised Behavioral Profiling Using Time-Series and Anomaly Detection")
        logger.info("="*80)
        
        # Ensure base directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all necessary directories from config exist"""
        dirs = [
            config.DATA_DIR, 
            config.PROCESSED_DATA_DIR, 
            config.MODELS_DIR, 
            config.RESULTS_DIR,
            config.PLOTS_DIR
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _check_file_exists(self, filepath, stage_name):
        """Helper to check if a required file exists before running a stage"""
        if not os.path.exists(filepath):
            logger.error(f"Missing required file for {stage_name}: {filepath}")
            logger.error(f"Please run the previous stages first to generate this file.")
            return False
        return True
    
    def stage_1_data_preprocessing(self, insider_list=None):
        """
        Stage 1: Data Preprocessing
        """
        logger.info(utils.generate_report_header("STAGE 1: DATA PREPROCESSING"))
        
        try:
            # Initialize Polars preprocessor
            preprocessor = data_preprocessing.DataPreprocessorPolars()
            
            # Run pipeline
            processed_data = preprocessor.run_pipeline()
            
            logger.info(f"✓ Stage 1 completed successfully.")
            return processed_data
            
        except Exception as e:
            logger.error(f"✗ Stage 1 failed: {e}")
            raise e
    
    def stage_2_feature_engineering(self):
        """
        Stage 2: Feature Engineering
        """
        logger.info(utils.generate_report_header("STAGE 2: FEATURE ENGINEERING"))
        
        # Check if Stage 1 output exists
        if not self._check_file_exists(config.PROCESSED_DATA_FILE, "Stage 2"):
            logger.error(f"Expected input file: {config.PROCESSED_DATA_FILE}")
            return

        try:
            feature_engineer = feature_engineering.FeatureEngineerPolars()
            
            # Run full pipeline
            feature_engineer.run_pipeline()
            
            logger.info(f"✓ Stage 2 completed: Feature files generated.")
            
        except Exception as e:
            logger.error(f"✗ Stage 2 failed: {e}")
            raise e
    
    def stage_3_model_training(self):
        """
        Stage 3: Model Training
        """
        logger.info(utils.generate_report_header("STAGE 3: MODEL TRAINING"))
        
        # Verify feature files exist
        required_files = [
            (config.DAILY_FEATURES_FILE, "Isolation Forest"),
            (config.SEQUENCE_DATA_FILE, "Deep Learning Models") 
        ]
        
        for file_path, model_name in required_files:
            if not self._check_file_exists(file_path, model_name):
                return

        # 1. Train Isolation Forest
        logger.info("\n>>> Training Isolation Forest...")
        try:
            if_detector, if_metrics = isolation_forest_model.main()
            self.models['isolation_forest'] = if_detector
            self.results['isolation_forest'] = if_metrics
            logger.info("✓ Isolation Forest training completed")
        except Exception as e:
            logger.error(f"✗ Isolation Forest training failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. Train LSTM Autoencoder
        logger.info("\n>>> Training LSTM Autoencoder...")
        try:
            lstm_model, lstm_metrics = lstm_autoencoder_model.main()
            self.models['lstm_autoencoder'] = lstm_model
            self.results['lstm_autoencoder'] = lstm_metrics
            logger.info("✓ LSTM Autoencoder training completed")
        except Exception as e:
            logger.error(f"✗ LSTM Autoencoder training failed: {e}")
        
        # 3. Train Deep Clustering
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
        """
        logger.info(utils.generate_report_header("STAGE 4: MODEL EVALUATION"))
        
        try:
            evaluator = model_evaluation.ModelEvaluator()
            evaluator.run_full_evaluation()
            logger.info("✓ Stage 4 completed: Comprehensive evaluation report generated")
        except Exception as e:
            logger.error(f"✗ Stage 4 failed: {e}")
    
    def stage_5_ensemble_integration(self):
        """
        Stage 5: Ensemble Integration
        """
        logger.info(utils.generate_report_header("STAGE 5: ENSEMBLE INTEGRATION"))
        
        try:
            methods = ['weighted', 'majority', 'cascade']
            best_method = None
            best_f1 = 0
            
            for method in methods:
                logger.info(f"Testing ensemble method: {method}")
                ensemble = ensemble_system.EnsembleDetector(method=method)
                results = ensemble.run_ensemble_pipeline()
                
                if results and 'metrics' in results and results['metrics']:
                    current_f1 = results['metrics'].get('f1_score', 0)
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        best_method = method
                else:
                    best_method = "unsupervised_ensemble"
            
            logger.info(f"\n✓ Stage 5 completed. Best performing method: {best_method}")
            return best_method
        except KeyError as e:
             logger.error(f"✗ Stage 5 failed (KeyError): {e} - likely missing columns due to pipeline issues.")
        except Exception as e:
            logger.error(f"✗ Stage 5 failed: {e}")

    def stage_6_visualization(self):
        """
        Stage 6: Results Visualization
        """
        logger.info(utils.generate_report_header("STAGE 6: VISUALIZATION"))
        
        try:
            dashboard = visualization.VisualizationDashboard()
            dashboard.generate_all_visualizations()
            logger.info("✓ Stage 6 completed: All visualizations generated")
        except Exception as e:
            logger.error(f"✗ Stage 6 failed: {e}")

    def run_full_pipeline(self, insider_list=None, skip_stages=None):
        """
        Run the complete pipeline from start to finish
        """
        if skip_stages is None:
            skip_stages = []
        
        logger.info("\n" + "="*80)
        logger.info("STARTING FULL PIPELINE EXECUTION")
        logger.info("="*80 + "\n")
        
        try:
            # Stage 1: Data Preprocessing
            if 1 not in skip_stages:
                self.stage_1_data_preprocessing(insider_list)
            else:
                logger.info("Skipping Stage 1: Data Preprocessing")
            
            # Stage 2: Feature Engineering
            if 2 not in skip_stages:
                self.stage_2_feature_engineering()
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
                self.stage_5_ensemble_integration()
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
            logger.info("✓ PIPELINE EXECUTION COMPLETED")
            logger.info("="*80 + "\n")
            
        except KeyboardInterrupt:
            logger.warning("\nPipeline execution interrupted by user.")
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
        if self.results:
            logger.info("\nModel Performance (on labeled validation set):")
            for model_name, result_dict in self.results.items():
                logger.info(f"\n{model_name.replace('_', ' ').title()}:")
                
                if result_dict is None or 'metrics' not in result_dict:
                    logger.warning("  No metrics available (unsupervised execution).")
                    continue
                
                metrics = result_dict['metrics']
                logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
                logger.info(f"  Recall:    {metrics.get('recall', 0):.4f}")
                logger.info(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}")
        else:
            logger.info("\nNo model training results to display.")
        
        # Output files
        logger.info("\nGenerated Output Locations:")
        logger.info(f"  - Processed Data: {config.PROCESSED_DATA_DIR}")
        logger.info(f"  - Trained Models: {config.MODELS_DIR}")
        logger.info(f"  - Alert Logs:     {config.RESULT_PATHS.get('alerts', 'See results dir')}")
        logger.info(f"  - Logs:           {config.LOGGING.get('file', 'logs/')}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Insider Threat Detection System - Main Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Pipeline options
    parser.add_argument('--full', action='store_true', help='Run complete pipeline')
    parser.add_argument('--stages', nargs='+', type=int, choices=range(1, 7), help='Run specific stages')
    parser.add_argument('--skip', nargs='+', type=int, choices=range(1, 7), help='Skip specific stages')
    
    # Quick flags
    parser.add_argument('--preprocess', action='store_true', help='Run Stage 1')
    parser.add_argument('--feature-eng', action='store_true', help='Run Stage 2')
    parser.add_argument('--train', action='store_true', help='Run Stage 3')
    parser.add_argument('--evaluate', action='store_true', help='Run Stage 4')
    
    # Data options
    parser.add_argument('--insider-list', type=str, help='Path to file with insider user IDs (for ground truth)')
    
    args = parser.parse_args()
    
    # Initialize
    pipeline = InsiderThreatDetectionPipeline()
    
    # Load insider list if provided
    insider_list = None
    if args.insider_list and os.path.exists(args.insider_list):
        try:
            with open(args.insider_list, 'r') as f:
                insider_list = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(insider_list)} insider user IDs for evaluation labeling.")
        except Exception as e:
            logger.error(f"Could not load insider list: {e}")
    
    # Execution Logic
    if args.stages:
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
    elif args.full:
        pipeline.run_full_pipeline(insider_list=insider_list)
    else:
        # Default behavior if no args provided
        logger.info("No specific arguments provided. Running full pipeline...")
        pipeline.run_full_pipeline(insider_list=insider_list)

if __name__ == "__main__":
    main()