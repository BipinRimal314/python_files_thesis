"""
Feature Importance Analysis using SHAP (SHapley Additive exPlanations)

This script analyzes the trained V2 Isolation Forest model to understand
WHICH features are driving the anomaly detection.

It answers the theoretical question: "What specific behaviors make an insider
look anomalous compared to their peers?"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path

# Use the V2 config
import config_v2 as config 
import utils

logger = utils.logger

class Feature_explainer:
    
    def __init__(self):
        self.model_path = config.MODEL_PATHS['isolation_forest_v2']
        
        # --- Dynamically get the file names ---
        subset = config.DATASET_SUBSET if hasattr(config, 'DATASET_SUBSET') and config.DATASET_SUBSET else []
        subset_name = "_".join(subset) if subset else "ALL"
        
        self.data_path = config.PROCESSED_DATA_DIR / f'engineered_static_features_{subset_name}_v2.csv'
        self.viz_dir = config.RESULT_PATHS['visualizations']
        
        # Ensure viz directory exists
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def run_analysis(self):
        logger.info(utils.generate_report_header("FEATURE IMPORTANCE ANALYSIS (SHAP)"))
        
        # 1. Load Model
        logger.info(f"Loading Isolation Forest model from {self.model_path.name}...")
        try:
            model = joblib.load(self.model_path)
        except FileNotFoundError:
            logger.error("Model file not found. Please train the V2 Isolation Forest first.")
            return

        # 2. Load Data
        logger.info(f"Loading feature data from {self.data_path.name}...")
        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            logger.error("Data file not found. Please run feature_engineering_v2.py first.")
            return

        # Prepare X (features) and identify the insider
        user_col = 'user_id'
        label_col = 'is_insider'
        feature_cols = [col for col in df.columns if col not in [user_col, label_col]]
        
        X = df[feature_cols]
        insiders = df[df[label_col] == 1]
        
        logger.info(f"Analyzing {len(X)} users with {len(feature_cols)} features.")
        logger.info(f"Found {len(insiders)} insider(s) for detailed diagnosis.")

        # 3. Create SHAP Explainer
        # Isolation Forest is a tree-based model, so we use TreeExplainer.
        # Note: TreeExplainer for IF can be slow on large datasets, so we sample if needed.
        logger.info("Initializing SHAP TreeExplainer (this may take a moment)...")
        
        # Use a background sample for speed if dataset is huge (optional, here we use full X since it's small ~5000 rows)
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        logger.info("Calculating SHAP values...")
        shap_values = explainer.shap_values(X)

        # 4. Generate Summary Plot (Global Importance)
        logger.info("Generating Summary Plot (Global Feature Importance)...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, show=False, plot_type="bar")
        plt.title("Global Feature Importance (Isolation Forest V2)")
        plt.tight_layout()
        
        save_path = self.viz_dir / "shap_summary_bar.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved summary plot to {save_path}")

        # 5. Generate Beeswarm Plot (Detailed Importance)
        # This shows not just magnitude, but direction (e.g., "Does high HTTP z-score make you an anomaly?")
        logger.info("Generating Beeswarm Plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.title("Feature Impact on Anomaly Score")
        plt.tight_layout()
        
        save_path = self.viz_dir / "shap_summary_beeswarm.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved beeswarm plot to {save_path}")

        # 6. Insider Diagnosis (Local Importance)
        # We want to see WHY the insider was flagged.
        if len(insiders) > 0:
            logger.info("Generating diagnosis plots for insiders...")
            
            # Find the index of the first insider
            insider_idx = insiders.index[0]
            insider_user_id = df.iloc[insider_idx][user_col]
            
            logger.info(f"Diagnosing Insider User: {insider_user_id}")
            
            # Force plot (Visualizing the push and pull of features)
            # Note: matplotlib=True is needed to save it as an image
            plt.figure(figsize=(20, 3))
            shap.force_plot(
                explainer.expected_value, 
                shap_values[insider_idx, :], 
                X.iloc[insider_idx, :], 
                matplotlib=True,
                show=False
            )
            
            save_path = self.viz_dir / f"shap_force_plot_{insider_user_id}.png"
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Saved force plot for {insider_user_id} to {save_path}")

            # Waterfall plot (Even clearer breakdown)
            plt.figure(figsize=(10, 6))
            # create a Explanation object for the waterfall plot
            shap_obj = shap.Explanation(
                values=shap_values[insider_idx], 
                base_values=explainer.expected_value, 
                data=X.iloc[insider_idx], 
                feature_names=feature_cols
            )
            shap.waterfall_plot(shap_obj, show=False, max_display=10)
            plt.title(f"Why is {insider_user_id} anomalous?")
            plt.tight_layout()
            
            save_path = self.viz_dir / f"shap_waterfall_{insider_user_id}.png"
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Saved waterfall plot to {save_path}")

        logger.info("="*80)
        logger.info("SHAP Analysis Complete.")
        logger.info("The visualizations in 'results/visualizations' explain the theory behind your model.")
        logger.info("="*80)

if __name__ == "__main__":
    analyzer = Feature_explainer()
    analyzer.run_analysis()