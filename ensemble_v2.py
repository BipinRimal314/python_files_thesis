"""
V2 Ensemble Model

This script combines the anomaly scores from the three trained V2 models
(Isolation Forest, Deep Clustering, LSTM Autoencoder) to create a
single, more robust ensemble score.

It aggregates the LSTM's sequence-level scores to the user-level
and then performs a weighted average.

This script *must* be run *after* all three model_v2.py scripts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Use the V2 config
import config_v2 as config 
import utils

logger = utils.logger

class EnsembleModelV2:
    
    def __init__(self):
        self.results_dir = config.RESULTS_DIR
        
        # --- Dynamically build the filenames ---
        subset = config.DATASET_SUBSET if hasattr(config, 'DATASET_SUBSET') and config.DATASET_SUBSET else []
        self.subset_name = "_".join(subset) if subset else "ALL"
        
        # Input files (predictions from other models)
        self.if_preds_path = self.results_dir / f"isolation_forest_predictions_{self.subset_name}_v2.csv"
        self.dc_preds_path = self.results_dir / f"deep_clustering_predictions_{self.subset_name}_v2.csv"
        self.lstm_preds_path = self.results_dir / f"lstm_autoencoder_predictions_{self.subset_name}_v2.csv"
        
        # Output file
        self.output_preds_path = self.results_dir / f"ensemble_predictions_{self.subset_name}_v2.csv"
        
        # Define the weights for each model.
        # IF and LSTM had the best AUCs, so we weight them higher.
        self.weights = {
            'if': 0.4,  # Isolation Forest (AUC: 0.91)
            'lstm': 0.4, # LSTM (AUC: 0.88)
            'dc': 0.2   # Deep Clustering (AUC: 0.71)
        }
        
    def load_predictions(self):
        """
        Loads the prediction files from all three models.
        """
        logger.info("Loading V2 predictions from all models...")
        try:
            self.if_preds = pd.read_csv(self.if_preds_path, usecols=['user_id', 'true_label', 'anomaly_score'])
            self.dc_preds = pd.read_csv(self.dc_preds_path, usecols=['user_id', 'anomaly_score'])
            self.lstm_preds = pd.read_csv(self.lstm_preds_path, usecols=['user_id', 'anomaly_score'])
            
            logger.info(f"Loaded {len(self.if_preds)} IF preds, {len(self.dc_preds)} DC preds, {len(self.lstm_preds)} LSTM preds")
            
        except FileNotFoundError as e:
            logger.error(f"Failed to load prediction file: {e}")
            logger.error("Please run all three V2 model scripts first.")
            return False
        return True

    def merge_predictions(self):
        """
        Aggregates LSTM scores to the user level and merges all scores.
        """
        logger.info("Merging model predictions...")
        
        # 1. Aggregate LSTM scores: A user's score is their *max* sequence score.
        logger.info("Aggregating LSTM scores to user-level (this may take a moment)...")
        lstm_user_scores = self.lstm_preds.groupby('user_id')['anomaly_score'].max().reset_index()
        
        # 2. Merge static models (IF and DC)
        # We use the IF predictions as the base, since it has the true_label
        merged_df = self.if_preds.merge(
            self.dc_preds, 
            on='user_id', 
            how='left', 
            suffixes=('_if', '_dc')
        )
        
        # 3. Merge aggregated LSTM scores
        merged_df = merged_df.merge(
            lstm_user_scores, 
            on='user_id', 
            how='left'
        )
        
        # Rename columns for clarity
        merged_df = merged_df.rename(columns={
            'anomaly_score_if': 'score_if',
            'anomaly_score_dc': 'score_dc',
            'anomaly_score': 'score_lstm'
        })
        
        # 4. Handle NaNs
        # If a user had no sequences (the "8-insider" bug), their LSTM score is NaN.
        # We will fill this with 0 (a "normal" score).
        merged_df['score_lstm'] = merged_df['score_lstm'].fillna(0)
        
        logger.info(f"Successfully merged scores for {len(merged_df)} users.")
        return merged_df

    def calculate_ensemble_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes all scores and calculates the final weighted ensemble score.
        """
        logger.info("Calculating weighted ensemble score...")
        
        # 1. Normalize all scores to a 0-1 range
        scaler = MinMaxScaler()
        df['score_if_norm'] = scaler.fit_transform(df[['score_if']])
        df['score_dc_norm'] = scaler.fit_transform(df[['score_dc']])
        df['score_lstm_norm'] = scaler.fit_transform(df[['score_lstm']])
        
        # 2. Calculate weighted average
        df['ensemble_score'] = (
            (df['score_if_norm'] * self.weights['if']) +
            (df['score_dc_norm'] * self.weights['dc']) +
            (df['score_lstm_norm'] * self.weights['lstm'])
        )
        
        # 3. Create the final output dataframe
        # We rename `ensemble_score` to `anomaly_score` so the
        # diagnose script can read it automatically.
        final_df = df[['user_id', 'true_label', 'ensemble_score']].rename(
            columns={'ensemble_score': 'anomaly_score'}
        )
        
        return final_df

    def save_predictions(self, df: pd.DataFrame):
        """
        Saves the final ensemble predictions to a CSV.
        """
        df.to_csv(self.output_preds_path, index=False)
        logger.info(f"Ensemble predictions saved to: {self.output_preds_path.name}")

    def run_pipeline(self):
        """Runs the full ensemble pipeline."""
        logger.info(utils.generate_report_header("ENSEMBLE MODEL PIPELINE (V2)"))
        
        if not self.load_predictions():
            return
            
        merged_df = self.merge_predictions()
        final_preds = self.calculate_ensemble_score(merged_df)
        self.save_predictions(final_preds)
        
        logger.info("="*80)
        logger.info("Ensemble V2 model complete!")
        logger.info("Next step: python diagnose_results_v2.py")
        logger.info("="*80)

def main():
    ensemble = EnsembleModelV2()
    ensemble.run_pipeline()

if __name__ == "__main__":
    main()