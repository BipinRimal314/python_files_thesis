"""
Feature Engineering Module using Polars
Optimized for performance and memory efficiency using Lazy API and Streaming
"""

import polars as pl
import numpy as np
import os
import joblib  # Using joblib for robust serialization
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import config
import utils

logger = utils.logger

class FeatureEngineerPolars:
    """
    Handles feature engineering using Polars for high performance.
    Generates:
    1. Daily aggregated features (for Isolation Forest)
    2. Static features and Scaler (for user profiling)
    3. Sequential features and Labels (for LSTM/Deep Clustering)
    """
    
    def __init__(self):
        self.input_file = config.PROCESSED_DATA_FILE
        
        # Outputs
        self.daily_output = config.DAILY_FEATURES_FILE
        self.sequence_output = config.SEQUENCE_DATA_FILE # Parquet version
        self.sequence_npy_output = config.PROCESSED_DATA_DIR / 'sequences.npy' # Numpy version for LSTM
        self.static_output = config.PROCESSED_DATA_DIR / 'static_features.csv'
        self.sequence_labels_output = config.SEQUENCE_LABELS_FILE
        
        # Scaler path
        self.scaler_output = config.MODEL_PATHS['static_scaler']
        
        # Define aggregations for daily profile
        self.daily_aggs = [
            pl.count('activity').alias('daily_activity_count'),
            pl.n_unique('pc_id').alias('unique_pcs'),
            pl.n_unique('ip').alias('unique_ips') if 'ip' in config.LOG_FILENAMES else pl.lit(0).alias('unique_ips'),
            
            # Time based features
            pl.col('date').dt.hour().min().alias('first_activity_hour'),
            pl.col('date').dt.hour().max().alias('last_activity_hour'),
            (pl.col('date').dt.hour().max() - pl.col('date').dt.hour().min()).alias('work_duration_hours'),
            
            # Activity type counts
            pl.col('activity').filter(pl.col('activity') == 'Logon').count().alias('logon_count'),
            pl.col('activity').filter(pl.col('activity') == 'Logoff').count().alias('logoff_count'),
            pl.col('activity').filter(pl.col('activity') == 'File Open').count().alias('file_access_count'),
            pl.col('activity').filter(pl.col('activity') == 'Email').count().alias('email_count'),
            pl.col('activity').filter(pl.col('activity') == 'Connect').count().alias('usb_connect_count'),
            pl.col('activity').filter(pl.col('activity') == 'http').count().alias('web_visit_count'),
            
            # After hours activity
            pl.col('date').filter(
                (pl.col('date').dt.hour() < 7) | (pl.col('date').dt.hour() > 19)
            ).count().alias('after_hours_activity')
        ]
        
        self.include_label = True

    def run_pipeline(self):
        """Run the full feature engineering pipeline"""
        logger.info("Starting Feature Engineering Pipeline (Polars Lazy Execution)...")
        
        if not os.path.exists(self.input_file):
            logger.error(f"Input file not found: {self.input_file}")
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
            
        # 1. Load Data (Lazy Scan)
        lf = self._scan_data()
        
        # 2. Daily Features (Lazy Plan)
        logger.info("Constructing execution graph for daily features...")
        daily_features_lf = self._create_daily_features_lazy(lf)
        
        # EXECUTE: Collect with Streaming to avoid OOM
        logger.info("Executing aggregation (Streaming Mode)...")
        daily_features = daily_features_lf.collect(streaming=True)
        
        # SAVE 1: Save daily features for Isolation Forest
        self._save_parquet(daily_features, self.daily_output)
        
        # 3. Static Features & Scaler (Summary per user)
        logger.info("Creating static features and fitting scaler...")
        static_features = self._create_static_features(daily_features)
        self._save_csv(static_features, self.static_output)
        
        # Fit and Save Scaler (Using joblib)
        self._fit_and_save_scaler(static_features)
        
        # 4. Sequence Features (For Deep Learning)
        logger.info("Creating sequences...")
        
        # Save Parquet version
        self._save_parquet(daily_features, self.sequence_output)
        
        # Create and Save Numpy version (Sliding Windows) + Labels for LSTM
        self._create_and_save_sequences(daily_features)
        
        logger.info("Feature engineering pipeline completed successfully.")
        return daily_features

    def _scan_data(self):
        """Lazy load processed logs"""
        logger.info(f"Scanning data from {self.input_file}")
        if str(self.input_file).endswith('.parquet'):
            return pl.scan_parquet(self.input_file)
        else:
            return pl.scan_csv(self.input_file, try_parse_dates=True)

    def _create_daily_features_lazy(self, lf):
        """Create daily behavioral profiles (LazyFrame)"""
        
        # Robust Date Casting inside LazyFrame
        lf = lf.with_columns(
            pl.col('date').cast(pl.Datetime, strict=False)
        )

        # Drop rows where date parsing failed
        lf = lf.filter(pl.col('date').is_not_null())
        
        # Check if 'is_insider' exists in schema to aggregate labels
        schema_cols = lf.collect_schema().names()
        aggs = self.daily_aggs.copy()
        
        if 'is_insider' in schema_cols:
            aggs.append(pl.col('is_insider').max().alias('is_anomaly'))
        
        # Group by User and Day
        daily_lf = lf.group_by(
            [pl.col('user_id').alias('user'), pl.col('date').dt.date().alias('day')]
        ).agg(aggs)
        
        daily_lf = daily_lf.fill_null(0)
        daily_lf = daily_lf.sort(['user', 'day'])
        
        return daily_lf

    def _create_static_features(self, daily_df):
        """Create static profiles"""
        exclude = ['user', 'day', 'is_anomaly']
        numeric_cols = [c for c in daily_df.columns if c not in exclude]
        
        aggs = []
        for col in numeric_cols:
            aggs.append(pl.col(col).mean().alias(f"{col}_mean"))
            aggs.append(pl.col(col).std().alias(f"{col}_std"))
            
        static_df = daily_df.group_by('user').agg(aggs).fill_null(0)
        logger.info(f"Generated static features for {len(static_df)} users")
        return static_df

    def _fit_and_save_scaler(self, static_df):
        """Fit StandardScaler on static features and save using joblib"""
        try:
            # Convert to pandas for sklearn compatibility
            df_pandas = static_df.to_pandas()
            
            # Drop non-feature columns
            cols_to_drop = ['user', 'day']
            features = df_pandas.drop(columns=[c for c in cols_to_drop if c in df_pandas.columns])
            
            # Fit Scaler
            scaler = StandardScaler()
            scaler.fit(features)
            
            # Save Scaler using joblib
            Path(self.scaler_output).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, self.scaler_output)
            logger.info(f"Saved static scaler to {self.scaler_output}")
            
        except Exception as e:
            logger.error(f"Failed to fit/save scaler: {e}")
            raise e

    def _create_and_save_sequences(self, daily_df):
        """
        Create 3D numpy arrays (samples, timesteps, features) and labels.
        """
        sequence_length = config.SEQUENCE_LENGTH
        
        # Identify feature columns
        exclude = ['user', 'day', 'is_anomaly']
        feature_cols = [c for c in daily_df.columns if c not in exclude]
        
        # Determine label availability
        has_labels = 'is_anomaly' in daily_df.columns
        
        logger.info(f"Processing sequences with length {sequence_length}...")
        
        pdf = daily_df.to_pandas()
        grouped = pdf.groupby('user')
        
        sequences = []
        labels = []
        
        for user, group in grouped:
            data = group[feature_cols].values
            
            # Get labels if available
            group_labels = group['is_anomaly'].values if has_labels else None
            
            if len(data) < sequence_length:
                continue
                
            # Create sliding windows
            for i in range(0, len(data) - sequence_length + 1):
                window = data[i : i + sequence_length]
                sequences.append(window)
                
                # Label is 1 if the LAST day in sequence is anomalous
                if has_labels:
                    labels.append(group_labels[i + sequence_length - 1])
                else:
                    labels.append(0)
        
        if sequences:
            X = np.array(sequences)
            y = np.array(labels)
            
            logger.info(f"Generated sequences shape: {X.shape}")
            logger.info(f"Generated labels shape: {y.shape}")
            
            # Save sequences
            np.save(self.sequence_npy_output, X)
            logger.info(f"Saved sequences to {self.sequence_npy_output}")
            
            # Save labels
            np.save(self.sequence_labels_output, y)
            logger.info(f"Saved sequence labels to {self.sequence_labels_output}")
            
        else:
            logger.warning("No sequences generated")
            np.save(self.sequence_npy_output, np.zeros((1, sequence_length, len(feature_cols))))
            np.save(self.sequence_labels_output, np.zeros((1,)))

    def _save_parquet(self, df, filepath):
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(filepath)
            logger.info(f"Saved features to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")
            raise e

    def _save_csv(self, df, filepath):
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            df.write_csv(filepath)
            logger.info(f"Saved features to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")
            raise e

if __name__ == "__main__":
    import sys
    if not logger.handlers:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        print("Running Feature Engineering Polars...")
        fe = FeatureEngineerPolars()
        fe.run_pipeline()
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)