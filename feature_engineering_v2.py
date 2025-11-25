"""
Feature Engineering Pipeline (V2: Multi-Dataset)

This script builds on the labeled '...LABELED.csv' file.
It is a multi-pass, memory-safe pipeline designed to create
advanced, relative features (Z-scores) to improve model precision.

[--- V2.4 FIX ---]
- Fixes AttributeError by correctly referencing the config variable as
  `config.PERFORMANCE['FORCE_RERUN_PASS1']`.
- Fixes AttributeError by importing StandardScaler directly and
  replacing the non-existent `utils.get_scaler()` calls.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil
from sklearn.preprocessing import StandardScaler # Import scaler

# Use the new V2 config
import config_v2 as config 
import utils

logger = utils.logger

class FeatureEngineerV2:
    
    def __init__(self, chunk_size=2000000):
        # --- (V2.1 CHECKPOINT) ---
        # Get the subset name from config
        subset = config.DATASET_SUBSET if hasattr(config, 'DATASET_SUBSET') and config.DATASET_SUBSET else []
        subset_name = "_".join(subset) if subset else "ALL"
        
        # Point to the LABELED file as input
        self.raw_log_path = config.PROCESSED_DATA_DIR / f'processed_unified_logs_{subset_name}_LABELED.csv'
        
        # Define all the new output files
        self.daily_unscaled_path = config.PROCESSED_DATA_DIR / f'daily_features_unscaled_{subset_name}_v2.csv'
        self.daily_scaled_path = config.PROCESSED_DATA_DIR / f'engineered_daily_features_{subset_name}_v2.csv'
        self.static_scaled_path = config.PROCESSED_DATA_DIR / f'engineered_static_features_{subset_name}_v2.csv'
        self.sequence_path = config.PROCESSED_DATA_DIR / f'sequences_{subset_name}_v2.npy'
        self.sequence_labels_path = config.PROCESSED_DATA_DIR / f'sequence_labels_{subset_name}_v2.npy'
        self.sequence_users_path = config.PROCESSED_DATA_DIR / f'sequence_users_{subset_name}_v2.npy'
        # --- (END V2.1) ---
        
        self.chunk_size = chunk_size
        logger.info(f"Feature Engineer V2 initialized. Reading from: {self.raw_log_path.name}")

    def _load_ldap_data(self) -> pd.DataFrame:
        """
        Loads and prepares the canonical LDAP data (role info).
        """
        try:
            # (V3.9 FIX) Look for the single, shared LDAP folder
            ldap_path = config.RAW_DATA_DIR / config.LDAP_PATH
            logger.info(f"Scanning for LDAP data in shared folder: {ldap_path}")
            if not ldap_path.is_dir():
                raise FileNotFoundError(f"LDAP folder not found at {ldap_path}")
            
            ldap_files = list(ldap_path.glob("*.csv"))
            if not ldap_files:
                raise FileNotFoundError(f"No LDAP CSVs found in {ldap_path}")
            
            ldap_dfs = []
            for f in ldap_files:
                try:
                    df = pd.read_csv(f, usecols=['user_id', 'role'])
                except ValueError:
                    # Fallback if 'role' is missing
                    logger.warning(f"File {f.name} missing 'role' column, skipping.")
                    continue
                ldap_dfs.append(df)
            
            ldap_master = pd.concat(ldap_dfs, ignore_index=True)
            ldap_master = ldap_master.drop_duplicates('user_id', keep='last')
            logger.info(f"Loaded canonical role data for {len(ldap_master)} users.")
            return ldap_master
            
        except Exception as e:
            logger.error(f"Failed to load LDAP data for role enrichment: {e}")
            return None

    def _create_daily_aggregates_from_chunks(self):
        """
        PASS 1: Reads the massive log file in chunks and creates a
        daily aggregated summary file.
        """
        logger.info(utils.generate_report_header("PASS 1: DAILY AGGREGATION"))
        if not self.raw_log_path.exists():
            raise FileNotFoundError(f"Master *labeled* log file not found: {self.raw_log_path}")

        temp_dir = config.PROCESSED_DATA_DIR / "temp_daily_chunks"
        temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"Reading {self.raw_log_path} in {self.chunk_size}-row chunks...")
        
        chunk_files = []
        
        # Define the aggregations we want to perform
        aggregations = {
            'is_logon': 'sum',
            'is_logoff': 'sum',
            'is_connect': 'sum',
            'is_disconnect': 'sum',
            'email_size_kb': ['sum', 'mean'],
            'has_attachment': 'sum',
            'is_large_email': 'sum',
            'is_suspicious_domain': 'sum',
            'is_sensitive_file': 'sum',
            'activity_type': 'count', # Total activity count
            'is_insider': 'max' # Mark the *day* as anomalous
        }

        # Define all columns we need to load from the master log file
        all_cols_to_load = [
            'date', 'user_id', 'dataset', 'is_insider', 'role', # <-- Load 'role'
            'activity_type', # for count
            'is_logon', 'is_logoff', 
            'is_connect', 'is_disconnect',
            'email_size_kb', 'has_attachment', 'is_large_email',
            'is_suspicious_domain', 
            'is_sensitive_file'
        ]
        
        with pd.read_csv(self.raw_log_path, chunksize=self.chunk_size, low_memory=False,
                         usecols=lambda c: c in all_cols_to_load,
                         parse_dates=['date']) as reader:
            
            for i, chunk in enumerate(reader):
                logger.info(f"Processing chunk {i+1}...")
                
                # Fill NaNs for aggregation columns before grouping
                for col in aggregations.keys():
                    if col in chunk.columns and col not in ['activity_type', 'is_insider']:
                         chunk[col] = chunk[col].fillna(0)
                
                # Group by role, user, and dataset
                daily_chunk = chunk.groupby([
                    pd.Grouper(key='date', freq='D'), 
                    'user_id', 
                    'dataset',
                    'role' # <-- Add role to the groupby
                ]).agg(aggregations)
                
                # Flatten multi-index columns (e.g., 'email_size_kb_sum')
                daily_chunk.columns = ['_'.join(col).strip() for col in daily_chunk.columns.values]
                daily_chunk = daily_chunk.reset_index()
                
                # Save temp file
                temp_path = temp_dir / f"temp_daily_chunk_{i}.csv"
                daily_chunk.to_csv(temp_path, index=False)
                chunk_files.append(temp_path)

        logger.info(f"All {len(chunk_files)} chunks processed. Merging daily summaries...")
        
        # Load all temp files and merge
        df_list = [pd.read_csv(f) for f in chunk_files]
        merged_df = pd.concat(df_list, ignore_index=True)
        
        # We must re-group one last time to combine users/days
        # that were split across chunk boundaries
        final_daily_df = merged_df.groupby(['date', 'user_id', 'dataset', 'role']).sum().reset_index()
        
        # Clean up temp files
        logger.info("Cleaning up temporary chunk files...")
        shutil.rmtree(temp_dir)
        
        # Save the unscaled daily features
        final_daily_df.to_csv(self.daily_unscaled_path, index=False)
        logger.info(f"PASS 1 complete. Saved {self.daily_unscaled_path} ({len(final_daily_df)} rows)")
        return final_daily_df

    def _enrich_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PASS 2: Loads the daily summary and calculates advanced
        self-relative (Z-Score) and peer-relative (Z-Score) features.
        """
        logger.info(utils.generate_report_header("PASS 2: FEATURE ENRICHMENT (Z-SCORES)"))
        
        # (V2.5 FIX) 'role' is already in the dataframe, just fill NaNs
        df['role'] = df['role'].fillna('UNKNOWN')
        logger.info("Verified 'role' column for Pass 2.")

        # Ensure correct types and sorting
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['user_id', 'date']).reset_index(drop=True)
        
        # Get list of features to analyze
        # We use the 'is_insider_max' as the label
        label_col = 'is_insider_max'
        base_features = [col for col in df.columns if col not in ['date', 'user_id', 'role', 'dataset', label_col]]

        # --- 1. Self-Relative Features (Rolling Z-Score) ---
        logger.info("Calculating self-relative Z-Scores (rolling 30-day)...")
        for col in base_features:
            # Group by user and calculate rolling mean/std
            user_group = df.groupby('user_id')[col]
            rolling_mean = user_group.rolling(window=30, min_periods=1).mean().reset_index(level=0, drop=True)
            rolling_std = user_group.rolling(window=30, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
            
            # Calculate Z-Score: (value - avg) / (std + epsilon)
            df[f'{col}_zscore_self'] = (df[col] - rolling_mean) / (rolling_std + 1e-10)

        # --- 2. Peer-Relative Features (Group Z-Score) ---
        logger.info("Calculating peer-relative Z-Scores (by role)...")
        for col in base_features:
            # Group by role/date and get peer avg/std
            role_group = df.groupby(['role', 'date'])[col]
            peer_mean = role_group.transform('mean')
            peer_std = role_group.transform('std').fillna(0)
            
            # Calculate Z-Score
            df[f'{col}_zscore_peer'] = (df[col] - peer_mean) / (peer_std + 1e-10)

        # Replace infinite values (from divide-by-zero) with 0
        df.replace([np.inf, -np.inf], 0, inplace=True)
        
        logger.info(f"PASS 2 complete. Enriched features created.")
        
        # Save scaled daily data (for LSTM)
        logger.info("Scaling and saving final daily features (for LSTM)...")
        
        daily_scaler = StandardScaler()
        
        feature_cols = [col for col in df.columns if col not in ['date', 'user_id', 'role', 'dataset', label_col]]
        df[feature_cols] = daily_scaler.fit_transform(df[feature_cols])
        
        utils.save_model(daily_scaler, config.MODEL_PATHS['daily_scaler_v2'])
        df.to_csv(self.daily_scaled_path, index=False)
        logger.info(f"Saved {self.daily_scaled_path}")
        
        return df

    def _create_static_features(self, df_daily_enriched: pd.DataFrame) -> pd.DataFrame:
        """
        PASS 3: Aggregates the enriched daily features into a
        per-user static profile (for IF/Clustering).
        """
        logger.info(utils.generate_report_header("PASS 3: STATIC FEATURE CREATION"))
        
        # We use the *enriched* dataframe from Pass 2
        label_col = 'is_insider_max'

        # Define aggregations. We want the 'max' of Z-Scores.
        static_aggs = {}
        zscore_cols = [col for col in df_daily_enriched.columns if 'zscore' in col]
        for col in zscore_cols:
            static_aggs[col] = ['max', 'mean'] # Get max and avg anomaly
            
        # Get total counts
        base_features = [col for col in df_daily_enriched.columns if col.endswith(('_sum', '_mean', '_count'))]
        for col in base_features:
            static_aggs[col] = 'sum'
            
        # Aggregate by user
        static_df = df_daily_enriched.groupby('user_id').agg(static_aggs)
        
        # Flatten columns
        static_df.columns = ['_'.join(col).strip() for col in static_df.columns.values]
        
        # Get the final label (is this user *ever* an insider?)
        static_labels = df_daily_enriched.groupby('user_id')[label_col].max()
        static_df = static_df.join(static_labels.rename('is_insider'))
        
        static_df = static_df.reset_index()
        static_df = static_df.fillna(0) # Fill any NaNs
        
        logger.info(f"Created static features for {len(static_df)} users")
        
        # Scale and save
        logger.info("Scaling and saving static features...")
        
        static_scaler = StandardScaler()
        
        feature_cols = [col for col in static_df.columns if col not in ['user_id', 'is_insider']]
        static_df[feature_cols] = static_scaler.fit_transform(static_df[feature_cols])
        
        utils.save_model(static_scaler, config.MODEL_PATHS['static_scaler_v2'])
        static_df.to_csv(self.static_scaled_path, index=False)
        logger.info(f"PASS 3 complete. Saved {self.static_scaled_path}")
        return static_df
        
    def _create_sequences(self, df_daily_scaled: pd.DataFrame):
        """
        PASS 4: Uses the enriched & scaled daily features
        to build sequential data (for LSTM).
        """
        logger.info(utils.generate_report_header("PASS 4: SEQUENCE CREATION"))
        
        df = df_daily_scaled
        label_col = 'is_insider_max'
        feature_cols = [col for col in df.columns if col not in ['date', 'user_id', 'role', 'dataset', label_col]]
        
        sequences = []
        labels = []
        users = []
        
        sequence_length = config.SEQUENCE_LENGTH
        stride = config.SEQUENCE_STRIDE
        
        logger.info(f"Creating sequences (length={sequence_length}, stride={stride})...")
        
        # (V2.5) Get a sorted list of unique users
        unique_users = df['user_id'].unique()
        unique_users.sort()
        
        for user in unique_users:
            user_data = df[df['user_id'] == user]
            
            # Sort by date to ensure order
            user_data = user_data.sort_values('date')
            
            user_features = user_data[feature_cols].values
            user_labels = user_data[label_col].values
            
            for i in range(0, len(user_features) - sequence_length + 1, stride):
                seq = user_features[i : i + sequence_length]
                label_window = user_labels[i : i + sequence_length]
                
                sequences.append(seq)
                # Label is 1 if *any* day in the sequence is anomalous
                labels.append(1 if np.any(label_window == 1) else 0)
                users.append(user)

        X = np.array(sequences)
        y = np.array(labels)
        u = np.array(users)
        
        logger.info(f"Created {len(X)} sequences from {len(df['user_id'].unique())} users")
        
        # Save sequence files
        np.save(self.sequence_path, X)
        np.save(self.sequence_labels_path, y)
        np.save(self.sequence_users_path, u)
        
        logger.info(f"PASS 4 complete. Saved sequence files to {self.sequence_path.parent}")
        return X, y, u

    def run_full_pipeline(self):
        """
        Orchestrates the full 4-pass feature engineering pipeline.
        """
        logger.info(utils.generate_report_header("FEATURE ENGINEERING PIPELINE V2 (MULTI-PASS)"))
        
        # --- PASS 1: Create Daily Aggregates from Chunks ---
        # Check if we can skip this (it's the slowest step)
        
        # --- (V2.4 FIX) ---
        # Correctly reference the flag from the PERFORMANCE dictionary
        if self.daily_unscaled_path.exists() and not config.PERFORMANCE['FORCE_RERUN_PASS1']:
        # --- (END V2.4 FIX) ---
            logger.info(f"Found {self.daily_unscaled_path}, skipping Pass 1.")
            daily_unscaled_df = pd.read_csv(self.daily_unscaled_path)
        else:
            daily_unscaled_df = self._create_daily_aggregates_from_chunks()
        
        if daily_unscaled_df is None:
            logger.error("Pass 1 failed. Aborting.")
            return

        # --- PASS 2: Enrich Daily Features (Z-Scores) ---
        daily_enriched_scaled_df = self._enrich_daily_features(daily_unscaled_df)
        
        # --- PASS 3: Create Static Features ---
        static_features_df = self._create_static_features(daily_enriched_scaled_df)
        
        # --- PASS 4: Create Sequences ---
        self._create_sequences(daily_enriched_scaled_df)
        
        logger.info("="*80)
        logger.info("V2 Feature Engineering Complete!")
        logger.info(f"  Static features created: {self.static_scaled_path}")
        logger.info(f"  Sequence files created: {self.sequence_path.parent}")
        logger.info("Next step: Train V2 models.")
        logger.info("="*80)

def main():
    engineer = FeatureEngineerV2()
    engineer.run_full_pipeline()

if __name__ == "__main__":
    main()