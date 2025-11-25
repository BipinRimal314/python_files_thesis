"""
Data Preprocessing Pipeline (V2: Multi-Dataset) - MEMORY SAFE
Handles loading, merging, and preparing *all* CMU-CERT datasets (r1-r5.2)
from the 'data/all_data' directory.

[--- V3.3 FIX ---]
- The _merge_all_chunks function is now memory-safe.
- It no longer loads all temp files into RAM.
- It reads each temp file, one by one, in chunks,
  and appends them to the final master file on disk.
- This will prevent the "zsh: killed" OOM error.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os
import shutil
from typing import Dict, List, Tuple

# Use the new config file
import config_v2 as config 
import utils

logger = utils.logger

class DataPreprocessorV2:
    
    def __init__(self, chunk_size=1000000):
        self.raw_data_dir = config.RAW_DATA_DIR
        self.processed_dir = config.PROCESSED_DATA_DIR
        self.dataset_folders = sorted([d for d in self.raw_data_dir.iterdir() if d.is_dir() and d.name.startswith('r')])
        self.temp_files = []
        self.master_file_path = self.processed_dir / 'processed_unified_logs_ALL.csv'
        self.chunk_size = chunk_size
        logger.info(f"Found {len(self.dataset_folders)} dataset folders in {self.raw_data_dir}")

        # Define all possible new feature columns that will be created
        self.feature_columns = [
            'is_logon', 'is_logoff', 'is_connect', 'is_disconnect',
            'email_size_kb', 'has_attachment', 'is_large_email',
            'is_suspicious_domain', 'file_extension', 'is_sensitive_file'
        ]

    def _cleanup_old_temp_files(self):
        """
        Deletes any 'temp_processed_*.csv' files from previous failed runs.
        """
        logger.info("Cleaning up old temporary files...")
        count = 0
        for f in self.processed_dir.glob("temp_processed_*.csv"):
            try:
                os.remove(f)
                count += 1
            except Exception as e:
                logger.warning(f"Could not remove old temp file {f}: {e}")
        logger.info(f"Removed {count} old temp files.")

    def _process_single_dataset(self, folder_path: Path) -> Path:
        """
        Processes all log files for a *single* dataset folder (e.g., 'r1')
        and saves them to a single temporary processed CSV.
        """
        dataset_id = folder_path.name
        logger.info(f"--- Processing single dataset: {dataset_id} ---")
        
        processed_dfs = []

        # Load and process each log type for *this folder only*
        for log_type, filename in config.LOG_FILENAMES.items():
            filepath = folder_path / filename
            if not filepath.exists():
                logger.warning(f"{filename} not found in {dataset_id}, skipping.")
                continue
            
            try:
                df = pd.read_csv(filepath, low_memory=False)
                logger.info(f"Loaded {filename} ({len(df)} rows)")
                
                # --- Preprocessing ---
                if 'date' not in df.columns:
                    logger.warning(f"'date' column not found in {filepath}. Skipping file.")
                    continue 
                
                df = utils.parse_datetime(df, 'date')
                df['activity_type'] = log_type
                
                # --- Standardize column names (No Hashing) ---
                if 'user' in df.columns:
                    df = df.rename(columns={'user': 'user_id'})
                elif 'from' in df.columns: # For email
                    df = df.rename(columns={'from': 'user_id'})
                else:
                    logger.warning(f"No 'user' or 'from' column in {filename}, rows may not be associated with a user.")
                    df['user_id'] = 'UNKNOWN'
                
                if 'pc' in df.columns:
                    df = df.rename(columns={'pc': 'pc_id'})

                # --- Feature Creation ---
                if log_type == 'logon' and 'activity' in df.columns:
                    df['is_logon'] = (df['activity'] == 'Logon').astype(int)
                    df['is_logoff'] = (df['activity'] == 'Logoff').astype(int)
                
                if log_type == 'device' and 'activity' in df.columns:
                    df['is_connect'] = (df['activity'] == 'Connect').astype(int)
                    df['is_disconnect'] = (df['activity'] == 'Disconnect').astype(int)

                if log_type == 'email':
                    if 'size' in df.columns:
                        df['email_size_kb'] = df['size'] / 1024
                        df['is_large_email'] = (df['email_size_kb'] > 1024).astype(int)
                    if 'attachments' in df.columns:
                        df['has_attachment'] = (df['attachments'].notna()).astype(int)
                
                if log_type == 'http' and 'url' in df.columns:
                    df['domain'] = df['url'].str.extract(r'https?://([^/]+)')[0]
                    suspicious_keywords = ['temp', 'anonymous', 'proxy', 'vpn', 'leak', 'hack']
                    df['is_suspicious_domain'] = df['domain'].str.lower().str.contains('|'.join(suspicious_keywords), na=False).astype(int)

                if log_type == 'file' and 'filename' in df.columns:
                    df['file_extension'] = df['filename'].str.split('.').str[-1].str.lower()
                    sensitive_extensions = ['doc', 'docx', 'xls', 'xlsx', 'pdf', 'txt', 'sql', 'db', 'zip', 'rar']
                    df['is_sensitive_file'] = df['file_extension'].isin(sensitive_extensions).astype(int)
                
                processed_dfs.append(df)
                
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                import traceback
                traceback.print_exc()
        
        # Merge all log types *for this one dataset*
        if not processed_dfs:
            logger.warning(f"No data processed for {dataset_id}.")
            return None

        unified_df = pd.concat(processed_dfs, ignore_index=True, sort=False)
        unified_df['dataset'] = dataset_id # Add the dataset ID
        
        # Ensure ALL possible feature columns exist
        for col in self.feature_columns:
            if col not in unified_df.columns:
                unified_df[col] = 0 # Create it as all zeros
            else:
                unified_df[col] = unified_df[col].fillna(0)
            
        # Clean and save to a temp file
        unified_df = utils.clean_dataframe(unified_df)
        temp_save_path = self.processed_dir / f"temp_processed_{dataset_id}.csv"
        
        try:
            unified_df.to_csv(temp_save_path, index=False)
            self.temp_files.append(temp_save_path)
            logger.info(f"Saved temporary file: {temp_save_path} ({len(unified_df)} rows)")
            return temp_save_path
        except Exception as e:
            logger.error(f"Failed to save temp file {temp_save_path}: {e}")
            return None

    def _merge_all_chunks(self) -> pd.DataFrame:
        """
        (--- FIX V3.3 ---)
        Merges all temporary processed files into one master dataframe
        using a memory-safe, chunked append method.
        """
        logger.info(f"Merging {len(self.temp_files)} processed temp files into master file...")
        
        if not self.temp_files:
            logger.error("No temp files were created. Cannot merge.")
            return None
        
        # Ensure master file is empty before we start
        if self.master_file_path.exists():
            os.remove(self.master_file_path)
            
        header_written = False
        total_rows = 0
        
        try:
            for f in self.temp_files:
                logger.info(f"Merging {f}...")
                with pd.read_csv(f, chunksize=self.chunk_size, low_memory=False) as reader:
                    for chunk in reader:
                        if not header_written:
                            # Write header for the *first chunk* of the *first file*
                            chunk.to_csv(self.master_file_path, index=False, mode='w')
                            header_written = True
                        else:
                            # Append all other chunks with no header
                            chunk.to_csv(self.master_file_path, index=False, mode='a', header=False)
                        
                        total_rows += len(chunk)
                
                # Clean up temp file *after* merging
                os.remove(f)

            logger.info(f"Master dataframe created: {total_rows} total rows")
            
            # Now, load the master file *one last time* to sort it
            # This is still a big memory hit, but let's try
            # If this fails, we can skip sorting
            try:
                logger.info("Loading master file to sort by date...")
                master_df = pd.read_csv(self.master_file_path)
                master_df = master_df.sort_values('date').reset_index(drop=True)
                logger.info("Sorting complete. Re-saving...")
                master_df.to_csv(self.master_file_path, index=False)
                return master_df
            except Exception as e:
                logger.warning(f"Could not load master file to sort: {e}. File will be unsorted, but complete.")
                # Return None to signal we can't do the next steps
                return None

        except Exception as e:
            logger.error(f"Failed to merge temp files: {e}")
            return None

    def _load_and_merge_static_data(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """
        Loads and de-duplicates static data (LDAP, Psychometric)
        and merges it into the master dataframe using the raw `user_id`.
        """
        logger.info("Loading and merging static data (LDAP, Psychometric)...")
        
        # 1. Load LDAP data
        try:
            ldap_path = self.raw_data_dir / config.LDAP_PATH
            ldap_files = list(ldap_path.glob("*.csv"))
            if not ldap_files:
                raise FileNotFoundError(f"No LDAP CSVs found in {ldap_path}")
                
            ldap_dfs = [pd.read_csv(f, usecols=['user_id', 'role', 'domain']) for f in ldap_files]
            ldap_master = pd.concat(ldap_dfs, ignore_index=True)
            ldap_master = ldap_master.drop_duplicates('user_id', keep='last')
            
            # Merge directly on `user_id`
            master_df = master_df.merge(ldap_master[['user_id', 'role', 'domain']], on='user_id', how='left')
            logger.info(f"Merged LDAP data for {len(ldap_master)} users")
            
        except Exception as e:
            logger.error(f"Failed to load/merge LDAP data: {e}")

        # 2. Load Psychometric data
        try:
            psych_dfs = []
            for folder in self.dataset_folders:
                filepath = folder / config.STATIC_FILENAMES['psychometric']
                if filepath.exists():
                    psych_dfs.append(pd.read_csv(filepath))
            
            if psych_dfs:
                psych_master = pd.concat(psych_dfs, ignore_index=True)
                psych_master = psych_master.drop_duplicates('user_id', keep='first')
                
                # Merge directly on `user_id`
                psych_cols = ['user_id', 'O', 'C', 'E', 'A', 'N']
                master_df = master_df.merge(psych_master[psych_cols], on='user_id', how='left')
                logger.info(f"Merged psychometric data for {len(psych_master)} users")
            
        except Exception as e:
            logger.error(f"Failed to load/merge psychometric data: {e}")
        
        return master_df

    def run_full_pipeline(self):
        """
        Runs the new V2 preprocessing pipeline.
        """
        logger.info(utils.generate_report_header("DATA PREPROCESSING PIPELINE (V2: MEMORY-SAFE)"))
        
        # Clean up old runs first
        self._cleanup_old_temp_files()

        # 1. Process each dataset folder individually
        for folder in self.dataset_folders:
            self._process_single_dataset(folder)
            
        # 2. Merge all processed temp files
        master_df = self._merge_all_chunks()
        if master_df is None:
            logger.error("Merging failed. Exiting.")
            return

        # 3. Load and merge static data (LDAP, Psychometric)
        master_df = self._load_and_merge_static_data(master_df)

        # 4. Final cleaning and save
        # Fill NaNs for *all* feature columns one last time
        for col in self.feature_columns:
            if col in master_df.columns:
                master_df[col] = master_df[col].fillna(0)
                
        master_df = utils.handle_missing_values(master_df, strategy='fill_mode')
        master_df['is_insider'] = 0 # Initialize label column
        
        logger.info(f"Saving new master file to {self.master_file_path}...")
        
        try:
            master_df.to_csv(self.master_file_path, index=False)
            logger.info("="*80)
            logger.info("V2 Preprocessing Complete!")
            logger.info(f"Master file saved: {len(master_df)} rows")
            logger.info("Next step: python label_insiders_v2.py")
            logger.info("="*80)
        except Exception as e:
            logger.error(f"Failed to save final master file: {e}")
        
        return master_df

def main():
    preprocessor = DataPreprocessorV2()
    preprocessor.run_full_pipeline()

if __name__ == "__main__":
    main()