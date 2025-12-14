"""
Data Preprocessing Pipeline (V2: Multi-Dataset) - MEMORY SAFE
Handles loading, merging, and preparing the CMU-CERT dataset subset.

[--- V4.0 FINAL FIX ---]
- The _merge_all_chunks function is now *TRULY* memory-safe.
- It no longer loads entire temp files (like the 14.59GB r3.1) into RAM.
- It now reads *each temp file* in chunks, fixes the schema
  *per chunk*, and appends *per chunk* to the master file.
- This fixes the "thrashing" / OOM error during the merge step.
- All other fixes (LDAP, Schema, Checkpointing) are retained.
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
        
        self.dataset_folders = []
        if hasattr(config, 'DATASET_SUBSET') and config.DATASET_SUBSET:
            logger.info(f"Processing SUBSET defined in config: {config.DATASET_SUBSET}")
            for d_name in config.DATASET_SUBSET:
                d_path = self.raw_data_dir / d_name
                if d_path.is_dir():
                    self.dataset_folders.append(d_path)
                else:
                    logger.warning(f"Dataset folder '{d_name}' from subset not found.")
        else:
            logger.warning("No DATASET_SUBSET defined in config, processing all r* folders.")
            self.dataset_folders = sorted([d for d in self.raw_data_dir.iterdir() if d.is_dir() and d.name.startswith('r')])

        self.temp_files_to_merge = [] # List of temp file paths to be merged
        
        subset_name = "_".join(config.DATASET_SUBSET) if hasattr(config, 'DATASET_SUBSET') and config.DATASET_SUBSET else "ALL"
        self.master_file_path = self.processed_dir / f'processed_unified_logs_{subset_name}.csv'
        self.final_enriched_file_path = self.processed_dir / f'processed_unified_logs_{subset_name}_ENRICHED.csv'
        
        self.chunk_size = chunk_size
        logger.info(f"Found {len(self.dataset_folders)} dataset folders to process.")

        self.all_possible_log_columns = [
            'date', 'user_id', 'pc_id', 'activity_type', 'dataset',
            'activity', 'is_logon', 'is_logoff',
            'is_connect', 'is_disconnect',
            'to', 'from', 'size', 'attachments', 'cc', 'bcc',
            'email_size_kb', 'has_attachment', 'is_large_email',
            'url', 'domain', 'is_suspicious_domain',
            'filename', 'content', 'file_extension', 'is_sensitive_file'
        ]
        self.numeric_feature_cols = [
            'is_logon', 'is_logoff', 'is_connect', 'is_disconnect',
            'email_size_kb', 'has_attachment', 'is_large_email',
            'is_suspicious_domain', 'is_sensitive_file'
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
        temp_save_path = self.processed_dir / f"temp_processed_{dataset_id}.csv"
        
        if temp_save_path.exists():
            logger.warning(f"Temp file {temp_save_path.name} already exists. Skipping processing.")
            self.temp_files_to_merge.append(temp_save_path)
            return temp_save_path
            
        logger.info(f"--- Processing single dataset: {dataset_id} ---")
        processed_dfs = []

        for log_type, filename in config.LOG_FILENAMES.items():
            filepath = folder_path / filename
            if not filepath.exists():
                logger.warning(f"{filename} not found in {dataset_id}, skipping.")
                continue
            
            try:
                df = pd.read_csv(filepath, low_memory=False)
                logger.info(f"Loaded {filename} ({len(df)} rows)")
                
                if 'date' not in df.columns:
                    logger.warning(f"'date' column not found in {filepath}. Skipping file.")
                    continue 
                
                df = utils.parse_datetime(df, 'date')
                df['activity_type'] = log_type
                
                if 'user' in df.columns:
                    df = df.rename(columns={'user': 'user_id'})
                elif 'from' in df.columns: # For email
                    df = df.rename(columns={'from': 'user_id'})
                else:
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
        
        if not processed_dfs:
            logger.warning(f"No data processed for {dataset_id}.")
            return None

        unified_df = pd.concat(processed_dfs, ignore_index=True, sort=False)
        unified_df['dataset'] = dataset_id 
        
        # --- (V3.9 SCHEMA-SAFE FIX) ---
        for col in self.all_possible_log_columns:
            if col not in unified_df.columns:
                unified_df[col] = np.nan 
        
        for col in self.numeric_feature_cols:
             if col in unified_df.columns:
                unified_df[col] = unified_df[col].fillna(0)
        
        all_cols_ordered = [c for c in self.all_possible_log_columns if c in unified_df.columns]
        other_cols = [c for c in unified_df.columns if c not in all_cols_ordered]
        unified_df = unified_df[all_cols_ordered + other_cols]
        # --- (END V3.9 SCHEMA-SAFE FIX) ---
            
        unified_df = utils.clean_dataframe(unified_df)
        
        try:
            unified_df.to_csv(temp_save_path, index=False)
            self.temp_files_to_merge.append(temp_save_path)
            logger.info(f"Saved temporary file: {temp_save_path} ({len(unified_df)} rows)")
            return temp_save_path
        except Exception as e:
            logger.error(f"Failed to save temp file {temp_save_path}: {e}")
            return None

    def _merge_all_chunks(self) -> bool:
        """
        (--- V4.0 FIX ---)
        Merges all temporary processed files into one master file.
        This version reads *each temp file* in chunks to
        prevent OOM errors from large temp files (like 14GB r3.1).
        """
        logger.info(f"Merging {len(self.temp_files_to_merge)} processed temp files into master file (streaming)...")
        
        if not self.temp_files_to_merge:
            logger.error("No temp files were created. Cannot merge.")
            return False
        
        if self.master_file_path.exists():
            logger.warning(f"Master file {self.master_file_path.name} already exists. Deleting.")
            os.remove(self.master_file_path)
            
        header_written = False
        total_rows = 0
        
        try:
            # Get the *master* column order from the *first* temp file
            # (V3.9 fix guarantees all temp files have same cols)
            master_columns = pd.read_csv(self.temp_files_to_merge[0], nrows=1).columns.tolist()
            
            for f_path in self.temp_files_to_merge:
                logger.info(f"Merging {f_path.name} *in chunks*...")
                
                with pd.read_csv(f_path, chunksize=self.chunk_size, low_memory=False) as reader:
                    for chunk in reader:
                        # Ensure schema is 100% consistent
                        
                        # Add any missing columns (shouldn't happen, but safe)
                        for col in master_columns:
                            if col not in chunk.columns:
                                chunk[col] = np.nan
                        
                        # Re-order to match master
                        chunk = chunk[master_columns]
                        
                        # Now save this fixed chunk
                        if not header_written:
                            chunk.to_csv(self.master_file_path, index=False, mode='w')
                            header_written = True
                        else:
                            chunk.to_csv(self.master_file_path, index=False, mode='a', header=False)
                        
                        total_rows += len(chunk)

            logger.info(f"Master dataframe created: {total_rows} total rows")
            
            # (V3.7 Checkpointing) We still do not delete the temp file
            logger.info("Successfully merged all temp files.")
            return True

        except Exception as e:
            logger.error(f"Failed to merge temp files: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_and_merge_static_data_streaming(self) -> bool:
        """
        Loads and de-duplicates static data (LDAP, Psychometric)
        and merges it into the master dataframe using a memory-safe
        streaming (chunked) method.
        Saves to a NEW file and replaces the old one.
        """
        logger.info("Loading and merging static data (LDAP, Psychometric) via streaming...")
        
        # 1. Load small static files into memory (these are small)
        ldap_master = None
        psych_master = None
        
        try:
            # (V3.9 LDAP FIX)
            ldap_path = self.raw_data_dir / config.LDAP_PATH
            logger.info(f"Scanning for LDAP data in shared folder: {ldap_path}")
            if not ldap_path.is_dir():
                raise FileNotFoundError(f"LDAP folder not found at {ldap_path}")
            
            ldap_files = list(ldap_path.glob("*.csv"))
            if not ldap_files:
                raise FileNotFoundError(f"No LDAP CSVs found in {ldap_path}")
            
            ldap_dfs = []
            for f in ldap_files:
                try:
                    df = pd.read_csv(f, usecols=['user_id', 'role', 'domain'])
                except ValueError:
                    df = pd.read_csv(f, usecols=['user_id', 'role'])
                ldap_dfs.append(df)

            ldap_master = pd.concat(ldap_dfs, ignore_index=True)
            ldap_master = ldap_master.drop_duplicates('user_id', keep='last')
            logger.info(f"Loaded and de-duplicated LDAP data for {len(ldap_master)} users")
            
        except Exception as e:
            logger.error(f"Failed to load/merge LDAP data: {e}")

        try:
            psych_dfs = []
            for folder in self.dataset_folders: 
                filepath = folder / config.STATIC_FILENAMES['psychometric']
                if filepath.exists():
                    psych_dfs.append(pd.read_csv(filepath))
            
            if psych_dfs:
                psych_master = pd.concat(psych_dfs, ignore_index=True)
                psych_master = psych_master.drop_duplicates('user_id', keep='first')
                logger.info(f"Loaded and de-duplicated psychometric data for {len(psych_master)} users")
        except Exception as e:
            logger.error(f"Failed to load/merge psychometric data: {e}")

        # 2. Stream-merge this data into the master file
        temp_enriched_file = self.processed_dir / 'temp_enriched.csv'
        header_written = False
        
        try:
            with pd.read_csv(self.master_file_path, chunksize=self.chunk_size, low_memory=False) as reader:
                for i, chunk in enumerate(reader):
                    logger.info(f"Enriching chunk {i+1}...")
                    
                    # Merge static data
                    if ldap_master is not None:
                        chunk = chunk.merge(ldap_master, on='user_id', how='left')
                    if psych_master is not None:
                        psych_cols = ['user_id', 'O', 'C', 'E', 'A', 'N']
                        cols_to_merge = [c for c in psych_cols if c in psych_master.columns]
                        chunk = chunk.merge(psych_master[cols_to_merge], on='user_id', how='left')
                    
                    if not header_written:
                        chunk.to_csv(temp_enriched_file, index=False, mode='w')
                        header_written = True
                    else:
                        chunk.to_csv(temp_enriched_file, index=False, mode='a', header=False)
            
            logger.info("Enrichment complete. Replacing master file...")
            shutil.move(temp_enriched_file, self.final_enriched_file_path)
            
            if self.master_file_path.exists():
                os.remove(self.master_file_path)
                
            logger.info(f"Successfully created final enriched file: {self.final_enriched_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stream-merge static data: {e}")
            if temp_enriched_file.exists():
                os.remove(temp_enriched_file)
            import traceback
            traceback.print_exc()
            return False

    def run_full_pipeline(self):
        """
        Runs the new V2 preprocessing pipeline.
        """
        logger.info(utils.generate_report_header("DATA PREPROCESSING PIPELINE (V2: MEMORY-SAFE)"))
        
        if self.final_enriched_file_path.exists() and not config.PERFORMANCE['FORCE_RERUN_PREPROCESSING']:
            logger.warning("="*80)
            logger.warning(f"Final file '{self.final_enriched_file_path.name}' already exists.")
            logger.warning("Skipping entire preprocessing pipeline.")
            logger.warning("To re-run, set FORCE_RERUN_PREPROCESSING = True in config_v2.py")
            logger.warning("="*80)
            return

        # --- (V3.8) We only clean up if we are forced to re-run from scratch
        if config.PERFORMANCE['FORCE_RERUN_PREPROCESSING']:
            self._cleanup_old_temp_files()
        
        # Smartly find existing temp files
        self.temp_files_to_merge = [] # Reset list
        for folder in self.dataset_folders:
            temp_path = self.processed_dir / f"temp_processed_{folder.name}.csv"
            if temp_path.exists():
                self.temp_files_to_merge.append(temp_path)
        
        if len(self.temp_files_to_merge) == len(self.dataset_folders):
             logger.info(f"Found {len(self.temp_files_to_merge)} existing temp files. Will skip processing and go straight to merging.")
        else:
            logger.info("Did not find all temp files. Will re-process missing ones.")
            self.temp_files_to_merge = [] # Clear the list
            self._cleanup_old_temp_files() # Start from a clean slate
            # 1. Process each dataset folder individually
            for folder in self.dataset_folders:
                self._process_single_dataset(folder)

        # 2. Merge all processed temp files
        if not self._merge_all_chunks():
            logger.error("Merging failed. Aborting.")
            return

        # 3. Load and merge static data (Streaming)
        if not self._load_and_merge_static_data_streaming():
            logger.error("Static data enrichment failed. Aborting.")
            return

        # 4. (V3.9 OOM FIX) Final pass to clean up
        temp_final_file = self.processed_dir / 'temp_final_with_labels.csv'
        header_written = False
        total_rows = 0
        try:
            logger.info("Performing final pass (cleanup and label init)...")
            with pd.read_csv(self.final_enriched_file_path, chunksize=self.chunk_size, low_memory=False) as reader:
                for i, chunk in enumerate(reader):
                    logger.info(f"Finalizing chunk {i+1}...")
                    
                    # Fill NaNs
                    for col in self.all_possible_log_columns:
                        if col in chunk.columns:
                            if pd.api.types.is_numeric_dtype(chunk[col]):
                                chunk[col] = chunk[col].fillna(0)
                            else:
                                chunk[col] = chunk[col].fillna('UNKNOWN')
                    
                    static_cols = ['role', 'domain', 'O', 'C', 'E', 'A', 'N']
                    for col in static_cols:
                        if col in chunk.columns:
                            if col in ['role', 'domain']:
                                chunk[col] = chunk[col].fillna('UNKNOWN')
                            else:
                                chunk[col] = chunk[col].fillna(0)
                    
                    chunk['is_insider'] = 0 # Initialize label column
                    
                    if not header_written:
                        chunk.to_csv(temp_final_file, index=False, mode='w')
                        header_written = True
                    else:
                        chunk.to_csv(temp_final_file, index=False, mode='a', header=False)
                    total_rows += len(chunk)
            
            shutil.move(temp_final_file, self.final_enriched_file_path)
            logger.info("="*80)
            logger.info("V2 Preprocessing Complete!")
            logger.info(f"Master file saved: {self.final_enriched_file_path.name} ({total_rows} rows)")
            logger.info("Next step: python label_insiders_v2.py")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Failed during final cleanup pass: {e}")
            if temp_final_file.exists():
                os.remove(temp_final_file)
        
        return

def main():
    preprocessor = DataPreprocessorV2()
    preprocessor.run_full_pipeline()

if __name__ == "__main__":
    main()