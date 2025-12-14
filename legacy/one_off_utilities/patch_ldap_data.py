"""
V2 Pipeline - Step 1.5 (HOTFIX SCRIPT)
Patches the 'role' and 'psychometric' columns into the
'...ENRICHED.csv' file.

This script fixes the LDAP path bug in data_preprocessing_v2.py
*without* forcing a 2-hour re-run.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil
from typing import Dict, List, Tuple

import config_v2 as config 
import utils

logger = utils.logger

def load_static_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and de-duplicates LDAP and Psychometric data
    from the correct *dataset subfolders*.
    """
    ldap_master = None
    psych_master = None
    
    # Get the subset of folders we are processing
    dataset_folders = []
    subset = config.DATASET_SUBSET
    logger.info(f"Loading static data for subset: {subset}")
    for d_name in subset:
        d_path = config.RAW_DATA_DIR / d_name
        if d_path.is_dir():
            dataset_folders.append(d_path)
            
    # 1. Load LDAP Data
    try:
        ldap_dfs = []
        logger.info("Scanning for LDAP data in dataset folders...")
        for folder in dataset_folders:
            # (FIX) Look inside the *folder* for LDAP
            ldap_path = folder / config.LDAP_PATH 
            if ldap_path.is_dir():
                ldap_files = list(ldap_path.glob("*.csv"))
                for f in ldap_files:
                    try:
                        ldap_cols = ['user_id', 'role', 'domain']
                        df = pd.read_csv(f, usecols=lambda c: c in ldap_cols)
                        ldap_dfs.append(df)
                    except ValueError:
                        ldap_cols = ['user_id', 'role']
                        df = pd.read_csv(f, usecols=lambda c: c in ldap_cols)
                        ldap_dfs.append(df)
        
        if not ldap_dfs:
            raise FileNotFoundError(f"No LDAP CSVs found in any dataset subfolders.")

        ldap_master = pd.concat(ldap_dfs, ignore_index=True)
        ldap_master = ldap_master.drop_duplicates('user_id', keep='last')
        logger.info(f"Loaded and de-duplicated LDAP data for {len(ldap_master)} users")
        
    except Exception as e:
        logger.error(f"Failed to load/merge LDAP data: {e}")

    # 2. Load Psychometric Data
    try:
        psych_dfs = []
        for folder in dataset_folders: 
            filepath = folder / config.STATIC_FILENAMES['psychometric']
            if filepath.exists():
                psych_dfs.append(pd.read_csv(filepath))
        
        if psych_dfs:
            psych_master = pd.concat(psych_dfs, ignore_index=True)
            psych_master = psych_master.drop_duplicates('user_id', keep='first')
            logger.info(f"Loaded and de-duplicated psychometric data for {len(psych_master)} users")
    except Exception as e:
        logger.error(f"Failed to load/merge psychometric data: {e}")
        
    return ldap_master, psych_master

def run_patch():
    """
    Reads the ENRICHED file in chunks, merges LDAP/Psychometric data,
    and replaces the original file.
    """
    logger.info(utils.generate_report_header("STATIC DATA PATCH SCRIPT (V2)"))
    
    # --- Dynamically get the file names ---
    subset = config.DATASET_SUBSET
    subset_name = "_".join(subset)
    target_file = config.PROCESSED_DATA_DIR / f'processed_unified_logs_{subset_name}_ENRICHED.csv'
    temp_file = config.PROCESSED_DATA_DIR / 'temp_patching.csv'
    # ---

    if not target_file.exists():
        logger.error(f"Master file not found: {target_file}")
        logger.error("Please run data_preprocessing_v2.py first.")
        return

    # 1. Load the LDAP and Psychometric data (this is small)
    ldap_df, psych_df = load_static_data()
    if ldap_df is None:
        logger.warning("Could not load LDAP data. 'role' column will be 'UNKNOWN'.")
    if psych_df is None:
        logger.warning("Could not load Psychometric data. 'O,C,E,A,N' columns will be 0.")

    # 2. Stream-read the big file, merge, and stream-write
    header_written = False
    try:
        with pd.read_csv(target_file, chunksize=config.PERFORMANCE['chunk_size'], low_memory=False) as reader:
            for i, chunk in enumerate(reader):
                logger.info(f"Patching chunk {i+1}...")
                
                # Drop old, empty columns if they exist
                chunk = chunk.drop(columns=['role', 'domain', 'O', 'C', 'E', 'A', 'N'], errors='ignore')
                
                # Merge the correct static data
                if ldap_df is not None:
                    chunk = chunk.merge(ldap_df, on='user_id', how='left')
                if psych_df is not None:
                    psych_cols = ['user_id', 'O', 'C', 'E', 'A', 'N']
                    cols_to_merge = [c for c in psych_cols if c in psych_df.columns]
                    chunk = chunk.merge(psych_df[cols_to_merge], on='user_id', how='left')
                
                # Fill NaNs for users not in the static files
                static_cols = ['role', 'domain', 'O', 'C', 'E', 'A', 'N']
                for col in static_cols:
                    if col in chunk.columns:
                        if col in ['role', 'domain']:
                            chunk[col] = chunk[col].fillna('UNKNOWN')
                        else:
                            chunk[col] = chunk[col].fillna(0)
                
                # Write to temp file
                if not header_written:
                    chunk.to_csv(temp_file, index=False, mode='w')
                    header_written = True
                else:
                    chunk.to_csv(temp_file, index=False, mode='a', header=False)

        # 3. Replace the old file with the patched file
        logger.info("Patching complete. Replacing original file...")
        shutil.move(temp_file, target_file)
        logger.info(f"Successfully patched {target_file.name}")
        logger.info("="*80)
        logger.info("PATCH COMPLETE.")
        logger.info("Your data is now 100% correct.")
        logger.info("Next step: python label_insiders_v2.py")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Error during patching: {e}")
        if temp_file.exists():
            os.remove(temp_file)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_patch()