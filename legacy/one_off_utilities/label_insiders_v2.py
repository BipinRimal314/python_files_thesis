"""
Label insider threats using CMU-CERT answers file (V2: Multi-Dataset)

[--- V2.3 FINAL FIX ---]
- This script *correctly* normalizes dataset names from insiders.csv.
- It now converts '2.0' -> 'r2' and '3.1-1' -> 'r3.1'.
- This will *finally* fix the 'Total insider records labeled: 0' bug.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil

# Use the new V2 config
import config_v2 as config 
import utils

logger = utils.logger

class InsiderLabelerV2:
    
    def __init__(self, answers_path: str = 'answers'):
        self.answers_path = Path(answers_path)
        self.insider_periods = []
        
        # --- (V3.5 CHECKPOINT) ---
        subset = config.DATASET_SUBSET if hasattr(config, 'DATASET_SUBSET') and config.DATASET_SUBSET else []
        subset_name = "_".join(subset) if subset else "ALL"
        
        self.input_file_path = config.PROCESSED_DATA_DIR / f'processed_unified_logs_{subset_name}_ENRICHED.csv'
        self.output_file_path = config.PROCESSED_DATA_DIR / f'processed_unified_logs_{subset_name}_LABELED.csv'
        # --- (END V3.5 CHECKPOINT) ---

    def load_answers(self):
        """Load the master insiders.csv file"""
        logger.info("Loading CMU-CERT master answers file (insiders.csv)...")
        
        insiders_file = self.answers_path / 'insiders.csv'
        if not insiders_file.exists():
            raise FileNotFoundError(f"Could not find insiders.csv in {self.answers_path}")
        
        insiders_df = pd.read_csv(insiders_file)
        logger.info(f"Loaded {len(insiders_df)} total insider incidents")
        
        # Parse time periods and create a list of rules
        for _, row in insiders_df.iterrows():
            try:
                # --- (V2.3 DATASET FIX) ---
                # This is the new, correct logic
                dataset_id_raw = str(row['dataset']) # e.g., '2.0' or '3.1-1'
                
                # '3.1-1' -> '3.1'
                dataset_id_normalized = dataset_id_raw.split('-')[0]
                
                # '2.0' -> 'r2'
                # '3.1' -> 'r3.1'
                dataset_id_final = "r" + dataset_id_normalized.replace(".0", "")
                # --- (END V2.3 FIX) ---

                self.insider_periods.append({
                    'user_id': str(row['user']), # Ensure user is string
                    'dataset': dataset_id_final, # Use the new, correct ID
                    'start': pd.to_datetime(row['start']),
                    'end': pd.to_datetime(row['end']),
                })
            except Exception as e:
                logger.warning(f"Could not parse dates for {row['user']} in {row['dataset']}: {e}")
        
        logger.info(f"Parsed and normalized {len(self.insider_periods)} insider activity rules")
        
        # Log a sample of rules for debugging
        logger.info("Sample of normalized rules:")
        for rule in self.insider_periods[:3]:
            logger.info(f"  - User: {rule['user_id']}, Dataset: {rule['dataset']}, Start: {rule['start'].date()}")
        if len(self.insider_periods) > 3:
             logger.info(f"  ...and {len(self.insider_periods)-3} more.")


    def label_data_in_chunks(self, chunk_size=1000000):
        """
        Loads the master processed data in chunks, applies temporal labels,
        and saves to a new file.
        """
        logger.info(utils.generate_report_header("LABELING INSIDER THREATS (V2: MULTI-DATASET)"))
        
        if not self.input_file_path.exists():
            raise FileNotFoundError(f"Master file not found: {self.input_file_path}\n"
                                    "Please run 'data_preprocessing_v2.py' first.")

        # 1. Load answers
        self.load_answers()
        
        # Create a lookup for rules to speed up matching
        rules_lookup = {}
        for p in self.insider_periods:
            key = (p['user_id'], p['dataset'])
            if key not in rules_lookup:
                rules_lookup[key] = []
            rules_lookup[key].append((p['start'], p['end']))
        
        # Log the rules we're looking for
        logger.info(f"Loaded {len(rules_lookup)} unique (user, dataset) rule combinations.")
        logger.info(f"Example rule keys: {list(rules_lookup.keys())[:5]}")

        # 2. Define temp output file
        temp_output_file = self.output_file_path.with_suffix('.tmp')
        
        total_rows_processed = 0
        total_labeled_records = 0
        header_written = False

        try:
            # 3. Process file in chunks
            with pd.read_csv(self.input_file_path, chunksize=chunk_size, low_memory=False) as reader:
                for i, chunk in enumerate(reader):
                    logger.info(f"Processing chunk {i+1} ({len(chunk)} rows)...")
                    
                    # Ensure correct dtypes for matching
                    chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
                    chunk['user_id'] = chunk['user_id'].astype(str)
                    chunk['dataset'] = chunk['dataset'].astype(str)
                    
                    if 'is_insider' not in chunk.columns:
                        chunk['is_insider'] = 0
                    
                    # Create an empty mask to hold all positive labels
                    final_insider_mask = pd.Series(False, index=chunk.index)
                    
                    # Find all unique user/dataset pairs in this chunk
                    chunk_pairs = chunk[['user_id', 'dataset']].drop_duplicates()
                    
                    for _, row in chunk_pairs.iterrows():
                        key = (row['user_id'], row['dataset'])
                        
                        # Check if this user/dataset pair has any rules
                        if key in rules_lookup:
                            # This user is a potential insider in this dataset
                            # Get all rules for them
                            periods = rules_lookup[key]
                            
                            # Find all rows in the chunk matching this user/dataset
                            user_dataset_mask = (chunk['user_id'] == key[0]) & (chunk['dataset'] == key[1])
                            
                            # For these rows, check the date
                            for start, end in periods:
                                date_mask = (chunk['date'] >= start) & (chunk['date'] <= end)
                                
                                # Combine all masks
                                insider_mask = user_dataset_mask & date_mask
                                # Add these rows to our final mask
                                final_insider_mask = final_insider_mask | insider_mask

                    # Apply all labels found in this chunk at once
                    chunk.loc[final_insider_mask, 'is_insider'] = 1
                    
                    chunk_labeled_count = final_insider_mask.sum()
                    if chunk_labeled_count > 0:
                        logger.info(f"  > Labeled {chunk_labeled_count} records in this chunk.")
                        total_labeled_records += chunk_labeled_count
                    
                    # 4. Save processed chunk to temp file
                    if not header_written:
                        chunk.to_csv(temp_output_file, index=False, mode='w')
                        header_written = True
                    else:
                        chunk.to_csv(temp_output_file, index=False, mode='a', header=False)
                    
                    total_rows_processed += len(chunk)

            logger.info(f"All chunks processed. Total rows: {total_rows_processed}")
            
            if total_labeled_records == 0:
                logger.warning("="*80)
                logger.warning("WARNING: Total insider records labeled is 0.")
                logger.warning("This means no matches were found between insiders.csv and your data.")
                logger.warning("Please check your 'answers/insiders.csv' file and dataset subset.")
                logger.warning("="*80)
            else:
                 logger.info(f"Total insider records labeled: {total_labeled_records}")

            # 5. Safely move the temp file to the *new* output path
            shutil.move(temp_output_file, self.output_file_path)
            logger.info(f"Successfully created: {self.output_file_path}")

        except Exception as e:
            logger.error(f"Error during chunked labeling: {e}")
            if temp_output_file.exists():
                os.remove(temp_output_file)
            import traceback
            traceback.print_exc()
        
        finally:
            if temp_output_file.exists():
                logger.warning(f"Temp file was not removed: {temp_output_file}")


def main():
    labeler = InsiderLabelerV2(answers_path='answers')
    labeler.label_data_in_chunks()

    logger.info("="*80)
    logger.info("V2 Labeling Complete!")
    # Get the dynamic output file name for the log
    subset = config.DATASET_SUBSET if hasattr(config, 'DATASET_SUBSET') and config.DATASET_SUBSET else []
    subset_name = "_".join(subset) if subset else "ALL"
    output_filename = f'processed_unified_logs_{subset_name}_LABELED.csv'
    
    logger.info(f"The file '{output_filename}' is now labeled.")
    logger.info("Next step: Run 'feature_engineering_v2.py'")
    logger.info("="*80)

if __name__ == "__main__":
    main()