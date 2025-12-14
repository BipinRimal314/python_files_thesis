"""
Label insider threats using CMU-CERT answers file.
[--- UPDATED, SIMPLIFIED VERSION ---]
This version loads the entire file into memory to ensure
labels are applied correctly and avoids chunking I/O bugs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import config
import utils

logger = utils.logger

class InsiderLabeler:
    """
    Parse CMU-CERT answers and label insider threats in processed data
    """
    
    def __init__(self, answers_path: str = 'answers'):
        self.answers_path = Path(answers_path)
        self.insiders_df = None
        self.insider_periods = []
        
    def load_answers(self):
        """Load the insiders.csv file"""
        logger.info("Loading CMU-CERT answers file...")
        
        insiders_file = self.answers_path / 'insiders.csv'
        if not insiders_file.exists():
            raise FileNotFoundError(f"Could not find insiders.csv in {self.answers_path}")
        
        self.insiders_df = pd.read_csv(insiders_file)
        logger.info(f"Loaded {len(self.insiders_df)} insider incidents")
        
        # Parse time periods for each insider
        for _, row in self.insiders_df.iterrows():
            try:
                # Parse dates - handle different formats
                start_date = pd.to_datetime(row['start'])
                end_date = pd.to_datetime(row['end'])
                
                self.insider_periods.append({
                    'user': row['user'],
                    'start': start_date,
                    'end': end_date,
                    'scenario': row['scenario'],
                    'dataset': row['dataset']
                })
            except Exception as e:
                logger.warning(f"Could not parse dates for {row['user']}: {e}")
        
        logger.info(f"Parsed {len(self.insider_periods)} insider activity periods")

    def label_data(self):
        """
        Loads the processed data, applies temporal labels, and saves it.
        """
        logger.info(utils.generate_report_header("LABELING INSIDER THREATS"))
        
        # 1. Load answers
        self.load_answers()
        
        # 2. Load processed data
        processed_file = config.PROCESSED_DATA_DIR / 'processed_unified_logs.csv'
        if not processed_file.exists():
            raise FileNotFoundError(f"Processed data not found: {processed_file}")
        
        logger.info(f"Loading processed data from {processed_file}...")
        df = pd.read_csv(processed_file)
        logger.info(f"Loaded {len(df):,} records")
        
        # 3. Prepare for labeling
        if 'date' not in df.columns:
            raise ValueError("No 'date' column found in processed data.")
            
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        user_col = 'user_pseudo' if 'user_pseudo' in df.columns else 'user'
        if user_col not in df.columns:
            raise ValueError("No user or user_pseudo column found.")
            
        logger.info(f"Labeling based on '{user_col}' column")
        
        # Initialize label column
        df['is_insider'] = 0
        total_labeled_records = 0
        
        # 4. Apply temporal labels
        logger.info("Applying temporal labels (only during attack periods)...")
        for period in self.insider_periods:
            insider_user = period['user']
            start_date = period['start']
            end_date = period['end']
            
            # Get the correct pseudonym for the insider
            pseudo_user = utils.pseudonymize_identifier(insider_user)
            
            # Create masks
            user_mask = (df[user_col] == pseudo_user)
            date_mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            
            # Combine masks
            insider_mask = user_mask & date_mask
            
            n_marked = insider_mask.sum()
            if n_marked > 0:
                df.loc[insider_mask, 'is_insider'] = 1
                total_labeled_records += n_marked
                logger.info(
                    f"  LABELED {insider_user} ({pseudo_user}): {n_marked:,} records "
                    f"({start_date.date()} to {end_date.date()})"
                )
        
        # 5. Report
        total_insider = (df['is_insider'] == 1).sum()
        total_normal = (df['is_insider'] == 0).sum()
        
        logger.info(f"\nTemporal labeling complete:")
        logger.info(f"  Total records labeled: {total_insider:,}")
        
        if total_insider == 0:
            logger.error("="*80)
            logger.error("FATAL ERROR: No records were labeled.")
            logger.error("This means no insiders were found in the data matching the answer key.")
            logger.error("Please check your data and answer files.")
            logger.error("="*80)
        else:
            logger.info(f"  Normal records:   {total_normal:,} ({total_normal/len(df)*100:.2f}%)")
            logger.info(f"  Insider records:  {total_insider:,} ({total_insider/len(df)*100:.4f}%)")

        # 6. Save the labeled data back to disk
        logger.info(f"Saving labeled data back to {processed_file}...")
        df.to_csv(processed_file, index=False)
        logger.info("File save complete.")
        
        return df

def main():
    """Main execution"""
    try:
        labeler = InsiderLabeler(answers_path='answers')
        df = labeler.label_data()
        
        if (df['is_insider'] == 1).sum() > 0:
            print("\n" + "="*80)
            print("✅ SUCCESS! Your data is now correctly labeled.")
            print("="*80)
            print("\nNext steps:")
            print("  1. Re-run feature engineering:")
            print("     python feature_engineering.py")
            print("\n  2. Re-train models (one by one):")
            print("     python isolation_forest_model.py")
            print("     python deep_clustering_lightweight.py")
            print("     python lstm_autoencoder_lighweight.py")
            print("\n  3. Run the final evaluation:")
            print("     python model_evaluation.py")
            print("\n" + "="*80 + "\n")
        else:
            print("\n" + "="*80)
            print("❌ FAILED. No records were labeled.")
            print("="*80)
            print("\nPlease check the logs above. The script could not find")
            print("any of the insiders from 'insiders.csv' in your")
            print("'processed_unified_logs.csv' file.")
            print("\n" + "="*80 + "\n")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()