"""
Label insider threats using CMU-CERT answers file
Reads insiders.csv and marks the corresponding users as malicious

[--- CHUNKED & PSEUDONYMIZATION-AWARE VERSION ---]
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
        """
        Initialize the labeler
        
        Args:
            answers_path: Path to the answers directory/zip
        """
        self.answers_path = Path(answers_path)
        self.insiders_df = None
        self.insider_users = set()
        self.insider_periods = []  # List of (user, start_date, end_date) tuples
        
        # --- NEW ---
        # Create a set of pseudonymized insider IDs for fast lookup
        self.pseudo_insider_users = set()
        self.pseudo_insider_periods = []
        
    def load_answers(self):
        """Load the insiders.csv file"""
        logger.info("Loading CMU-CERT answers file...")
        
        # Try to find insiders.csv
        insiders_file = self.answers_path / 'insiders.csv'
        
        if not insiders_file.exists():
            # Try without path
            insiders_file = Path('insiders.csv')
        
        if not insiders_file.exists():
            raise FileNotFoundError(
                f"Could not find insiders.csv in {self.answers_path}\n"
                "Please make sure you've extracted answers.zip"
            )
        
        # Read insiders.csv
        self.insiders_df = pd.read_csv(insiders_file)
        
        logger.info(f"Loaded {len(self.insiders_df)} insider incidents")
        
        # Extract unique insider users
        self.insider_users = set(self.insiders_df['user'].unique())
        
        # --- NEW: Create pseudonymized sets ---
        self.pseudo_insider_users = {utils.pseudonymize_identifier(u) for u in self.insider_users}
        
        logger.info(f"Identified {len(self.insider_users)} unique insider users:")
        for user in sorted(self.insider_users):
            logger.info(f"  - {user} (Pseudo: {utils.pseudonymize_identifier(user)})")
        
        # Parse time periods for each insider
        for _, row in self.insiders_df.iterrows():
            try:
                # Parse dates - handle different formats
                start_date = pd.to_datetime(row['start'])
                end_date = pd.to_datetime(row['end'])
                
                # Original period
                self.insider_periods.append({
                    'user': row['user'],
                    'start': start_date,
                    'end': end_date,
                    'scenario': row['scenario'],
                    'dataset': row['dataset']
                })
                
                # --- NEW: Pseudonymized period ---
                self.pseudo_insider_periods.append({
                    'user_pseudo': utils.pseudonymize_identifier(row['user']),
                    'start': start_date,
                    'end': end_date,
                    'scenario': row['scenario'],
                    'dataset': row['dataset']
                })
                
            except Exception as e:
                logger.warning(f"Could not parse dates for {row['user']}: {e}")
        
        logger.info(f"Parsed {len(self.insider_periods)} insider activity periods")
        
        return self.insider_users
    
    def label_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simple labeling: Mark any activity by insider users as malicious
        
        Args:
            df: Dataframe with user activity
            
        Returns:
            Dataframe with 'is_insider' column
        """
        logger.info("Applying simple labeling (all activity from insider users)...")
        
        # Identify user column
        user_col = None
        if 'user_pseudo' in df.columns:
            user_col = 'user_pseudo'
        elif 'user' in df.columns:
            user_col = 'user'
        
        if user_col is None:
            raise ValueError("No user column found in dataframe")
        
        # Initialize label column if it doesn't exist (it should)
        if 'is_insider' not in df.columns:
            df['is_insider'] = 0
        
        # --- FIX: Use the correct list for comparison ---
        if user_col == 'user_pseudo':
            mask = df[user_col].isin(self.pseudo_insider_users)
        else:
            mask = df[user_col].isin(self.insider_users)
            
        n_marked = mask.sum()
        if n_marked > 0:
            df.loc[mask, 'is_insider'] = 1
            logger.info(f"  Labeled {n_marked:,} records for insiders in this chunk")
        
        return df
    
    def label_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Temporal labeling: Only mark activity during the attack period
        More precise but requires date column
        
        Args:
            df: Dataframe with user activity and date column
            
        Returns:
            Dataframe with 'is_insider' column
        """
        # logger.info("Applying temporal labeling (only during attack periods)...")
        
        # Check for date column
        if 'date' not in df.columns:
            logger.warning("No 'date' column found, falling back to simple labeling")
            return self.label_simple(df)
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Identify user column
        user_col = None
        if 'user_pseudo' in df.columns:
            user_col = 'user_pseudo'
        elif 'user' in df.columns:
            user_col = 'user'
        
        if user_col is None:
            raise ValueError("No user column found in dataframe")
        
        # Initialize label column if it doesn't exist
        if 'is_insider' not in df.columns:
            df['is_insider'] = 0
        
        # --- FIX: Use the correct period list for comparison ---
        if user_col == 'user_pseudo':
            periods_to_check = self.pseudo_insider_periods
            user_key = 'user_pseudo'
        else:
            periods_to_check = self.insider_periods
            user_key = 'user'

        # Mark each insider period
        for period in periods_to_check:
            insider_user = period[user_key]
            start_date = period['start']
            end_date = period['end']
            
            user_mask = df[user_col] == insider_user
            
            # Date range mask
            date_mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            
            # Combine masks
            insider_mask = user_mask & date_mask
            
            n_marked = insider_mask.sum()
            if n_marked > 0:
                df.loc[insider_mask, 'is_insider'] = 1
                logger.info(
                    f"  {insider_user}: Labeled {n_marked:,} records "
                    f"({start_date.date()} to {end_date.date()})"
                )
        
        return df
    
    def label_processed_data(self, method: str = 'temporal'):
        """
        Label the processed unified logs
        
        Args:
            method: 'simple' or 'temporal'
        """
        logger.info(utils.generate_report_header("LABELING INSIDER THREATS"))
        
        # Load answers
        self.load_answers()
        
        # Load processed data
        processed_file = config.PROCESSED_DATA_DIR / 'processed_unified_logs.csv'
        
        if not processed_file.exists():
            raise FileNotFoundError(
                f"Processed data not found: {processed_file}\n"
                "Run data_preprocessing.py first"
            )
        
        # --- FIX: Process in chunks to avoid memory errors ---
        chunk_size = config.PERFORMANCE.get('chunk_size', 50000)
        chunks = []
        total_rows = 0
        total_insider_rows = 0
        
        logger.info(f"Loading and labeling {processed_file} in chunks of {chunk_size}...")
        
        for chunk in pd.read_csv(processed_file, chunksize=chunk_size):
            total_rows += len(chunk)
            logger.info(f"  Processing chunk (Rows {total_rows - len(chunk)} to {total_rows})...")
            
            # Apply labeling
            if method == 'temporal':
                chunk = self.label_temporal(chunk)
            else:
                chunk = self.label_simple(chunk)
            
            total_insider_rows += (chunk['is_insider'] == 1).sum()
            chunks.append(chunk)
        
        if not chunks:
            logger.error("No data was loaded or processed.")
            return

        logger.info("Re-assembling labeled chunks...")
        df = pd.concat(chunks, ignore_index=True)
        # --- END FIX ---
        
        # Save labeled data (overwrite the file)
        logger.info(f"Saving fully labeled data back to {processed_file}...")
        df.to_csv(processed_file, index=False)
        logger.info(f"\nLabeled data saved to {processed_file}")
        
        # Create summary report
        self.create_summary_report(df)
        
        return df
    
    def create_summary_report(self, df: pd.DataFrame):
        """Create a summary report of insider labels"""
        logger.info("\n" + "="*80)
        logger.info("INSIDER LABELING SUMMARY")
        logger.info("="*80)
        
        user_col = 'user_pseudo' if 'user_pseudo' in df.columns else 'user'
        
        # Per-user breakdown
        if user_col in df.columns:
            logger.info("\nInsider Activity by User:")
            insider_data = df[df['is_insider'] == 1]
            
            if len(insider_data) > 0:
                user_stats = insider_data.groupby(user_col).agg({
                    'is_insider': 'count',
                    'activity_type': lambda x: x.value_counts().to_dict() if 'activity_type' in df.columns else {}
                })
                
                for user in user_stats.index:
                    count = user_stats.loc[user, 'is_insider']
                    logger.info(f"  {user}: {count:,} malicious records")
        
        # Activity type breakdown
        if 'activity_type' in df.columns:
            logger.info("\nInsider Activity by Type:")
            insider_by_type = df[df['is_insider'] == 1]['activity_type'].value_counts()
            for activity_type, count in insider_by_type.items():
                logger.info(f"  {activity_type}: {count:,}")
        
        # Dataset quality metrics
        total = len(df)
        insider_count = (df['is_insider'] == 1).sum()
        normal_count = (df['is_insider'] == 0).sum()
        
        logger.info(f"\nDataset Balance:")
        logger.info(f"  Total records:    {total:,}")
        logger.info(f"  Normal (0):       {normal_count:,} ({normal_count/total*100:.2f}%)")
        logger.info(f"  Insider (1):      {insider_count:,} ({insider_count/total*100:.4f}%)")
        
        if insider_count > 0:
            logger.info(f"  Imbalance ratio:  1:{normal_count/insider_count:.0f}")
        else:
            logger.warning("  NO INSIDER RECORDS WERE LABELED. Check your 'answers' files and user IDs.")
        
        logger.info("\n" + "="*80)
        
        # Save summary to file
        summary_file = config.RESULTS_DIR / 'insider_labeling_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("INSIDER LABELING SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total records: {total:,}\n")
            f.write(f"Normal: {normal_count:,} ({normal_count/total*100:.2f}%)\n")
            f.write(f"Insider: {insider_count:,} ({insider_count/total*100:.4f}%)\n")
            f.write(f"\nInsider Users:\n")
            for user in sorted(self.insider_users):
                f.write(f"  - {user}\n")
        
        logger.info(f"Summary saved to {summary_file}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Label insider threats using CMU-CERT answers')
    parser.add_argument('--answers-path', default='answers', help='Path to answers directory')
    parser.add_argument('--method', choices=['simple', 'temporal'], default='temporal',
                       help='Labeling method: simple (all user activity) or temporal (only during attack)')
    
    args = parser.parse_args()
    
    try:
        labeler = InsiderLabeler(args.answers_path)
        df = labeler.label_processed_data(method=args.method)
        
        if (df['is_insider'] == 1).sum() == 0:
            print("\n" + "="*80)
            print("⚠️ WARNING: Labeling complete, but 0 insider records were found.")
            print("="*80)
            print("\nThis likely means the users in 'insiders.csv' (like ONS0995)")
            print("do not match the users in your 'data/raw/' files.")
            print("Please double-check you have the correct dataset (r2) and answers.")
        else:
            print("\n" + "="*80)
            print("✅ SUCCESS! Your data is now labeled with insider threats.")
            print("="*80)
        
        print("\nNext steps:")
        print("  1. Validate the labels:")
        print("     python validate_labels.py")
        print("\n  2. Re-run feature engineering:")
        print("     python feature_engineering.py")
        print("\n  3. Re-train models:")
        print("     python main.py --train")
        print("\n  4. Evaluate with proper metrics:")
        print("     python main.py --evaluate")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*80)
        print("❌ ERROR")
        print("="*80)
        print("\nTroubleshooting:")
        print("  1. Make sure you've extracted answers.zip")
        print("  2. Check that insiders.csv is in the answers/ directory")
        print("  3. Run data_preprocessing.py first if you haven't")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()