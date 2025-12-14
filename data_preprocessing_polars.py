"""
Data Preprocessing Pipeline (Polars Version)
Optimized for performance and memory usage using Polars LazyFrames and Streaming.

Replaces: data_preprocessing_v2.py
"""

import polars as pl
import pandas as pd  # For reading insider ground truth
import config
import utils
from pathlib import Path
import os
import shutil
from datetime import datetime

logger = utils.logger

class DataPreprocessorPolars:
    def __init__(self):
        self.raw_data_dir = config.RAW_DATA_DIR
        self.processed_dir = config.PROCESSED_DATA_DIR
        
        # Determine datasets to process
        self.dataset_folders = []
        if config.DATASET_SUBSET:
            logger.info(f"Processing SUBSET defined in config: {config.DATASET_SUBSET}")
            for d_name in config.DATASET_SUBSET:
                d_path = self.raw_data_dir / d_name
                if d_path.is_dir():
                    self.dataset_folders.append(d_path)
                else:
                    logger.warning(f"Dataset folder '{d_name}' not found.")
        else:
            self.dataset_folders = sorted([d for d in self.raw_data_dir.iterdir() if d.is_dir() and d.name.startswith('r')])
            
        # Output paths
        subset_name = "_".join(config.DATASET_SUBSET) if config.DATASET_SUBSET else "ALL"
        self.final_file_path = self.processed_dir / f'processed_unified_logs_{subset_name}_ENRICHED.csv'
        
        # Schema definitions (to ensure type consistency across chunks)
        self.common_schema = {
            'date': pl.Utf8, # Parse later
            'user': pl.Utf8,
            'pc': pl.Utf8,
            'activity': pl.Utf8,
            'id': pl.Utf8, # Various IDs
            'to': pl.Utf8,
            'from': pl.Utf8,
            'cc': pl.Utf8,
            'bcc': pl.Utf8,
            'size': pl.Int64,
            'attachments': pl.Utf8,
            'url': pl.Utf8,
            'content': pl.Utf8,
            'filename': pl.Utf8
        }
        
        # Load insider ground truth
        self.insider_users, self.insider_windows = self._load_insider_labels()

    def _load_insider_labels(self):
        """
        Load insider user IDs and their malicious time windows from answers/insiders.csv.
        
        Returns:
            insider_users: set of user IDs who are insiders
            insider_windows: dict mapping user_id -> list of (start, end) datetime tuples
        """
        # answers folder is at same level as data folder
        answers_dir = self.raw_data_dir.parent.parent / 'answers'
        insiders_path = answers_dir / 'insiders.csv'
        
        insider_users = set()
        insider_windows = {}
        
        if not insiders_path.exists():
            logger.warning(f"Insider labels not found at {insiders_path}")
            return insider_users, insider_windows
        
        try:
            df = pd.read_csv(insiders_path)
            
            # Filter by datasets we're processing
            dataset_prefixes = [d.name for d in self.dataset_folders]
            
            for _, row in df.iterrows():
                # Check if this insider belongs to a dataset we're processing
                dataset_str = str(row['dataset'])
                
                # Match dataset (e.g., '4.1' matches 'r4.1')
                matches_dataset = any(
                    dataset_str == d.replace('r', '') or dataset_str == d 
                    for d in dataset_prefixes
                )
                
                if not matches_dataset:
                    continue
                    
                user = row['user']
                insider_users.add(user)
                
                # Parse time windows
                try:
                    start = pd.to_datetime(row['start'])
                    end = pd.to_datetime(row['end'])
                    
                    if user not in insider_windows:
                        insider_windows[user] = []
                    insider_windows[user].append((start, end))
                except Exception as e:
                    logger.warning(f"Could not parse dates for user {user}: {e}")
            
            logger.info(f"✓ Loaded {len(insider_users)} insider users from ground truth")
            if insider_users:
                logger.info(f"  Insiders: {list(insider_users)[:10]}{'...' if len(insider_users) > 10 else ''}")
                # Log window details
                total_windows = sum(len(w) for w in insider_windows.values())
                logger.info(f"  Total malicious time windows: {total_windows}")
            
        except Exception as e:
            logger.error(f"Failed to load insider labels: {e}")
        
        return insider_users, insider_windows

    def _create_insider_windows_df(self):
        """
        Create a Polars DataFrame with insider time windows for efficient joining.
        
        Returns:
            pl.DataFrame with columns: user_id, window_start, window_end
        """
        if not self.insider_windows:
            return None
            
        rows = []
        for user, windows in self.insider_windows.items():
            for start, end in windows:
                rows.append({
                    'user_id': user,
                    'window_start': start,
                    'window_end': end
                })
        
        windows_df = pl.DataFrame(rows)
        windows_df = windows_df.with_columns([
            pl.col('window_start').cast(pl.Datetime),
            pl.col('window_end').cast(pl.Datetime)
        ])
        
        logger.info(f"Created insider windows DataFrame with {len(rows)} entries")
        return windows_df

    def _get_start_options(self, file_path: Path, log_type: str) -> tuple[bool, list[str] | None]:
        """
        Determines if a CSV has headers and returns (has_header, new_columns).
        Heuristic: If first line starts with '{', it's data (headerless).
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                
            if first_line.startswith('{'):
                # Headerless
                if log_type == 'http':
                    return False, ['id', 'date', 'user_id', 'pc_id', 'url']
                elif log_type == 'logon':
                    return False, ['id', 'date', 'user_id', 'pc_id', 'activity']
                elif log_type == 'device':
                    return False, ['id', 'date', 'user_id', 'pc_id', 'activity']
                elif log_type == 'email':
                    # Guessing common format for headerless email if encountered
                    return False, ['id', 'date', 'user_id', 'pc_id', 'to', 'cc', 'bcc', 'from', 'activity', 'size', 'attachments', 'content']
                elif log_type == 'file':
                     return False, ['id', 'date', 'user_id', 'pc_id', 'filename', 'content']
                else:
                    # Fallback
                    return False, None
            else:
                return True, None
        except Exception as e:
            logger.warning(f"Could not read file head for {file_path}: {e}")
            return True, None

    def _process_log_file(self, file_path: Path, log_type: str, dataset_id: str) -> pl.LazyFrame:
        """
        Lazily reads and processes a single log file.
        """
        if not file_path.exists():
            return None
            
        try:
            # Check for headers
            has_header, new_columns = self._get_start_options(file_path, log_type)
            
            # Lazy read
            if new_columns:
                q = pl.scan_csv(file_path, has_header=has_header, new_columns=new_columns, ignore_errors=True, infer_schema_length=10000)
            else:
                q = pl.scan_csv(file_path, has_header=has_header, ignore_errors=True, infer_schema_length=10000)
            
            # Standardize columns
            # 1. Rename user/pc columns if they weren't set by new_columns
            if 'user' in q.columns:
                q = q.rename({'user': 'user_id'})
            elif 'from' in q.columns and log_type == 'email':
                q = q.rename({'from': 'user_id'})
            elif 'user_id' not in q.columns:
                 q = q.with_columns(pl.lit('UNKNOWN').alias('user_id'))
                 
            if 'pc' in q.columns:
                q = q.rename({'pc': 'pc_id'})
            elif 'pc_id' not in q.columns:
                q = q.with_columns(pl.lit(None).cast(pl.Utf8).alias('pc_id'))

            # 2. Add metadata
            q = q.with_columns([
                pl.lit(log_type).alias('activity_type'),
                pl.lit(dataset_id).alias('dataset')
            ])
            
            # 3. Date parsing - Robust Strategy
            # Try MM/DD/YYYY HH:MM:SS first (common in r4.2), then YYYY-MM-DD
            q = q.with_columns(
                pl.col('date').str.strptime(pl.Datetime, format='%m/%d/%Y %H:%M:%S', strict=False).alias('date_fmt1')
            )
            q = q.with_columns(
                 pl.coalesce([
                     pl.col('date_fmt1'),
                     pl.col('date').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S', strict=False)
                 ]).alias('date_parsed')
            ).drop(['date', 'date_fmt1']).rename({'date_parsed': 'date'})

            # 4. Feature Extraction based on log type
            if log_type == 'logon':
                if 'activity' in q.columns:
                   q = q.with_columns([
                       (pl.col('activity') == 'Logon').cast(pl.Int8).alias('is_logon'),
                       (pl.col('activity') == 'Logoff').cast(pl.Int8).alias('is_logoff')
                   ])
            
            elif log_type == 'device':
                if 'activity' in q.columns:
                    q = q.with_columns([
                        (pl.col('activity') == 'Connect').cast(pl.Int8).alias('is_connect'),
                        (pl.col('activity') == 'Disconnect').cast(pl.Int8).alias('is_disconnect')
                    ])
                
            elif log_type == 'email':
                # Check column existence before using
                exprs = []
                if 'size' in q.columns:
                     exprs.append((pl.col('size') / 1024).alias('email_size_kb'))
                     exprs.append((pl.col('size') > 1024*1024).cast(pl.Int8).alias('is_large_email'))
                else:
                     exprs.append(pl.lit(0.0).alias('email_size_kb'))
                     exprs.append(pl.lit(0).cast(pl.Int8).alias('is_large_email'))
                     
                if 'attachments' in q.columns:
                    exprs.append(pl.col('attachments').is_not_null().cast(pl.Int8).alias('has_attachment'))
                else:
                    exprs.append(pl.lit(0).cast(pl.Int8).alias('has_attachment'))
                    
                q = q.with_columns(exprs)
                
            elif log_type == 'http':
                if 'url' in q.columns:
                    # Domain extraction
                    q = q.with_columns(
                        pl.col('url').str.extract(r'https?://([^/]+)', 1).alias('domain')
                    )
                    suspicious_keywords = ['temp', 'anonymous', 'proxy', 'vpn', 'leak', 'hack']
                    pattern = "|".join(suspicious_keywords)
                    q = q.with_columns(
                        pl.col('domain').str.to_lowercase().str.contains(pattern).cast(pl.Int8).alias('is_suspicious_domain')
                    )

            elif log_type == 'file':
                if 'filename' in q.columns:
                    q = q.with_columns(
                        pl.col('filename').str.split('.').list.last().str.to_lowercase().alias('file_extension')
                    )
                    sensitive_exts = ['doc', 'docx', 'xls', 'xlsx', 'pdf', 'txt', 'sql', 'db', 'zip', 'rar']
                    q = q.with_columns(
                        pl.col('file_extension').is_in(sensitive_exts).cast(pl.Int8).alias('is_sensitive_file')
                    )

            return q

        except Exception as e:
            logger.error(f"Error preparing query for {file_path}: {e}")
            return None

    def _load_static_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Loads LDAP and Psychometric data into DataFrames (small enough for memory).
        """
        # LDAP
        ldap_df = pl.DataFrame()
        try:
            ldap_path = self.raw_data_dir / config.LDAP_PATH
            if ldap_path.is_dir():
                csv_files = list(ldap_path.glob("*.csv"))
                if csv_files:
                    # Scan all LDAP CSVs and correct columns
                    lazy_ldaps = []
                    for f in csv_files:
                         q = pl.scan_csv(f)
                         # Select only relevant columns if they exist
                         if 'role' in q.columns:
                             q = q.select(['user_id', 'role', 'domain'] if 'domain' in q.columns else ['user_id', 'role'])
                             lazy_ldaps.append(q)
                    
                    if lazy_ldaps:
                        ldap_df = pl.concat(lazy_ldaps).unique(subset=['user_id'], keep='last').collect()
                        logger.info(f"Loaded LDAP data: {len(ldap_df)} users")
        except Exception as e:
            logger.error(f"Failed to load LDAP: {e}")

        # Psychometric
        psych_df = pl.DataFrame()
        try:
            # Helper to find psychometric files across dataset folders
            psych_files = []
            for d in self.dataset_folders:
                p_file = d / 'psychometric.csv'
                if p_file.exists():
                    psych_files.append(p_file)
            
            if psych_files:
                q_list = [pl.scan_csv(f) for f in psych_files]
                # Concat and dedup
                psych_df = pl.concat(q_list).unique(subset=['user_id'], keep='first').collect()
                logger.info(f"Loaded Psychometric data: {len(psych_df)} users")
                
        except Exception as e:
            logger.error(f"Failed to load psychometric data: {e}")

        return ldap_df, psych_df

    def run_pipeline(self):
        logger.info(utils.generate_report_header("DATA PREPROCESSING (POLARS)"))
        
        lazy_frames = []
        
        # 1. Build LazyFrames for all log files
        for folder in self.dataset_folders:
            dataset_id = folder.name
            logger.info(f"Scanning dataset: {dataset_id}")
            
            for log_type, filename in config.LOG_FILENAMES.items():
                file_path = folder / filename
                lf = self._process_log_file(file_path, log_type, dataset_id)
                if lf is not None:
                    lazy_frames.append(lf)

        if not lazy_frames:
            logger.error("No data found to process.")
            return

        # 2. Configure the Unified LazyFrame
        # Align columns locally before concat? Polars strict concat might fail if columns differ.
        # We need a robust way to ensure all frames have the same columns.
        # Strategy: Diagonal concat (fills missing with nulls)
        
        logger.info("Constructing execution graph (Diagonal Concat)...")
        unified_lf = pl.concat(lazy_frames, how="diagonal")
        
        # 3. Load Static Data (Eager)
        ldap_df, psych_df = self._load_static_data()
        
        # 4. Join Static Data
        # Since static data is small, we can join it to the large LazyFrame
        if not ldap_df.is_empty():
            # Convert to Lazy for joining
            unified_lf = unified_lf.join(ldap_df.lazy(), on='user_id', how='left')
            
        if not psych_df.is_empty():
            # Select relevant columns
            cols = ['user_id', 'O', 'C', 'E', 'A', 'N']
            p_lazy = psych_df.select([c for c in cols if c in psych_df.columns]).lazy()
            unified_lf = unified_lf.join(p_lazy, on='user_id', how='left')

        # 5. Fill NaNs & Final Cleanup
        # Ensure 'role' and 'domain' exist (from LDAP)
        if 'role' not in unified_lf.columns:
            unified_lf = unified_lf.with_columns(pl.lit('UNKNOWN').alias('role'))
        else:
             unified_lf = unified_lf.with_columns(pl.col('role').fill_null('UNKNOWN'))
             
        if 'domain' not in unified_lf.columns:
             # domain from LDAP, distinct from url domain? 
             # In _process_log_file for http, we create 'domain'. 
             # LDAP domain is usually 'functional unit' or similar. 
             # If collision, Polars suffix?
             # _process_log_file 'domain' is from URL. 
             # LDAP 'domain' might conflict. 
             # Checks: _load_static_data selects ['user_id', 'role', 'domain'].
             # If we join, we might have duplicate 'domain' if http logs have it.
             # But http logs only exist in 'http' rows.
             # The join is on user_id.
             # We should probably rename LDAP domain to 'user_domain' or similar if needed.
             pass
        
        # Ensure Psychometric columns exist
        for c in ['O', 'C', 'E', 'A', 'N']:
            if c not in unified_lf.columns:
                unified_lf = unified_lf.with_columns(pl.lit(0).alias(c))
            else:
                 unified_lf = unified_lf.with_columns(pl.col(c).fill_null(0))

        # Define fill values
        fill_values = {
            'is_logon': 0, 'is_logoff': 0, 'is_connect': 0, 'is_disconnect': 0,
            'is_large_email': 0, 'has_attachment': 0, 'is_suspicious_domain': 0, 'is_sensitive_file': 0,
            'email_size_kb': 0.0
        }
        
        # Apply fills (efficiently using expressions)
        unified_lf = unified_lf.with_columns([
            pl.col(c).fill_null(v) for c, v in fill_values.items() if c in unified_lf.collect_schema().names()
        ])
        
        # 6. Mark insider records based on ground truth (TIME-WINDOW BASED)
        # Only mark records as insider=1 if user_id matches AND timestamp is within malicious window
        if self.insider_windows:
            windows_df = self._create_insider_windows_df()
            
            if windows_df is not None:
                logger.info("Applying time-window based insider labeling...")
                
                # Strategy: For each record, check if it matches any insider window
                # This is done by joining and filtering on time range
                
                # First, add default is_insider=0
                unified_lf = unified_lf.with_columns(pl.lit(0).alias('is_insider'))
                
                # Collect to DataFrame for the join (required for cross-join filtering)
                # This is memory-intensive but necessary for accurate labeling
                try:
                    logger.info("Collecting data for time-window labeling (this may take a while)...")
                    unified_df = unified_lf.collect(streaming=True)
                    
                    # Create the insider marking using a join approach
                    # For each insider user's window, mark matching records
                    for user, windows in self.insider_windows.items():
                        for start, end in windows:
                            # Create mask for this window
                            mask = (
                                (unified_df['user_id'] == user) & 
                                (unified_df['date'] >= start) & 
                                (unified_df['date'] <= end)
                            )
                            # Update is_insider for matching rows
                            unified_df = unified_df.with_columns(
                                pl.when(mask)
                                .then(pl.lit(1))
                                .otherwise(pl.col('is_insider'))
                                .alias('is_insider')
                            )
                    
                    # Count labeled records
                    insider_count = unified_df.filter(pl.col('is_insider') == 1).height
                    total_count = unified_df.height
                    logger.info(f"✓ Labeled {insider_count:,} records as insider ({100*insider_count/total_count:.4f}% of {total_count:,} total)")
                    
                    # Convert back to LazyFrame for streaming output
                    unified_lf = unified_df.lazy()
                    
                except Exception as e:
                    logger.error(f"Time-window labeling failed: {e}. Falling back to user-only labeling.")
                    import traceback
                    traceback.print_exc()
                    # Fallback: mark all records for insider users
                    insider_list = list(self.insider_users)
                    unified_lf = unified_lf.with_columns(
                        pl.when(pl.col('user_id').is_in(insider_list))
                        .then(pl.lit(1))
                        .otherwise(pl.lit(0))
                        .alias('is_insider')
                    )
            else:
                logger.warning("No insider windows created - all records will have is_insider=0")
                unified_lf = unified_lf.with_columns(pl.lit(0).alias('is_insider'))
        else:
            logger.warning("No insider labels loaded - all records will have is_insider=0")
            unified_lf = unified_lf.with_columns(pl.lit(0).alias('is_insider'))

        # 7. Execute and Stream to CSV
        output_file = self.processed_dir / "processed_unified_logs.csv"
        logger.info(f"Streaming merged data to {output_file}...")
        
        # Remove existing file if present
        if output_file.exists():
            output_file.unlink()
            
        # Sink to CSV
        try:
            unified_lf.sink_csv(output_file)
            logger.info("✓ Preprocessing complete.")
            
            # Log summary of insider labeling
            if self.insider_users:
                logger.info(f"  Insider users labeled: {list(self.insider_users)[:5]}{'...' if len(self.insider_users) > 5 else ''}")
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    preprocessor = DataPreprocessorPolars()
    preprocessor.run_pipeline()

if __name__ == "__main__":
    main()
