"""
Feature Engineering for Insider Threat Detection
[--- UPDATED VERSION ---]
- FIXES: KeyError: "Column(s) ['after_hours_count', ...] do not exist"
- Creates TWO separate feature sets:
    1. engineered_static_features.csv: (Users x Features) for IF/Clustering
    2. engineered_daily_features.csv: (User-Days x Features) for LSTM
- Correctly scales both datasets separately using two different scalers.
- Creates sequences from the scaled daily data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import config
import utils

logger = utils.logger

class FeatureEngineer:
    
    def __init__(self):
        """Initialize the feature engineer with two separate scalers."""
        self.static_scaler = StandardScaler() # For user-level static features
        self.daily_scaler = StandardScaler()  # For daily time-series features
        self.label_encoders = {}
        self.static_feature_names = []
        self.daily_feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from activity logs (hour, day_of_week, etc.)"""
        logger.info("Creating temporal features...")
        df = df.copy()
        if 'date' not in df.columns:
            logger.warning("No 'date' column found, skipping temporal features")
            return df
        
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
        
        logger.info("Temporal features created")
        return df
    
    def create_user_aggregated_features(self, df: pd.DataFrame, time_window: str = '1D') -> pd.DataFrame:
        """Create aggregated features per user over time windows (e.g., daily)"""
        logger.info(f"Creating user aggregated features with window: {time_window}")
        
        user_col = 'user_pseudo' if 'user_pseudo' in df.columns else 'user'
        
        if user_col not in df.columns or 'date' not in df.columns:
            logger.warning("Required columns not found for aggregation")
            return pd.DataFrame()
        
        df_temp = df.set_index('date')
        
        # Group by user and time window
        grouped = df_temp.groupby([pd.Grouper(freq=time_window), user_col])
        
        # --- Aggregation Logic (FIXED) ---
        # The dictionary keys *must* be existing columns from df_temp
        aggregations = {
            'activity_type': 'count'
        }
        
        # Source columns from data_preprocessing.py / create_temporal_features
        if 'is_weekend' in df_temp.columns:
            aggregations['is_weekend'] = 'max' # Was it a weekend day?
        if 'is_business_hours' in df_temp.columns:
            aggregations['is_business_hours'] = lambda x: (x == 0).sum() # Source col
        
        # Device features
        if 'is_connect' in df_temp.columns:
            aggregations['is_connect'] = 'sum' # Source col
            
        # Email features
        if 'has_attachment' in df_temp.columns:
            aggregations['has_attachment'] = 'sum' # Source col
        if 'email_size_kb' in df_temp.columns:
            # This creates a MultiIndex column, which we handle below
            aggregations['email_size_kb'] = ['sum', 'mean'] # Source col
            
        # HTTP features
        if 'is_suspicious_domain' in df_temp.columns:
            aggregations['is_suspicious_domain'] = 'sum' # Source col

        # Psychometric features (take the user's static score)
        psy_cols = ['O', 'C', 'E', 'A', 'N']
        for col in psy_cols:
            if col in df_temp.columns:
                aggregations[col] = 'mean' # 'mean', 'max', 'min' all work for a static value
                
        # --- Run Aggregation ---
        logger.info("Running aggregation...")
        agg_features = grouped.agg(aggregations)
        
        # --- Handle MultiIndex Columns (created by 'email_size_kb') ---
        new_cols = []
        for col in agg_features.columns:
            if isinstance(col, tuple):
                if col[0] == 'email_size_kb' and col[1] == 'sum':
                    new_cols.append('total_email_size_kb')
                elif col[0] == 'email_size_kb' and col[1] == 'mean':
                    new_cols.append('avg_email_size_kb')
                else:
                    new_cols.append(f"{col[0]}_{col[1]}") # Failsafe
            else:
                new_cols.append(col)
        
        agg_features.columns = new_cols
        
        # --- Rename other columns to their target names ---
        agg_features = agg_features.rename(columns={
            'activity_type': 'activity_count',
            'is_business_hours': 'after_hours_count',
            'is_connect': 'device_connects',
            'has_attachment': 'emails_with_attachments',
            'is_suspicious_domain': 'suspicious_http_count'
        })
        
        # Handle activity type distribution (can be slow)
        if 'activity_type' in df_temp.columns:
            logger.info("Aggregating activity type counts (this may be slow)...")
            activity_counts = grouped['activity_type'].value_counts().unstack(fill_value=0)
            activity_counts.columns = [f'count_{col}' for col in activity_counts.columns]
            agg_features = agg_features.join(activity_counts)
        
        agg_features = agg_features.reset_index().fillna(0)
        
        logger.info(f"Created {len(agg_features.columns)-2} aggregated daily features")
        return agg_features
    
    def create_statistical_features(self, df_daily: pd.DataFrame, user_col: str) -> pd.DataFrame:
        """Create statistical features per user (mean, std, etc. across all their days)"""
        logger.info("Creating statistical features per user...")
        
        # Select numeric columns (features) to aggregate
        numeric_cols = [col for col in df_daily.columns if col not in [user_col, 'date', 'is_insider']]
        
        if not numeric_cols:
            logger.warning("No numeric columns found for statistical features.")
            return pd.DataFrame(df_daily[user_col].unique(), columns=[user_col])

        # Group by user and calculate stats
        user_stats = df_daily.groupby(user_col)[numeric_cols].agg(['mean', 'std', 'min', 'max', 'median'])
        
        # Flatten column names
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
        user_stats = user_stats.reset_index()
        
        logger.info(f"Created {len(user_stats.columns)-1} statistical features")
        return user_stats
    
    def scale_features(self, df: pd.DataFrame, scaler_type: str, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features using the correct scaler (static or daily)"""
        
        scaler = self.static_scaler if scaler_type == 'static' else self.daily_scaler
        logger.info(f"Scaling {scaler_type} features (fit={fit})...")
        df = df.copy()
        
        # Select numeric columns (exclude labels and identifiers)
        user_col = 'user_pseudo' if 'user_pseudo' in df.columns else 'user'
        exclude_cols = ['is_insider', 'date', user_col, 'user', 'pc', 'user_pseudo']
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
        
        if not numeric_cols:
            logger.warning(f"No numeric columns found for {scaler_type} scaling")
            return df
        
        # Ensure scaler is fitted if not fitting now
        if not fit and not hasattr(scaler, 'scale_'):
            logger.error(f"Scaler for {scaler_type} has not been fit. Run with fit=True first.")
            # Try to load it as a fallback
            try:
                if scaler_type == 'static':
                    scaler = utils.load_model(config.MODEL_PATHS['static_scaler'])
                else:
                    scaler = utils.load_model(config.MODEL_PATHS['daily_scaler'])
            except FileNotFoundError:
                logger.error(f"Could not load scaler {scaler_type} from disk. Aborting scaling.")
                return df

        if fit:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = scaler.transform(df[numeric_cols])
        
        logger.info(f"Scaled {len(numeric_cols)} {scaler_type} features")
        
        if scaler_type == 'static':
            self.static_scaler = scaler
        else:
            self.daily_scaler = scaler
            
        return df
    
    def create_sequence_features(self, df_daily_scaled: pd.DataFrame, 
                                sequence_length: int = config.SEQUENCE_LENGTH,
                                stride: int = config.SEQUENCE_STRIDE) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Create sequences for LSTM models from *scaled daily aggregated features*"""
        
        logger.info(f"Creating sequences from scaled daily data (length={sequence_length}, stride={stride})")
        
        user_col = 'user_pseudo' if 'user_pseudo' in df_daily_scaled.columns else 'user'
        
        # Sort by user and date
        df_daily_scaled = df_daily_scaled.sort_values([user_col, 'date']).reset_index(drop=True)
        
        # Get feature columns (all numeric, already scaled)
        feature_cols = [col for col in df_daily_scaled.columns if col not in 
                       [user_col, 'date', 'is_insider', 'user', 'pc', 'user_pseudo']]
        self.daily_feature_names = feature_cols
        
        sequences = []
        labels = []
        user_ids = []
        
        # Create sequences per user
        for user in df_daily_scaled[user_col].unique():
            user_data = df_daily_scaled[df_daily_scaled[user_col] == user]
            user_features = user_data[feature_cols].values
            user_labels = user_data['is_insider'].values
            
            # Create sliding window sequences
            for i in range(0, len(user_features) - sequence_length + 1, stride):
                seq = user_features[i:i+sequence_length]
                # Label is 1 if *any* day in the sequence is an insider day
                label = 1 if np.any(user_labels[i:i+sequence_length] == 1) else 0
                
                sequences.append(seq)
                labels.append(label)
                user_ids.append(user)
        
        if not sequences:
            logger.warning("No sequences were created. Check sequence_length and data.")
            return np.array([]), np.array([]), []

        sequences = np.array(sequences)
        labels = np.array(labels)
        
        logger.info(f"Created {len(sequences)} sequences from {len(df_daily_scaled[user_col].unique())} users")
        
        return sequences, labels, user_ids
    
    def run_full_pipeline(self, df: pd.DataFrame, time_window: str = '1D') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run complete feature engineering pipeline"""
        logger.info(utils.generate_report_header("FEATURE ENGINEERING PIPELINE"))
        
        # --- Daily (Time-Series) Pipeline ---
        logger.info("--- Starting Daily Features Pipeline (for LSTM) ---")
        # 1. Create temporal features (hour, day_of_week) on raw data
        df_temporal = self.create_temporal_features(df)
        
        # 2. Create aggregated daily features (unscaled)
        daily_agg_df = self.create_user_aggregated_features(df_temporal, time_window)
        
        if len(daily_agg_df) == 0:
            logger.error("Failed to create aggregated features"); return pd.DataFrame(), pd.DataFrame()
        
        # 3. Add daily labels
        user_col = 'user_pseudo' if 'user_pseudo' in daily_agg_df.columns else 'user'
        if 'is_insider' in df.columns:
            daily_labels = df_temporal.set_index('date').groupby(
                [pd.Grouper(freq=time_window), user_col]
            )['is_insider'].max().reset_index()
            daily_agg_df = daily_agg_df.merge(daily_labels, on=[user_col, 'date'], how='left')
            daily_agg_df['is_insider'] = daily_agg_df['is_insider'].fillna(0)
        else:
            daily_agg_df['is_insider'] = 0
            
        # 4. Scale daily features
        daily_scaled_df = self.scale_features(daily_agg_df, scaler_type='daily', fit=True)
        utils.save_model(self.daily_scaler, config.MODEL_PATHS['daily_scaler'])
        
        
        # --- Static (Per-User) Pipeline ---
        logger.info("\n--- Starting Static Features Pipeline (for IF/Clustering) ---")
        # 1. Create statistical features from *unscaled* daily data
        static_df = self.create_statistical_features(daily_agg_df, user_col)
        
        # 2. Add user-level label
        if 'is_insider' in daily_agg_df.columns:
            user_labels = daily_agg_df.groupby(user_col)['is_insider'].max().reset_index()
            static_df = static_df.merge(user_labels, on=user_col, how='left')
            static_df['is_insider'] = static_df['is_insider'].fillna(0)
        else:
            static_df['is_insider'] = 0
            
        # 3. Scale static features
        static_scaled_df = self.scale_features(static_df, scaler_type='static', fit=True)
        self.static_feature_names = [col for col in static_scaled_df.columns if col not in [user_col, 'is_insider']]
        utils.save_model(self.static_scaler, config.MODEL_PATHS['static_scaler'])
        
        # Save encoders
        utils.save_model(self.label_encoders, config.MODEL_PATHS['label_encoders'])
        
        logger.info(f"Static features (scaled) completed: {len(static_scaled_df)} samples, {len(self.static_feature_names)} features")
        logger.info(f"Daily features (scaled) completed: {len(daily_scaled_df)} samples")
        
        return daily_scaled_df, static_scaled_df


def main():
    """Main execution function"""
    try:
        df = utils.load_dataframe('processed_unified_logs.csv')
    except Exception as e:
        logger.error(f"Preprocessed data not found. Run data_preprocessing.py first. Error: {e}")
        return
    
    engineer = FeatureEngineer()
    
    # Run pipeline to get both feature sets
    daily_scaled_df, static_scaled_df = engineer.run_full_pipeline(df, time_window='1D')
    
    if len(daily_scaled_df) == 0 or len(static_scaled_df) == 0:
        logger.error("Feature engineering failed.")
        return
    
    # Save static (user-level) features for IF and Clustering
    utils.save_dataframe(static_scaled_df, 'engineered_static_features.csv')
    
    # Save daily (time-series) features for sequence creation
    utils.save_dataframe(daily_scaled_df, 'engineered_daily_features.csv')
    
    # Create sequences from the *scaled* daily data
    logger.info("Creating sequences from *scaled daily aggregated data*...")
    sequences, labels, user_ids = engineer.create_sequence_features(daily_scaled_df)
    
    # Save sequences
    np.save(config.PROCESSED_DATA_DIR / 'sequences.npy', sequences)
    np.save(config.PROCESSED_DATA_DIR / 'sequence_labels.npy', labels)
    np.save(config.PROCESSED_DATA_DIR / 'sequence_users.npy', user_ids)
    
    logger.info("Sequences saved for LSTM training")
    
    # Print summary
    print("\n" + "="*80)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*80)
    print(f"\nStatic Features Dataset (for Isolation Forest / Clustering):")
    print(f"  Samples: {len(static_scaled_df)}")
    print(f"  Features: {len(engineer.static_feature_names)}")
    
    print(f"\nDaily Features Dataset (intermediate, for sequences):")
    print(f"  Samples: {len(daily_scaled_df)}")
    
    if len(sequences) > 0:
        print(f"\nSequence Dataset (for LSTM):")
        print(f"  Sequences: {len(sequences)}")
        print(f"  Sequence Length: {sequences.shape[1]}")
        print(f"  Features per Timestep: {sequences.shape[2]}")
    else:
        print(f"\nSequence Dataset (for LSTM):")
        print("  ⚠️ No sequences created. Check data or config.SEQUENCE_LENGTH.")
        
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()