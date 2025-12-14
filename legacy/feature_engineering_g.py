"""
Feature Engineering for Insider Threat Detection
Creates time-series and behavioral features from preprocessed logs

[--- CORRECTED SCALING VERSION ---]
This version fixes a logic bug.
1. Creates unscaled daily_agg_df
2. Creates unscaled stat_df from daily_agg_df
3. Scales daily_agg_df with 'daily_scaler'
4. Scales stat_df with 'static_scaler'
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
        self.static_scaler = StandardScaler()
        self.daily_scaler = StandardScaler()
        self.label_encoders = {}
        self.static_feature_names = []
        self.daily_feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df['is_night_shift'] = df['hour'].between(22, 6).astype(int)
        df['is_early_morning'] = df['hour'].between(0, 6).astype(int)
        df['is_late_night'] = df['hour'].between(20, 23).astype(int)
        
        logger.info("Temporal features created")
        return df
    
    def create_user_aggregated_features(self, df: pd.DataFrame, time_window: str = '1D') -> pd.DataFrame:
        logger.info(f"Creating DAILY aggregated features with window: {time_window}")
        user_col = 'user_pseudo' if 'user_pseudo' in df.columns else 'user'
        if user_col not in df.columns or 'date' not in df.columns:
            logger.warning("Required columns not found for aggregation")
            return pd.DataFrame()
        
        df_temp = df.copy().set_index('date')
        grouped = df_temp.groupby([pd.Grouper(freq=time_window), user_col])
        
        agg_features = grouped.agg(activity_count=('activity_type', 'count'))
        
        if 'activity_type' in df_temp.columns:
            logger.info("Aggregating activity type counts...")
            activity_counts = df_temp.groupby([pd.Grouper(freq=time_window), user_col, 'activity_type']).size().unstack(fill_value=0)
            activity_counts.columns = [f'count_{col}' for col in activity_counts.columns]
            agg_features = agg_features.join(activity_counts)
        
        # Add other aggregations
        if 'is_weekend' in df_temp.columns:
            agg_features['weekend_activity_count'] = grouped['is_weekend'].sum()
        if 'is_business_hours' in df_temp.columns:
            agg_features['after_hours_count'] = grouped['is_business_hours'].apply(lambda x: (x == 0).sum())
        if 'is_connect' in df_temp.columns:
            agg_features['device_connects'] = grouped['is_connect'].sum()
        # Add other features from your preprocessing as needed...
        
        agg_features = agg_features.reset_index().fillna(0)
        logger.info(f"Created {len(agg_features.columns)-2} daily aggregated features")
        return agg_features
    
    def create_statistical_features(self, daily_agg_df: pd.DataFrame, user_col: str) -> pd.DataFrame:
        logger.info("Creating STATIC statistical features per user...")
        numeric_cols = daily_agg_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['is_insider']]
        
        if not numeric_cols:
            logger.warning("No numeric columns found for statistical features.")
            return pd.DataFrame(daily_agg_df[user_col].unique(), columns=[user_col])

        user_stats = daily_agg_df.groupby(user_col)[numeric_cols].agg(['mean', 'std', 'min', 'max', 'median', 'sum'])
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
        user_stats = user_stats.fillna(0).reset_index()
        
        logger.info(f"Created {len(user_stats.columns)-1} static statistical features")
        return user_stats

    def create_sequence_features(self, daily_scaled_df: pd.DataFrame, 
                                sequence_length: int = config.SEQUENCE_LENGTH,
                                stride: int = config.SEQUENCE_STRIDE) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        logger.info(f"Creating sequences from aggregated data (length={sequence_length}, stride={stride})")
        user_col = 'user_pseudo' if 'user_pseudo' in daily_scaled_df.columns else 'user'
        df = daily_scaled_df.sort_values([user_col, 'date']).reset_index(drop=True)
        
        feature_cols = self.daily_feature_names
        if not feature_cols:
             logger.error("Daily feature names not set. Scaling might have failed.")
             return np.array([]), np.array([]), []

        sequences, labels, user_ids = [], [], []
        
        for user in df[user_col].unique():
            user_data = df[df[user_col] == user][feature_cols].values
            user_labels = df[df[user_col] == user]['is_insider'].values if 'is_insider' in df.columns else np.zeros(len(user_data))
            
            for i in range(0, len(user_data) - sequence_length + 1, stride):
                seq = user_data[i:i+sequence_length]
                label = 1 if np.any(user_labels[i:i+sequence_length] == 1) else 0
                sequences.append(seq)
                labels.append(label)
                user_ids.append(user)
        
        if not sequences:
            logger.warning("No sequences were created.")
            return np.array([]), np.array([]), []

        sequences, labels = np.array(sequences), np.array(labels)
        logger.info(f"Created {len(sequences)} sequences from {len(df[user_col].unique())} users")
        return sequences, labels, user_ids
    
    def create_behavioral_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating behavioral ratio features...")
        df = df.copy()
        if 'after_hours_count' in df.columns and 'activity_count' in df.columns:
            df['after_hours_ratio'] = df['after_hours_count'] / (df['activity_count'] + 1e-10)
        if 'weekend_activity_count' in df.columns and 'activity_count' in df.columns:
            df['weekend_activity_ratio'] = df['weekend_activity_count'] / (df['activity_count'] + 1e-10)
        # Add other ratios...
        logger.info("Behavioral ratio features created")
        return df
    
    def scale_features(self, df: pd.DataFrame, scaler: StandardScaler, fit: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        logger.info(f"Scaling features (fit={fit})...")
        df = df.copy()
        user_col = 'user_pseudo' if 'user_pseudo' in df.columns else 'user'
        exclude_cols = ['is_insider', 'date', user_col, 'user_pseudo', 'user']
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
        
        if not numeric_cols:
            logger.warning("No numeric columns found for scaling")
            return df, []
        
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols]) if fit else scaler.transform(df[numeric_cols])
        logger.info(f"Scaled {len(numeric_cols)} features")
        return df, numeric_cols
    
    def save_artifacts(self):
        utils.save_model(self.static_scaler, config.MODELS_DIR / 'static_scaler.pkl')
        utils.save_model(self.daily_scaler, config.MODELS_DIR / 'daily_scaler.pkl')
        utils.save_model(self.label_encoders, config.MODEL_PATHS['label_encoders'])
        logger.info("Feature engineering artifacts saved")
    
    def run_full_pipeline(self, df: pd.DataFrame, time_window: str = '1D') -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(utils.generate_report_header("FEATURE ENGINEERING PIPELINE"))
        
        # === NEW LOGIC ORDER ===
        
        # 1. Create temporal features on raw logs
        df = self.create_temporal_features(df)
        
        # 2. Create UNSCALED daily aggregated features
        daily_agg_df = self.create_user_aggregated_features(df, time_window)
        if daily_agg_df.empty:
            logger.error("Failed to create aggregated features"); return pd.DataFrame(), pd.DataFrame()
        daily_agg_df = self.create_behavioral_ratios(daily_agg_df)
        
        # 3. Handle daily labels
        user_col = 'user_pseudo' if 'user_pseudo' in daily_agg_df.columns else 'user'
        if 'is_insider' in df.columns:
            daily_labels = df.set_index('date').groupby([pd.Grouper(freq=time_window), user_col])['is_insider'].max().reset_index()
            daily_agg_df = daily_agg_df.merge(daily_labels, on=[user_col, 'date'], how='left')
            daily_agg_df['is_insider'] = daily_agg_df['is_insider'].fillna(0)
        else:
            daily_agg_df['is_insider'] = 0
            
        # 4. Create UNSCALED static features from UNSCALED daily features
        stat_df = self.create_statistical_features(daily_agg_df, user_col)
        
        # 5. Add static LDAP/Psychometric features
        static_cols = ['user_pseudo', 'role', 'domain', 'O', 'C', 'E', 'A', 'N']
        # Filter columns that actually exist in the dataframe
        static_cols_exist = [col for col in static_cols if col in df.columns]
        if 'user_pseudo' not in static_cols_exist:
            static_cols_exist = [user_col] + static_cols_exist
            
        static_user_data = df[static_cols_exist].drop_duplicates(subset=[user_col]).reset_index(drop=True)
        
        # One-hot encode categorical static features
        cat_cols_to_encode = [col for col in ['role', 'domain'] if col in static_user_data.columns]
        if cat_cols_to_encode:
            static_user_data = pd.get_dummies(static_user_data, columns=cat_cols_to_encode, dummy_na=True)
        
        stat_df = stat_df.merge(static_user_data, on=user_col, how='left')
        stat_df = stat_df.fillna(0) # Fill NaNs from merge and OHE
        
        # 6. Add final static label
        if 'is_insider' in daily_agg_df.columns:
            labels = daily_agg_df.groupby(user_col)['is_insider'].max().reset_index()
            stat_df = stat_df.merge(labels, on=user_col, how='left')
            stat_df['is_insider'] = stat_df['is_insider'].fillna(0)
        
        # 7. Scale daily features AND save feature names
        daily_scaled_df, self.daily_feature_names = self.scale_features(daily_agg_df, self.daily_scaler, fit=True)
        
        # 8. Scale static features AND save feature names
        static_scaled_df, self.static_feature_names = self.scale_features(stat_df, self.static_scaler, fit=True)
        
        # 9. Save artifacts
        self.save_artifacts()
        
        logger.info(f"Daily aggregation completed: {len(daily_scaled_df)} samples, {len(self.daily_feature_names)} features")
        logger.info(f"Static features completed: {len(static_scaled_df)} samples, {len(self.static_feature_names)} features")
        
        return daily_scaled_df, static_scaled_df

def main():
    try:
        df = utils.load_dataframe('processed_unified_logs.csv')
    except:
        logger.error("Preprocessed data not found. Run data_preprocessing.py first."); return
    
    engineer = FeatureEngineer()
    daily_scaled_df, static_scaled_df = engineer.run_full_pipeline(df, time_window='1D')
    
    if daily_scaled_df.empty or static_scaled_df.empty:
        logger.error("Feature engineering failed."); return
    
    # Save static features
    utils.save_dataframe(static_scaled_df, 'engineered_static_features.csv')
    
    # Save daily features (for sequence creation)
    utils.save_dataframe(daily_scaled_df, 'engineered_daily_features.csv')
    
    # Create sequences from SCALED daily features
    logger.info("Creating sequences from *scaled* daily aggregated data...")
    sequences, labels, user_ids = engineer.create_sequence_features(daily_scaled_df)
    
    if len(sequences) > 0:
        np.save(config.PROCESSED_DATA_DIR / 'sequences.npy', sequences)
        np.save(config.PROCESSED_DATA_DIR / 'sequence_labels.npy', labels)
        np.save(config.PROCESSED_DATA_DIR / 'sequence_users.npy', user_ids)
        logger.info("Sequences saved for LSTM training")
    
    # Print summary
    print("\n" + "="*80)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*80)
    print(f"\nStatic Features Dataset (for Isolation Forest, Deep Clustering):")
    print(f"  Samples: {len(static_scaled_df)}")
    print(f"  Features: {len(engineer.static_feature_names)}")
    
    print(f"\nDaily Features Dataset (intermediate):")
    print(f"  Samples: {len(daily_scaled_df)}")
    print(f"  Features: {len(engineer.daily_feature_names)}")
    
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