"""
Data Preprocessing and Feature Engineering
Prepares data for machine learning models
"""

import pandas as pd
import numpy as np
from datetime import timedelta

class DataPreprocessor:
    """Preprocess and engineer features for demand forecasting"""
    
    def __init__(self, orders_path='data/raw/orders_synthetic.csv',
                 demographics_path='data/external/demographics_60642.csv'):
        self.orders = pd.read_csv(orders_path, parse_dates=['order_datetime', 'order_date'])
        self.demographics = pd.read_csv(demographics_path)
        
    def aggregate_to_intervals(self, interval_minutes=30):
        """Aggregate orders into time intervals"""
        print(f"Aggregating orders into {interval_minutes}-minute intervals...")
        
        # Filter completed orders only
        completed = self.orders[self.orders['order_status'] == 'Completed'].copy()
        
        # Create interval timestamp
        completed['interval_start'] = completed['order_datetime'].dt.floor(f'{interval_minutes}min')
        
        # Aggregate by interval
        agg_dict = {
            'order_id': 'count',  # Number of orders
            'order_subtotal': 'sum',  # Total revenue
            'prep_time_min': 'mean',  # Average prep time
            'delivery_time_min': 'mean',  # Average delivery time
            'total_delivery_time_min': 'mean',  # Average total time
            'avoidable_wait_min': 'mean',  # Average wait
            'rating': 'mean',  # Average rating
            'num_items': 'sum'  # Total items
        }
        
        intervals = completed.groupby('interval_start').agg(agg_dict).reset_index()
        intervals.columns = ['interval_start', 'order_count', 'total_revenue', 
                            'avg_prep_time', 'avg_delivery_time', 'avg_total_time',
                            'avg_wait_time', 'avg_rating', 'total_items']
        
        # Create complete time range (fill missing intervals with 0)
        date_range = pd.date_range(
            start=completed['order_datetime'].min().floor('D'),
            end=completed['order_datetime'].max().ceil('D'),
            freq=f'{interval_minutes}min'
        )
        
        full_intervals = pd.DataFrame({'interval_start': date_range})
        intervals = full_intervals.merge(intervals, on='interval_start', how='left')
        intervals['order_count'] = intervals['order_count'].fillna(0)
        intervals['total_revenue'] = intervals['total_revenue'].fillna(0)
        
        print(f"Created {len(intervals)} intervals")
        print(f"Non-zero intervals: {(intervals['order_count'] > 0).sum()}")
        
        return intervals
    
    def add_temporal_features(self, df):
        """Add time-based features"""
        print("Adding temporal features...")
        
        df['hour'] = df['interval_start'].dt.hour
        df['day_of_week'] = df['interval_start'].dt.dayofweek
        df['day_of_month'] = df['interval_start'].dt.day
        df['week_of_year'] = df['interval_start'].dt.isocalendar().week
        df['month'] = df['interval_start'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Time of day categories
        df['time_of_day'] = pd.cut(df['hour'], 
                                    bins=[0, 6, 11, 14, 18, 22, 24],
                                    labels=['night', 'morning', 'lunch', 'afternoon', 'dinner', 'late'],
                                    include_lowest=True)
        
        # Peak hours
        df['is_lunch_peak'] = ((df['hour'] >= 11) & (df['hour'] <= 13)).astype(int)
        df['is_dinner_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 20)).astype(int)
        df['is_peak_hour'] = (df['is_lunch_peak'] | df['is_dinner_peak']).astype(int)
        
        # Cyclical encoding for hour and day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def add_lag_features(self, df, target_col='order_count', lags=[1, 2, 3, 6, 12, 24, 48]):
        """Add lagged demand features"""
        print(f"Adding lag features for {target_col}...")
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def add_rolling_features(self, df, target_col='order_count', windows=[3, 6, 12, 24]):
        """Add rolling statistics"""
        print(f"Adding rolling features for {target_col}...")
        
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].shift(1).rolling(window=window).max()
        
        return df
    
    def add_demographic_features(self, df):
        """Add demographic features from zip code"""
        print("Adding demographic features...")
        
        # Add all demographic features
        for col in self.demographics.columns:
            if col != 'zip_code':
                df[col] = self.demographics[col].values[0]
        
        # Normalize numeric demographics
        df['income_normalized'] = df['median_income'] / 100000  # Scale to 0-2 range
        df['age_normalized'] = df['median_age'] / 100  # Scale to 0-1 range
        df['population_normalized'] = df['population'] / 50000  # Scale
        
        return df
    
    def add_weather_features(self, df):
        """Add weather features from original orders"""
        print("Adding weather features...")
        
        # Merge weather data from original orders
        weather_data = self.orders[['order_datetime', 'weather_condition', 'temperature_c']].copy()
        weather_data['interval_start'] = weather_data['order_datetime'].dt.floor('30min')
        weather_data = weather_data.groupby('interval_start').first().reset_index()
        
        df = df.merge(weather_data[['interval_start', 'weather_condition', 'temperature_c']], 
                     on='interval_start', how='left')
        
        # Forward fill missing weather
        df['weather_condition'] = df['weather_condition'].fillna(method='ffill')
        df['temperature_c'] = df['temperature_c'].fillna(method='ffill')
        
        # One-hot encode weather
        weather_dummies = pd.get_dummies(df['weather_condition'], prefix='weather')
        df = pd.concat([df, weather_dummies], axis=1)
        
        # Temperature categories
        df['temp_category'] = pd.cut(df['temperature_c'], 
                                     bins=[-10, 0, 10, 20, 30],
                                     labels=['very_cold', 'cold', 'mild', 'warm'])
        temp_dummies = pd.get_dummies(df['temp_category'], prefix='temp')
        df = pd.concat([df, temp_dummies], axis=1)
        
        return df
    
    def add_holiday_features(self, df):
        """Add holiday indicators"""
        print("Adding holiday features...")
        
        # Major US holidays
        holidays = [
            ('2025-09-07', 'Labor Day'),
            ('2025-10-31', 'Halloween'),
            ('2025-11-28', 'Thanksgiving'),
            ('2025-12-25', 'Christmas'),
            ('2026-01-01', 'New Year'),
        ]
        
        df['is_holiday'] = 0
        df['holiday_name'] = ''
        
        for date_str, name in holidays:
            date = pd.to_datetime(date_str)
            mask = df['interval_start'].dt.date == date.date()
            df.loc[mask, 'is_holiday'] = 1
            df.loc[mask, 'holiday_name'] = name
        
        return df
    
    def create_train_test_split(self, df, test_size_days=30):
        """Split data into train and test sets (time-based)"""
        print(f"Splitting data: last {test_size_days} days for testing...")
        
        split_date = df['interval_start'].max() - timedelta(days=test_size_days)
        
        train = df[df['interval_start'] < split_date].copy()
        test = df[df['interval_start'] >= split_date].copy()
        
        print(f"Train set: {len(train)} intervals ({train['interval_start'].min()} to {train['interval_start'].max()})")
        print(f"Test set: {len(test)} intervals ({test['interval_start'].min()} to {test['interval_start'].max()})")
        
        return train, test
    
    def process_all(self, interval_minutes=30, save=True):
        """Run complete preprocessing pipeline"""
        print("="*60)
        print("Starting Data Preprocessing Pipeline")
        print("="*60)
        
        # Aggregate to intervals
        df = self.aggregate_to_intervals(interval_minutes)
        
        # Add all features
        df = self.add_temporal_features(df)
        df = self.add_demographic_features(df)
        df = self.add_weather_features(df)
        df = self.add_holiday_features(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        
        # Drop rows with NaN in lag/rolling features (first few rows)
        initial_rows = len(df)
        df = df.dropna()
        print(f"Dropped {initial_rows - len(df)} rows with missing lag/rolling features")
        
        # Split train/test
        train, test = self.create_train_test_split(df)
        
        if save:
            df.to_csv('data/processed/features_full.csv', index=False)
            train.to_csv('data/processed/features_train.csv', index=False)
            test.to_csv('data/processed/features_test.csv', index=False)
            print("\nSaved processed data:")
            print("  - data/processed/features_full.csv")
            print("  - data/processed/features_train.csv")
            print("  - data/processed/features_test.csv")
        
        print("\n" + "="*60)
        print("Preprocessing Complete!")
        print("="*60)
        
        return df, train, test

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df, train, test = preprocessor.process_all()
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Features: {df.shape[1]}")
    print(f"\nTarget variable (order_count) statistics:")
    print(df['order_count'].describe())
