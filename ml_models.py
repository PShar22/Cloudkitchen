"""
Machine Learning Forecasting Models
Random Forest, XGBoost, and LightGBM models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
from datetime import datetime

class MLForecaster:
    """Base class for ML forecasting models"""
    
    def __init__(self, name="ML Model"):
        self.name = name
        self.model = None
        self.feature_cols = None
        self.target_col = 'order_count'
        
    def prepare_features(self, df, feature_type='time_only'):
        """Select features based on type"""
        
        # Base temporal features
        time_features = [
            'hour', 'day_of_week', 'day_of_month', 'week_of_year', 'month',
            'is_weekend', 'is_lunch_peak', 'is_dinner_peak', 'is_peak_hour',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_holiday'
        ]
        
        # Lag and rolling features
        lag_features = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
        
        # Weather features (only numeric dummy variables)
        weather_features = [col for col in df.columns if ('weather_' in col or 'temp_' in col) and col not in ['weather_condition', 'temp_category']]
        
        # Demographic features
        demographic_features = [
            'population', 'median_income', 'median_age', 'poverty_rate',
            'income_normalized', 'age_normalized', 'population_normalized'
        ]
        
        if feature_type == 'time_only':
            self.feature_cols = time_features + lag_features
        elif feature_type == 'time_weather':
            self.feature_cols = time_features + lag_features + weather_features
        elif feature_type == 'demographic_aware':
            self.feature_cols = time_features + lag_features + weather_features + demographic_features
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
        
        # Filter to only existing columns
        self.feature_cols = [col for col in self.feature_cols if col in df.columns]
        
        return df[self.feature_cols]
    
    def fit(self, train_df, feature_type='time_only'):
        """Train the model"""
        X_train = self.prepare_features(train_df, feature_type)
        y_train = train_df[self.target_col]
        
        print(f"Training {self.name} with {len(self.feature_cols)} features ({feature_type})...")
        self.model.fit(X_train, y_train)
        
        return self
    
    def predict(self, test_df):
        """Make predictions"""
        X_test = test_df[self.feature_cols]
        predictions = self.model.predict(X_test)
        return np.maximum(predictions, 0)  # Ensure non-negative
    
    def evaluate(self, y_true, y_pred):
        """Calculate metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance.head(top_n)
        return None
    
    def save(self, filepath):
        """Save model"""
        joblib.dump(self.model, filepath)
        print(f"Model saved: {filepath}")
    
    def load(self, filepath):
        """Load model"""
        self.model = joblib.load(filepath)
        print(f"Model loaded: {filepath}")


class RandomForestForecaster(MLForecaster):
    """Random Forest model"""
    
    def __init__(self, n_estimators=100, max_depth=15, random_state=42):
        super().__init__(name="Random Forest")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )


class XGBoostForecaster(MLForecaster):
    """XGBoost model"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        super().__init__(name="XGBoost")
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )


class LightGBMForecaster(MLForecaster):
    """LightGBM model"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        super().__init__(name="LightGBM")
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )


def run_ml_models(train_path='data/processed/features_train.csv',
                 test_path='data/processed/features_test.csv',
                 target_col='order_count'):
    """Train and evaluate all ML models"""
    
    print("="*70)
    print("Running Machine Learning Forecasting Models")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train = pd.read_csv(train_path, parse_dates=['interval_start'])
    test = pd.read_csv(test_path, parse_dates=['interval_start'])
    
    print(f"Train set: {len(train)} intervals")
    print(f"Test set: {len(test)} intervals")
    
    y_test = test[target_col].values
    
    # Initialize models
    models = [
        RandomForestForecaster(n_estimators=100, max_depth=15),
        XGBoostForecaster(n_estimators=100, max_depth=6, learning_rate=0.1),
        LightGBMForecaster(n_estimators=100, max_depth=6, learning_rate=0.1)
    ]
    
    # Feature types to test
    feature_types = ['time_only', 'demographic_aware']
    
    results = []
    
    for model in models:
        for feature_type in feature_types:
            print(f"\n{'='*70}")
            print(f"{model.name} - {feature_type.replace('_', ' ').title()}")
            print('='*70)
            
            # Train
            model.fit(train, feature_type=feature_type)
            print(f"Features used: {len(model.feature_cols)}")
            
            # Predict
            y_pred = model.predict(test)
            
            # Evaluate
            metrics = model.evaluate(y_test, y_pred)
            
            print(f"\nPerformance Metrics:")
            print(f"  MAE:  {metrics['MAE']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            
            # Feature importance
            importance = model.get_feature_importance(top_n=10)
            if importance is not None:
                print(f"\nTop 10 Important Features:")
                for idx, row in importance.iterrows():
                    print(f"  {row['feature']:30s}: {row['importance']:.4f}")
            
            # Save model
            model_filename = f"models/{model.name.lower().replace(' ', '_')}_{feature_type}.pkl"
            model.save(model_filename)
            
            # Store results
            results.append({
                'Model': model.name,
                'Feature_Type': feature_type,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE'],
                'Num_Features': len(model.feature_cols)
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('MAE')
    
    print("\n" + "="*70)
    print("ML Models Comparison (sorted by MAE)")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('results/tables/ml_results.csv', index=False)
    print("\nSaved: results/tables/ml_results.csv")
    
    # Compare time_only vs demographic_aware
    print("\n" + "="*70)
    print("Impact of Demographic Features")
    print("="*70)
    
    for model_name in results_df['Model'].unique():
        model_results = results_df[results_df['Model'] == model_name]
        time_only = model_results[model_results['Feature_Type'] == 'time_only'].iloc[0]
        demo_aware = model_results[model_results['Feature_Type'] == 'demographic_aware'].iloc[0]
        
        mae_improvement = ((time_only['MAE'] - demo_aware['MAE']) / time_only['MAE']) * 100
        rmse_improvement = ((time_only['RMSE'] - demo_aware['RMSE']) / time_only['RMSE']) * 100
        mape_improvement = ((time_only['MAPE'] - demo_aware['MAPE']) / time_only['MAPE']) * 100
        
        print(f"\n{model_name}:")
        print(f"  MAE improvement:  {mae_improvement:+.2f}%")
        print(f"  RMSE improvement: {rmse_improvement:+.2f}%")
        print(f"  MAPE improvement: {mape_improvement:+.2f}%")
    
    # Best model
    best = results_df.iloc[0]
    print(f"\n{'='*70}")
    print(f"Best Model: {best['Model']} ({best['Feature_Type']})")
    print(f"  MAE:  {best['MAE']:.4f}")
    print(f"  RMSE: {best['RMSE']:.4f}")
    print(f"  MAPE: {best['MAPE']:.2f}%")
    print("="*70)
    
    return results_df


if __name__ == "__main__":
    results = run_ml_models()
