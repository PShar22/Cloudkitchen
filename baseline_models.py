"""
Baseline Forecasting Models
Simple models to establish performance benchmarks
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class BaselineForecaster:
    """Collection of baseline forecasting methods"""
    
    def __init__(self):
        self.name = "Baseline"
        self.predictions = None
        
    def fit(self, train_data, target_col='order_count'):
        """Fit baseline model"""
        self.train_data = train_data
        self.target_col = target_col
        return self
    
    def predict(self, test_data):
        """Make predictions - to be implemented by subclasses"""
        raise NotImplementedError
    
    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }


class NaiveForecaster(BaselineForecaster):
    """Naive forecast: use last observed value"""
    
    def __init__(self):
        super().__init__()
        self.name = "Naive (Last Value)"
    
    def predict(self, test_data):
        """Predict using last observed value"""
        predictions = []
        last_value = self.train_data[self.target_col].iloc[-1]
        
        for _ in range(len(test_data)):
            predictions.append(last_value)
        
        return np.array(predictions)


class MovingAverageForecaster(BaselineForecaster):
    """Moving average forecast"""
    
    def __init__(self, window=24):
        super().__init__()
        self.window = window
        self.name = f"Moving Average (window={window})"
    
    def predict(self, test_data):
        """Predict using moving average"""
        # Use last 'window' values from training data
        recent_values = self.train_data[self.target_col].tail(self.window).values
        ma_value = recent_values.mean()
        
        predictions = np.full(len(test_data), ma_value)
        return predictions


class SeasonalNaiveForecaster(BaselineForecaster):
    """Seasonal naive: use value from same time last week"""
    
    def __init__(self, seasonal_period=48):  # 48 intervals = 1 day for 30-min intervals
        super().__init__()
        self.seasonal_period = seasonal_period
        self.name = f"Seasonal Naive (period={seasonal_period})"
    
    def predict(self, test_data):
        """Predict using seasonal pattern"""
        predictions = []
        train_values = self.train_data[self.target_col].values
        
        for i in range(len(test_data)):
            # Look back by seasonal_period
            if len(train_values) >= self.seasonal_period:
                seasonal_value = train_values[-self.seasonal_period]
            else:
                seasonal_value = train_values[-1]
            
            predictions.append(seasonal_value)
        
        return np.array(predictions)


class HourlyAverageForecaster(BaselineForecaster):
    """Average by hour of day"""
    
    def __init__(self):
        super().__init__()
        self.name = "Hourly Average"
        self.hourly_averages = None
    
    def fit(self, train_data, target_col='order_count'):
        """Calculate average for each hour"""
        super().fit(train_data, target_col)
        self.hourly_averages = train_data.groupby('hour')[target_col].mean().to_dict()
        return self
    
    def predict(self, test_data):
        """Predict using hourly averages"""
        predictions = test_data['hour'].map(self.hourly_averages).values
        # Fill any missing hours with overall mean
        overall_mean = self.train_data[self.target_col].mean()
        predictions = np.where(np.isnan(predictions), overall_mean, predictions)
        return predictions


class HourDayAverageForecaster(BaselineForecaster):
    """Average by hour and day of week"""
    
    def __init__(self):
        super().__init__()
        self.name = "Hour-Day Average"
        self.hour_day_averages = None
    
    def fit(self, train_data, target_col='order_count'):
        """Calculate average for each hour-day combination"""
        super().fit(train_data, target_col)
        self.hour_day_averages = train_data.groupby(['hour', 'day_of_week'])[target_col].mean().to_dict()
        self.overall_mean = train_data[target_col].mean()
        return self
    
    def predict(self, test_data):
        """Predict using hour-day averages"""
        predictions = []
        
        for _, row in test_data.iterrows():
            key = (row['hour'], row['day_of_week'])
            pred = self.hour_day_averages.get(key, self.overall_mean)
            predictions.append(pred)
        
        return np.array(predictions)


def run_baseline_models(train_path='data/processed/features_train.csv',
                       test_path='data/processed/features_test.csv',
                       target_col='order_count'):
    """Run all baseline models and compare results"""
    
    print("="*60)
    print("Running Baseline Forecasting Models")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train = pd.read_csv(train_path, parse_dates=['interval_start'])
    test = pd.read_csv(test_path, parse_dates=['interval_start'])
    
    print(f"Train set: {len(train)} intervals")
    print(f"Test set: {len(test)} intervals")
    
    y_test = test[target_col].values
    
    # Initialize models
    models = [
        NaiveForecaster(),
        MovingAverageForecaster(window=24),
        MovingAverageForecaster(window=48),
        SeasonalNaiveForecaster(seasonal_period=48),
        HourlyAverageForecaster(),
        HourDayAverageForecaster()
    ]
    
    # Train and evaluate each model
    results = []
    
    for model in models:
        print(f"\n{model.name}:")
        print("-" * 40)
        
        # Fit and predict
        model.fit(train, target_col)
        y_pred = model.predict(test)
        
        # Evaluate
        metrics = model.evaluate(y_test, y_pred)
        
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        
        results.append({
            'Model': model.name,
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE']
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('MAE')
    
    print("\n" + "="*60)
    print("Baseline Models Comparison (sorted by MAE)")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('results/tables/baseline_results.csv', index=False)
    print("\nSaved: results/tables/baseline_results.csv")
    
    # Identify best baseline
    best_model = results_df.iloc[0]
    print(f"\nBest Baseline Model: {best_model['Model']}")
    print(f"  MAE: {best_model['MAE']:.4f}")
    print(f"  RMSE: {best_model['RMSE']:.4f}")
    print(f"  MAPE: {best_model['MAPE']:.2f}%")
    
    return results_df


if __name__ == "__main__":
    results = run_baseline_models()
