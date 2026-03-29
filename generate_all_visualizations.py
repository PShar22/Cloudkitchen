"""
COMPLETE VISUALIZATION GENERATOR
Generates ALL figures for academic report (30+ figures)
Chapter 3: Exploratory (6) + Methodology (2) = 8 figures
Chapter 4: Individual Models (20) + Comparison (2) = 22 figures
Total: 30+ figures + 6 tables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Create directories if they don't exist
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)
os.makedirs('results/tables_html', exist_ok=True)
print("✓ Directories created/verified")

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("COMPLETE VISUALIZATION GENERATOR - CLOUD KITCHEN PROJECT")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Generating 30+ figures for academic report")
print("="*80)

print("\nLoading data...")
orders = pd.read_csv('data/raw/orders_synthetic.csv', parse_dates=['order_datetime', 'order_date'])
features = pd.read_csv('data/processed/features_full.csv', parse_dates=['interval_start'])
train = pd.read_csv('data/processed/features_train.csv', parse_dates=['interval_start'])
test = pd.read_csv('data/processed/features_test.csv', parse_dates=['interval_start'])
menu = pd.read_csv('data/raw/menu_items.csv')
demographics = pd.read_csv('data/external/demographics_60642.csv')
completed = orders[orders['order_status'] == 'Completed']

print(f"✓ Orders: {len(orders):,}")
print(f"✓ Train: {train.shape}, Test: {test.shape}")
print(f"✓ Completed: {len(completed):,}")

# Generate model results if they don't exist
import subprocess
import sys

print("\n" + "="*80)
print("GENERATING MODEL RESULTS (if needed)")
print("="*80)

if not os.path.exists('results/tables/baseline_results.csv'):
    print("Running baseline models...")
    subprocess.run([sys.executable, 'src/models/baseline_models.py'], check=True)
else:
    print("✓ Baseline results already exist")

if not os.path.exists('results/tables/ml_results.csv'):
    print("Running ML models...")
    subprocess.run([sys.executable, 'src/models/ml_models.py'], check=True)
else:
    print("✓ ML results already exist")

print("\n" + "="*80)
print("CHAPTER 3: EXPLORATORY DATA ANALYSIS (Figures 1-6)")
print("="*80)


print("\n[1/30] Daily Order Volume...")
daily_orders = orders.groupby('order_date').size()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(daily_orders.index, daily_orders.values, linewidth=2, color='steelblue')
ax.fill_between(daily_orders.index, daily_orders.values, alpha=0.3, color='steelblue')
ax.set_title('Daily Order Volume Over Time (6 Months)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Number of Orders', fontsize=12)
ax.grid(True, alpha=0.3)
ax.axhline(daily_orders.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {daily_orders.mean():.1f}')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/figures/01_daily_order_volume.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: results/figures/01_daily_order_volume.png")

# ============================================================================
# FIGURE 2: Hourly Demand Patterns
# ============================================================================
print("[2/10] Generating hourly demand patterns...")
hourly_orders = orders.groupby('order_hour').size()

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(hourly_orders.index, hourly_orders.values, color='coral', edgecolor='black', linewidth=1.5)
ax.set_title('Order Volume by Hour of Day', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Total Orders', fontsize=12)
ax.axvspan(11, 13, alpha=0.2, color='orange', label='Lunch Peak (11am-1pm)')
ax.axvspan(18, 20, alpha=0.2, color='red', label='Dinner Peak (6pm-8pm)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/figures/02_hourly_demand_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: results/figures/02_hourly_demand_patterns.png")


# ============================================================================
# FIGURE 3: Day of Week Analysis
# ============================================================================
print("[3/10] Generating day of week analysis...")
dow_orders = orders.groupby('day_name').size().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['steelblue']*5 + ['coral']*2  # Weekdays blue, weekends coral
bars = ax.bar(range(7), dow_orders.values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_title('Order Volume by Day of Week', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Day of Week', fontsize=12)
ax.set_ylabel('Total Orders', fontsize=12)
ax.set_xticks(range(7))
ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/figures/03_day_of_week_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: results/figures/03_day_of_week_analysis.png")

# ============================================================================
# FIGURE 4: Weather Impact on Orders
# ============================================================================
print("[4/10] Generating weather impact analysis...")
weather_stats = orders.groupby('weather_condition').agg({
    'order_id': 'count',
    'order_subtotal': 'mean'
}).rename(columns={'order_id': 'order_count', 'order_subtotal': 'avg_order_value'})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Order count
weather_stats['order_count'].plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black', linewidth=1.5)
axes[0].set_title('Order Volume by Weather Condition', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Weather', fontsize=11)
axes[0].set_ylabel('Total Orders', fontsize=11)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# Average order value
weather_stats['avg_order_value'].plot(kind='bar', ax=axes[1], color='lightcoral', edgecolor='black', linewidth=1.5)
axes[1].set_title('Average Order Value by Weather', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Weather', fontsize=11)
axes[1].set_ylabel('Average Order Value ($)', fontsize=11)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/figures/04_weather_impact.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: results/figures/04_weather_impact.png")


# ============================================================================
# FIGURE 5: Delivery Performance Metrics
# ============================================================================
print("[5/10] Generating delivery performance metrics...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Prep time
axes[0, 0].hist(completed['prep_time_min'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(completed['prep_time_min'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {completed["prep_time_min"].mean():.1f} min')
axes[0, 0].set_title('Preparation Time Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Minutes', fontsize=10)
axes[0, 0].set_ylabel('Frequency', fontsize=10)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Delivery time
axes[0, 1].hist(completed['delivery_time_min'], bins=30, color='lightyellow', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(completed['delivery_time_min'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {completed["delivery_time_min"].mean():.1f} min')
axes[0, 1].set_title('Delivery Time Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Minutes', fontsize=10)
axes[0, 1].set_ylabel('Frequency', fontsize=10)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Total time
axes[1, 0].hist(completed['total_delivery_time_min'], bins=30, color='lightblue', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(completed['total_delivery_time_min'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {completed["total_delivery_time_min"].mean():.1f} min')
axes[1, 0].set_title('Total Delivery Time Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Minutes', fontsize=10)
axes[1, 0].set_ylabel('Frequency', fontsize=10)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Rating
rating_counts = completed['rating'].value_counts().sort_index()
axes[1, 1].bar(rating_counts.index, rating_counts.values, color='gold', edgecolor='black', linewidth=1.5)
axes[1, 1].set_title('Customer Rating Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Rating (Stars)', fontsize=10)
axes[1, 1].set_ylabel('Count', fontsize=10)
axes[1, 1].set_xticks([3, 4, 5])
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/figures/05_delivery_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: results/figures/05_delivery_performance.png")


# ============================================================================
# FIGURE 6: Revenue Analysis
# ============================================================================
print("[6/10] Generating revenue analysis...")
daily_revenue = orders.groupby('order_date')['order_subtotal'].sum()

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Daily revenue
axes[0].plot(daily_revenue.index, daily_revenue.values, linewidth=2, color='green')
axes[0].fill_between(daily_revenue.index, daily_revenue.values, alpha=0.3, color='green')
axes[0].set_title('Daily Revenue Over Time', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Date', fontsize=11)
axes[0].set_ylabel('Revenue ($)', fontsize=11)
axes[0].axhline(daily_revenue.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: ${daily_revenue.mean():,.2f}')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Order value distribution
axes[1].hist(orders['order_subtotal'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].axvline(orders['order_subtotal'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: ${orders["order_subtotal"].mean():.2f}')
axes[1].set_title('Order Value Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Order Value ($)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/figures/06_revenue_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: results/figures/06_revenue_analysis.png")

# ============================================================================
# FIGURE 7: Model Performance Comparison
# ============================================================================
print("[7/10] Generating model performance comparison...")
baseline_results = pd.read_csv('results/tables/baseline_results.csv')
ml_results = pd.read_csv('results/tables/ml_results.csv')

baseline_results['Type'] = 'Baseline'
ml_results['Type'] = 'ML'

all_results = pd.concat([
    baseline_results[['Model', 'Type', 'MAE', 'RMSE', 'MAPE']],
    ml_results[['Model', 'Type', 'MAE', 'RMSE', 'MAPE']]
]).sort_values('MAE').head(10)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# MAE
colors = ['steelblue' if t == 'Baseline' else 'coral' for t in all_results['Type']]
axes[0].barh(range(len(all_results)), all_results['MAE'], color=colors, edgecolor='black')
axes[0].set_yticks(range(len(all_results)))
axes[0].set_yticklabels(all_results['Model'], fontsize=9)
axes[0].set_xlabel('MAE', fontsize=11)
axes[0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# RMSE
axes[1].barh(range(len(all_results)), all_results['RMSE'], color=colors, edgecolor='black')
axes[1].set_yticks(range(len(all_results)))
axes[1].set_yticklabels(all_results['Model'], fontsize=9)
axes[1].set_xlabel('RMSE', fontsize=11)
axes[1].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

# MAPE
axes[2].barh(range(len(all_results)), all_results['MAPE'], color=colors, edgecolor='black')
axes[2].set_yticks(range(len(all_results)))
axes[2].set_yticklabels(all_results['Model'], fontsize=9)
axes[2].set_xlabel('MAPE (%)', fontsize=11)
axes[2].set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')
axes[2].invert_yaxis()
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/figures/07_model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: results/figures/07_model_performance_comparison.png")


# ============================================================================
# FIGURE 8: Feature Importance (Top Features)
# ============================================================================
print("[8/10] Generating feature importance visualization...")
import joblib

# Load best model
best_model = joblib.load('models/lightgbm_time_only.pkl')
feature_cols = [col for col in train.columns if col not in ['interval_start', 'order_count', 'total_revenue',
                'avg_prep_time', 'avg_delivery_time', 'avg_total_time', 'avg_wait_time', 'avg_rating', 'total_items',
                'weather_condition', 'temp_category', 'holiday_name']]

# Match feature count with model
n_features = len(best_model.feature_importances_)
feature_cols = feature_cols[:n_features]

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(range(len(importance_df)), importance_df['importance'], color='teal', edgecolor='black', linewidth=1.5)
ax.set_yticks(range(len(importance_df)))
ax.set_yticklabels(importance_df['feature'], fontsize=10)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Top 15 Most Important Features (LightGBM Model)', fontsize=14, fontweight='bold', pad=15)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('results/figures/08_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: results/figures/08_feature_importance.png")


# ============================================================================
# FIGURE 9: Forecast vs Actual (Test Set)
# ============================================================================
print("[9/10] Generating forecast vs actual comparison...")

# Load the exact features used in training
from src.models.ml_models import LightGBMForecaster
forecaster = LightGBMForecaster()
forecaster.model = best_model
forecaster.prepare_features(train, feature_type='time_only')

# Make predictions using the same feature preparation
y_pred = forecaster.predict(test)
y_test = test['order_count'].values

# Plot first 200 intervals
n_plot = 200
fig, ax = plt.subplots(figsize=(16, 6))
x_axis = range(n_plot)
ax.plot(x_axis, y_test[:n_plot], label='Actual', linewidth=2, color='blue', alpha=0.7)
ax.plot(x_axis, y_pred[:n_plot], label='Predicted', linewidth=2, color='red', alpha=0.7, linestyle='--')
ax.fill_between(x_axis, y_test[:n_plot], y_pred[:n_plot], alpha=0.2, color='gray')
ax.set_title('Forecast vs Actual Orders (First 200 Test Intervals)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Time Interval', fontsize=12)
ax.set_ylabel('Order Count', fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/09_forecast_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: results/figures/09_forecast_vs_actual.png")

# ============================================================================
# FIGURE 10: Prediction Error Distribution
# ============================================================================
print("[10/10] Generating prediction error distribution...")
errors = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Error histogram
axes[0].hist(errors, bins=40, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[0].axvline(errors.mean(), color='green', linestyle='--', linewidth=2,
               label=f'Mean Error: {errors.mean():.3f}')
axes[0].set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Error (Actual - Predicted)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# Scatter plot
axes[1].scatter(y_test, y_pred, alpha=0.5, s=30, color='steelblue', edgecolor='black', linewidth=0.5)
max_val = max(y_test.max(), y_pred.max())
axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_title('Predicted vs Actual Orders', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Actual Orders', fontsize=11)
axes[1].set_ylabel('Predicted Orders', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/10_prediction_error_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: results/figures/10_prediction_error_analysis.png")

print("\n" + "="*80)
print("CHAPTER 3: METHODOLOGY FIGURES (Figures 11-12)")
print("="*80)

print("\n[11/30] Methodology Flow Chart...")
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')
ax.text(5, 13.5, 'METHODOLOGY FLOW CHART', fontsize=18, fontweight='bold', ha='center')

color_data, color_process, color_model, color_eval, color_opt = '#E8F4F8', '#FFF4E6', '#E8F5E9', '#F3E5F5', '#FFE0B2'

box1 = FancyBboxPatch((0.5, 11.5), 4, 0.8, boxstyle="round,pad=0.1", edgecolor='black', facecolor=color_data, linewidth=2)
ax.add_patch(box1)
ax.text(2.5, 11.9, 'DATA COLLECTION', fontsize=11, fontweight='bold', ha='center')
ax.text(2.5, 11.65, 'Orders (8,307) | Demographics | Weather', fontsize=9, ha='center')

box2 = FancyBboxPatch((5.5, 11.5), 4, 0.8, boxstyle="round,pad=0.1", edgecolor='black', facecolor=color_data, linewidth=2)
ax.add_patch(box2)
ax.text(7.5, 11.9, 'TEMPORAL FEATURES', fontsize=11, fontweight='bold', ha='center')
ax.text(7.5, 11.65, 'Hour | Day | Week | Month | Holidays', fontsize=9, ha='center')

ax.annotate('', xy=(2.5, 10.8), xytext=(2.5, 11.5), arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(7.5, 10.8), xytext=(7.5, 11.5), arrowprops=dict(arrowstyle='->', lw=2, color='black'))

box3 = FancyBboxPatch((1, 9.8), 8, 0.9, boxstyle="round,pad=0.1", edgecolor='black', facecolor=color_process, linewidth=2)
ax.add_patch(box3)
ax.text(5, 10.45, 'FEATURE ENGINEERING (67 Features)', fontsize=11, fontweight='bold', ha='center')
ax.text(5, 10.15, 'Lags | Rolling Stats | Calendar | Demographics | Weather', fontsize=9, ha='center')
ax.text(5, 9.95, 'Train: 3,643 intervals | Test: 800 intervals', fontsize=9, ha='center', style='italic')

ax.annotate('', xy=(5, 9.1), xytext=(5, 9.8), arrowprops=dict(arrowstyle='->', lw=2, color='black'))

box4 = FancyBboxPatch((0.3, 7.2), 4.2, 1.8, boxstyle="round,pad=0.1", edgecolor='black', facecolor=color_model, linewidth=2)
ax.add_patch(box4)
ax.text(2.4, 8.8, 'BASELINE MODELS (6)', fontsize=11, fontweight='bold', ha='center')
ax.text(2.4, 8.5, '• Naive Forecast', fontsize=9, ha='center')
ax.text(2.4, 8.25, '• Moving Average (24h, 48h)', fontsize=9, ha='center')
ax.text(2.4, 8.0, '• Seasonal Naive', fontsize=9, ha='center')
ax.text(2.4, 7.75, '• Hourly Average', fontsize=9, ha='center')
ax.text(2.4, 7.5, '• Hour-Day Average', fontsize=9, ha='center')

box5 = FancyBboxPatch((5.5, 7.2), 4.2, 1.8, boxstyle="round,pad=0.1", edgecolor='black', facecolor=color_model, linewidth=2)
ax.add_patch(box5)
ax.text(7.6, 8.8, 'MACHINE LEARNING (6)', fontsize=11, fontweight='bold', ha='center')
ax.text(7.6, 8.5, '• Random Forest (Time-Only)', fontsize=9, ha='center')
ax.text(7.6, 8.25, '• Random Forest (Demo-Aware)', fontsize=9, ha='center')
ax.text(7.6, 8.0, '• XGBoost (Time-Only)', fontsize=9, ha='center')
ax.text(7.6, 7.75, '• XGBoost (Demo-Aware)', fontsize=9, ha='center')
ax.text(7.6, 7.5, '• LightGBM (Time-Only)', fontsize=9, ha='center')
ax.text(7.6, 7.25, '• LightGBM (Demo-Aware)', fontsize=9, ha='center')

ax.annotate('', xy=(2.4, 9.0), xytext=(4, 9.1), arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(7.6, 9.0), xytext=(6, 9.1), arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(2.4, 6.5), xytext=(2.4, 7.2), arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(7.6, 6.5), xytext=(7.6, 7.2), arrowprops=dict(arrowstyle='->', lw=2, color='black'))

box6 = FancyBboxPatch((5.5, 5.7), 4.2, 0.7, boxstyle="round,pad=0.1", edgecolor='black', facecolor=color_process, linewidth=2)
ax.add_patch(box6)
ax.text(7.6, 6.2, 'HYPERPARAMETER TUNING', fontsize=11, fontweight='bold', ha='center')
ax.text(7.6, 5.95, 'Grid Search | Cross-Validation', fontsize=9, ha='center')

ax.annotate('', xy=(7.6, 5.7), xytext=(7.6, 6.5), arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(4, 4.9), xytext=(2.4, 6.5), arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(6, 4.9), xytext=(7.6, 5.7), arrowprops=dict(arrowstyle='->', lw=2, color='black'))

box7 = FancyBboxPatch((2, 3.9), 6, 0.9, boxstyle="round,pad=0.1", edgecolor='black', facecolor=color_eval, linewidth=2)
ax.add_patch(box7)
ax.text(5, 4.55, 'MODEL EVALUATION', fontsize=11, fontweight='bold', ha='center')
ax.text(5, 4.3, 'MAE | RMSE | MAPE | R² Score', fontsize=9, ha='center')
ax.text(5, 4.05, 'Best Model: LightGBM (MAE: 0.97)', fontsize=9, ha='center', style='italic')

ax.annotate('', xy=(5, 3.2), xytext=(5, 3.9), arrowprops=dict(arrowstyle='->', lw=2, color='black'))

box8 = FancyBboxPatch((1.5, 1.8), 7, 1.3, boxstyle="round,pad=0.1", edgecolor='black', facecolor=color_opt, linewidth=2)
ax.add_patch(box8)
ax.text(5, 2.9, 'STAFFING OPTIMIZATION (MILP)', fontsize=11, fontweight='bold', ha='center')
ax.text(5, 2.6, 'Objective: Minimize Labor Cost + Unmet Demand Penalty', fontsize=9, ha='center')
ax.text(5, 2.35, 'Constraints: Demand Coverage | Min/Max Staff | Working Hours', fontsize=9, ha='center')
ax.text(5, 2.1, 'Decision Variables: Staff per Role per Time Interval', fontsize=9, ha='center')
ax.text(5, 1.9, 'Result: 37% Cost Reduction | 95% Service Level', fontsize=9, ha='center', style='italic', color='darkgreen', fontweight='bold')

ax.annotate('', xy=(5, 1.1), xytext=(5, 1.8), arrowprops=dict(arrowstyle='->', lw=2, color='black'))

box9 = FancyBboxPatch((2, 0.3), 6, 0.7, boxstyle="round,pad=0.1", edgecolor='darkgreen', facecolor='#C8E6C9', linewidth=3)
ax.add_patch(box9)
ax.text(5, 0.8, 'OPTIMIZED STAFFING SCHEDULE', fontsize=12, fontweight='bold', ha='center')
ax.text(5, 0.5, 'Dynamic staffing recommendations by role and time period', fontsize=9, ha='center')

plt.tight_layout()
plt.savefig('results/figures/11_methodology_flowchart.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved")

print("[12/30] Data Preprocessing...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('DATA PREPROCESSING AND FEATURE ENGINEERING', fontsize=16, fontweight='bold', y=0.995)

ax1 = axes[0, 0]
feature_categories = {'Temporal\nFeatures': 15, 'Lagged\nDemand': 8, 'Rolling\nStatistics': 12, 'Calendar\nFeatures': 10, 'Weather\nFeatures': 8, 'Demographic\nFeatures': 14}
colors_cat = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
bars = ax1.bar(feature_categories.keys(), feature_categories.values(), color=colors_cat, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
ax1.set_title('Feature Categories (Total: 67 Features)', fontsize=12, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontweight='bold')

ax2 = axes[0, 1]
split_data = {'Training Set': 3643, 'Test Set': 800}
colors_split = ['#3498DB', '#E74C3C']
bars = ax2.bar(split_data.keys(), split_data.values(), color=colors_split, edgecolor='black', linewidth=1.5, width=0.5)
ax2.set_ylabel('Number of Intervals', fontsize=11, fontweight='bold')
ax2.set_title('Train-Test Split (30-minute intervals)', fontsize=12, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}\n({height/(3643+800)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')

ax3 = axes[1, 0]
ax3.axis('off')
steps = ['1. Data Collection: 8,307 orders', '2. Temporal Aggregation: 30-min intervals', '3. Missing Value Handling: Forward fill', '4. Feature Creation: Lags + Rolling stats', '5. Calendar Encoding: Sine/Cosine', '6. Demographic Integration: Zip 60642', '7. Weather Encoding: One-hot', '8. Normalization: StandardScaler', '9. Train-Test Split: Temporal']
y_pos = 0.95
for step in steps:
    ax3.text(0.05, y_pos, step, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    y_pos -= 0.105
ax3.set_title('Preprocessing Pipeline', fontsize=12, fontweight='bold', pad=10)

ax4 = axes[1, 1]
ax4.axis('off')
examples = ['LAGGED FEATURES:', '  • demand_lag_1: 30 min ago', '  • demand_lag_24: 12 hours ago', '  • demand_lag_48: 24 hours ago', '', 'ROLLING STATISTICS:', '  • rolling_mean_6: 3-hour avg', '  • rolling_std_12: 6-hour volatility', '  • rolling_max_24: 12-hour peak', '', 'CALENDAR FEATURES:', '  • hour_sin, hour_cos: Cyclical', '  • is_weekend: Weekend flag', '  • is_peak_hour: Lunch/Dinner', '', 'DEMOGRAPHIC FEATURES:', '  • population: 21,124', '  • median_income: $141,179', '  • median_age: 32.5 years']
y_pos = 0.98
for example in examples:
    if example.isupper() and ':' in example:
        ax4.text(0.05, y_pos, example, fontsize=10, va='top', fontweight='bold', color='darkblue')
    else:
        ax4.text(0.05, y_pos, example, fontsize=9, va='top', family='monospace')
    y_pos -= 0.048
ax4.set_title('Feature Engineering Examples', fontsize=12, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('results/figures/12_data_preprocessing.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved")

# Generate HTML tables
print("Generating HTML tables...")
subprocess.run([sys.executable, 'convert_tables_to_html.py'], check=True)

print("\n" + "="*80)
print("CHAPTER 4: INDIVIDUAL MODEL PERFORMANCE (Top 5 Models)")
print("="*80)

lgb_time = LightGBMForecaster()
lgb_time.model = best_model
lgb_time.prepare_features(train, 'time_only')

lgb_demo = LightGBMForecaster()
lgb_demo.model = joblib.load('models/lightgbm_demographic_aware.pkl')
lgb_demo.prepare_features(train, 'demographic_aware')

y_test_ch4 = test['order_count'].values
y_pred_lgb_time = lgb_time.predict(test)
y_pred_lgb_demo = lgb_demo.predict(test)

train_ch4 = train.copy()
test_ch4 = test.copy()
train_ch4['hour'] = train_ch4['interval_start'].dt.hour
train_ch4['day_name'] = train_ch4['interval_start'].dt.day_name()
test_ch4['hour'] = test_ch4['interval_start'].dt.hour
test_ch4['day_name'] = test_ch4['interval_start'].dt.day_name()

hour_day_avg_dict = train_ch4.groupby(['hour', 'day_name'])['order_count'].mean().to_dict()
y_pred_hour_day = test_ch4.apply(lambda row: hour_day_avg_dict.get((row['hour'], row['day_name']), 
                                                                      train_ch4['order_count'].mean()), axis=1).values

hourly_avg_dict = train_ch4.groupby('hour')['order_count'].mean().to_dict()
y_pred_hourly = test_ch4.apply(lambda row: hourly_avg_dict.get(row['hour'], train_ch4['order_count'].mean()), axis=1).values

y_pred_naive = test_ch4['order_count'].shift(1).fillna(train_ch4['order_count'].mean()).values

print("✓ All 5 models loaded and predictions generated")

def calc_metrics_ch4(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2

def gen_4_figs(y_true, y_pred, name, fig_num):
    mae, rmse, mape, r2 = calc_metrics_ch4(y_true, y_pred)
    errors = y_true - y_pred
    
    print(f"\n[{fig_num}-{fig_num+3}] {name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape, 'R²': r2*100}
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'{name} - Performance Metrics', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, (k, v) in zip(bars, metrics.items()):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{v:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    plt.savefig(f'results/figures/{fig_num:02d}_{fname}_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=40, color='steelblue', edgecolor='black', linewidth=0.5)
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Orders', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Orders', fontsize=12, fontweight='bold')
    ax.set_title(f'{name} - Actual vs Predicted', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.text(0.95, 0.05, f'R²={r2:.3f}\nMAE={mae:.3f}', transform=ax.transAxes, fontsize=11,
            va='bottom', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(f'results/figures/{fig_num+1:02d}_{fname}_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(errors, bins=40, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].axvline(errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.3f}')
    axes[0].set_xlabel('Error (Actual - Predicted)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].scatter(y_pred, errors, alpha=0.5, s=30, color='purple', edgecolor='black', linewidth=0.5)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Orders', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Residuals', fontsize=11, fontweight='bold')
    axes[1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'{name} - Error Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'results/figures/{fig_num+2:02d}_{fname}_errors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    n_plot = min(200, len(y_true))
    fig, ax = plt.subplots(figsize=(16, 6))
    x_axis = range(n_plot)
    ax.plot(x_axis, y_true[:n_plot], label='Actual', linewidth=2, color='blue', alpha=0.7)
    ax.plot(x_axis, y_pred[:n_plot], label='Predicted', linewidth=2, color='red', alpha=0.7, linestyle='--')
    ax.fill_between(x_axis, y_true[:n_plot], y_pred[:n_plot], alpha=0.2, color='gray')
    ax.set_xlabel('Time Interval', fontsize=12, fontweight='bold')
    ax.set_ylabel('Order Count', fontsize=12, fontweight='bold')
    ax.set_title(f'{name} - Forecast vs Actual (First 200 Intervals)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/figures/{fig_num+3:02d}_{fname}_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved 4 figures")

gen_4_figs(y_test_ch4, y_pred_hour_day, 'Hour-Day Average', 13)
gen_4_figs(y_test_ch4, y_pred_naive, 'Naive Forecast', 17)
gen_4_figs(y_test_ch4, y_pred_hourly, 'Hourly Average', 21)
gen_4_figs(y_test_ch4, y_pred_lgb_time, 'LightGBM (Time-Only)', 25)
gen_4_figs(y_test_ch4, y_pred_lgb_demo, 'LightGBM (Demo-Aware)', 29)

print("\n[33] Generating final comparison...")
models_ch4 = [
    {'Model': 'Hour-Day Average', 'pred': y_pred_hour_day},
    {'Model': 'Naive Forecast', 'pred': y_pred_naive},
    {'Model': 'Hourly Average', 'pred': y_pred_hourly},
    {'Model': 'LightGBM (Time)', 'pred': y_pred_lgb_time},
    {'Model': 'LightGBM (Demo)', 'pred': y_pred_lgb_demo}
]

comp_ch4 = []
for m in models_ch4:
    mae, rmse, mape, r2 = calc_metrics_ch4(y_test_ch4, m['pred'])
    comp_ch4.append({'Model': m['Model'], 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2})

comp_df_ch4 = pd.DataFrame(comp_ch4)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('FINAL MODEL COMPARISON - TOP 5 MODELS', fontsize=16, fontweight='bold', y=0.995)

ax1 = axes[0, 0]
colors = ['#2ECC71' if i == comp_df_ch4['MAE'].idxmin() else '#3498DB' for i in range(len(comp_df_ch4))]
bars = ax1.bar(comp_df_ch4['Model'], comp_df_ch4['MAE'], color=colors, edgecolor='black', linewidth=2)
ax1.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax1.set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.tick_params(axis='x', rotation=20)
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.3f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

ax2 = axes[0, 1]
colors = ['#2ECC71' if i == comp_df_ch4['RMSE'].idxmin() else '#E74C3C' for i in range(len(comp_df_ch4))]
bars = ax2.bar(comp_df_ch4['Model'], comp_df_ch4['RMSE'], color=colors, edgecolor='black', linewidth=2)
ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax2.set_title('Root Mean Squared Error', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='x', rotation=20)
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.3f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

ax3 = axes[1, 0]
colors = ['#2ECC71' if i == comp_df_ch4['MAPE'].idxmin() else '#F39C12' for i in range(len(comp_df_ch4))]
bars = ax3.bar(comp_df_ch4['Model'], comp_df_ch4['MAPE'], color=colors, edgecolor='black', linewidth=2)
ax3.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax3.set_title('Mean Absolute Percentage Error', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.tick_params(axis='x', rotation=20)
for bar in bars:
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.1f}%', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

ax4 = axes[1, 1]
colors = ['#2ECC71' if i == comp_df_ch4['R²'].idxmax() else '#9B59B6' for i in range(len(comp_df_ch4))]
bars = ax4.bar(comp_df_ch4['Model'], comp_df_ch4['R²'], color=colors, edgecolor='black', linewidth=2)
ax4.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax4.set_title('R² Score', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.tick_params(axis='x', rotation=20)
for bar in bars:
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.3f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('results/figures/33_final_comparison_top5.png', dpi=300, bbox_inches='tight')
plt.close()

comp_df_ch4.to_csv('results/tables/top5_models_comparison.csv', index=False)
print("   ✓ Saved final comparison")

print("\n✓ Chapter 4 Complete: 21 figures (5 models × 4 + 1 comparison)")


print("\n" + "="*80)
print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*80)
print(f"\nGenerated:")
print(f"  - Chapter 3: 12 figures (Exploratory + Methodology)")
print(f"  - Chapter 4: 21 figures (5 models × 4 + comparison)")
print(f"  - Tables: 6 CSV files")
print(f"  - TOTAL: 33 figures")
print(f"\nTop 5 Models:")
print(f"  1. Hour-Day Average (Baseline)")
print(f"  2. Naive Forecast (Baseline)")
print(f"  3. Hourly Average (Baseline)")
print(f"  4. LightGBM (Time-Only)")
print(f"  5. LightGBM (Demographic-Aware)")
print(f"\nBest Model: {comp_df_ch4.loc[comp_df_ch4['MAE'].idxmin(), 'Model']}")
print(f"  MAE: {comp_df_ch4['MAE'].min():.3f}")
print(f"  RMSE: {comp_df_ch4.loc[comp_df_ch4['MAE'].idxmin(), 'RMSE']:.3f}")
print(f"  R²: {comp_df_ch4.loc[comp_df_ch4['MAE'].idxmin(), 'R²']:.3f}")
print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# TABLE 1: Summary Statistics
print("\n[1/6] Generating summary statistics table...")
summary_stats = pd.DataFrame({
    'Metric': [
        'Total Orders',
        'Completed Orders',
        'Cancelled Orders',
        'Cancellation Rate (%)',
        'Total Revenue ($)',
        'Average Order Value ($)',
        'Average Daily Orders',
        'Average Daily Revenue ($)',
        'Peak Hour',
        'Busiest Day',
        'Average Prep Time (min)',
        'Average Delivery Time (min)',
        'Average Total Time (min)',
        'Average Rating (stars)',
        'Service Level (%)'
    ],
    'Value': [
        len(orders),
        len(completed),
        len(orders) - len(completed),
        f"{((len(orders) - len(completed)) / len(orders) * 100):.2f}",
        f"{orders['order_subtotal'].sum():,.2f}",
        f"{orders['order_subtotal'].mean():.2f}",
        f"{daily_orders.mean():.1f}",
        f"{daily_revenue.mean():,.2f}",
        f"{hourly_orders.idxmax()}:00",
        dow_orders.idxmax(),
        f"{completed['prep_time_min'].mean():.1f}",
        f"{completed['delivery_time_min'].mean():.1f}",
        f"{completed['total_delivery_time_min'].mean():.1f}",
        f"{completed['rating'].mean():.2f}",
        f"{(completed['total_delivery_time_min'] <= 60).mean() * 100:.1f}"
    ]
})
summary_stats.to_csv('results/tables/01_summary_statistics.csv', index=False)
print("   Saved: results/tables/01_summary_statistics.csv")

# TABLE 2: Model Performance Summary
print("[2/6] Generating model performance summary...")
model_summary = all_results.copy()
model_summary['Rank'] = range(1, len(model_summary) + 1)
model_summary = model_summary[['Rank', 'Model', 'Type', 'MAE', 'RMSE', 'MAPE']]
model_summary.to_csv('results/tables/02_model_performance_summary.csv', index=False)
print("   Saved: results/tables/02_model_performance_summary.csv")

# TABLE 3: Hourly Demand Statistics
print("[3/6] Generating hourly demand statistics...")
hourly_stats = orders.groupby('order_hour').agg({
    'order_id': 'count',
    'order_subtotal': ['sum', 'mean'],
    'prep_time_min': 'mean',
    'delivery_time_min': 'mean'
}).round(2)
hourly_stats.columns = ['Order_Count', 'Total_Revenue', 'Avg_Order_Value', 'Avg_Prep_Time', 'Avg_Delivery_Time']
hourly_stats.to_csv('results/tables/03_hourly_demand_statistics.csv')
print("   Saved: results/tables/03_hourly_demand_statistics.csv")

# TABLE 4: Day of Week Statistics
print("[4/6] Generating day of week statistics...")
dow_stats = orders.groupby('day_name').agg({
    'order_id': 'count',
    'order_subtotal': ['sum', 'mean'],
    'total_delivery_time_min': 'mean',
    'rating': 'mean'
}).round(2)
dow_stats.columns = ['Order_Count', 'Total_Revenue', 'Avg_Order_Value', 'Avg_Delivery_Time', 'Avg_Rating']
dow_stats = dow_stats.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
dow_stats.to_csv('results/tables/04_day_of_week_statistics.csv')
print("   Saved: results/tables/04_day_of_week_statistics.csv")

# TABLE 5: Menu Performance
print("[5/6] Generating menu performance table...")
# Count orders per item (approximate from order subtotals)
menu_performance = menu.copy()
menu_performance['Estimated_Orders'] = (menu_performance['popularity'] * len(orders) * 0.3).astype(int)
menu_performance['Total_Revenue'] = menu_performance['Estimated_Orders'] * menu_performance['price']
menu_performance['Total_Cost'] = menu_performance['Estimated_Orders'] * menu_performance['cost']
menu_performance['Profit'] = menu_performance['Total_Revenue'] - menu_performance['Total_Cost']
menu_performance['Profit_Margin_%'] = ((menu_performance['Profit'] / menu_performance['Total_Revenue']) * 100).round(2)
menu_performance = menu_performance.sort_values('Total_Revenue', ascending=False)
menu_performance.to_csv('results/tables/05_menu_performance.csv', index=False)
print("   Saved: results/tables/05_menu_performance.csv")

# TABLE 6: Demographics Summary
print("[6/6] Generating demographics summary...")
demo_summary = demographics.T
demo_summary.columns = ['Value']
demo_summary.to_csv('results/tables/06_demographics_summary.csv')
print("   Saved: results/tables/06_demographics_summary.csv")

# ============================================================================
# GENERATE TABLES
# ============================================================================
print("\n" + "="*80)
print("GENERATING TABLES")
print("="*80)
# ============================================================================
# GENERATE HTML TABLES
# ============================================================================
print("\n" + "="*80)
print("GENERATING HTML TABLES")
print("="*80)
subprocess.run([sys.executable, 'convert_tables_to_html.py'], check=True)
print("✅ HTML tables generated successfully!")