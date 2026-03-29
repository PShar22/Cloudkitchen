# Demographic-Aware Predictive Optimization for Cloud Kitchen Operations

MSc Data Science Final Project

## Overview

This project develops a demographic-aware machine learning framework for cloud kitchen demand forecasting and staffing optimization. The framework combines 5 forecasting models (3 baseline + 2 LightGBM) with mixed-integer linear programming for optimal staffing decisions.

**Key Results:** Best model achieves MAE of 0.890 (Hour-Day Average baseline). Optimization reduces labor costs by 37% while improving service levels from 90% to 95%.

## Required Files Before Running

Before running the main script, ensure these files and directories exist:

### Data Files (Must Exist)
```
data/
├── external/
│   └── demographics_60642.csv
├── processed/
│   ├── features_full.csv
│   ├── features_test.csv
│   └── features_train.csv
└── raw/
    ├── kitchen_info.csv
    ├── menu_items.csv
    └── orders_synthetic.csv
```

### Model Files (Must Exist)
```
models/
├── lightgbm_demographic_aware.pkl
└── lightgbm_time_only.pkl
```

### Source Code (Must Exist)
```
src/
├── data/
├── models/
│   └── ml_models.py (contains LightGBMForecaster class)
└── optimization/
```

### Auto-Generated Files
The script automatically creates and manages:
- `results/` directories (created automatically)
- `results/tables/baseline_results.csv` (generated if missing)
- `results/tables/ml_results.csv` (generated if missing)

## What Gets Generated

When you run `generate_all_visualizations.py`, the following will be created:

### Figures (33 PNG files)
```
results/figures/
├── 01_daily_order_volume.png
├── 02_hourly_demand_patterns.png
├── 03_day_of_week_analysis.png
├── 04_weather_impact.png
├── 05_delivery_performance.png
├── 06_revenue_analysis.png
├── 07_model_performance_comparison.png
├── 08_feature_importance.png
├── 09_forecast_vs_actual.png
├── 10_prediction_error_analysis.png
├── 11_methodology_flowchart.png
├── 12_data_preprocessing.png
├── 13-16_hour_day_average_*.png (4 files)
├── 17-20_naive_forecast_*.png (4 files)
├── 21-24_hourly_average_*.png (4 files)
├── 25-28_lightgbm_time_only_*.png (4 files)
├── 29-32_lightgbm_demo_aware_*.png (4 files)
└── 33_final_comparison_top5.png
```

### Tables (11 CSV files)
```
results/tables/
├── 01_summary_statistics.csv
├── 02_model_performance_summary.csv
├── 03_hourly_demand_statistics.csv
├── 04_day_of_week_statistics.csv
├── 05_menu_performance.csv
├── 06_demographics_summary.csv
├── all_models_comparison.csv
├── baseline_results.csv (if not exists)
├── final_comparison.csv
├── ml_results.csv (if not exists)
└── top5_models_comparison.csv
```

### HTML Tables (8 HTML files)
```
results/tables_html/
├── 01_summary_statistics.html
├── 02_model_performance_summary.html
├── 03_hourly_demand_statistics.html
├── 04_day_of_week_statistics.html
├── 05_menu_performance.html
├── 06_demographics_summary.html
├── all_models_comparison.html
├── baseline_results.html
└── ml_results.html
```

## Project Structure

```
├── data/                       # Raw and processed datasets
├── src/                        # Source code (data, models, optimization)
├── notebooks/                  # Jupyter notebooks for analysis
├── models/                     # Trained models (2 LightGBM models)
├── results/                    # Generated figures and tables
├── COMPLETE_VISUALIZATIONS_REPORT.ipynb  # All figures and tables
├── generate_all_visualizations.py  # Main script (generates everything)
└── convert_tables_to_html.py   # Table conversion utility
```

## Requirements

### Python Dependencies

**Option 1: Install from requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option 2: Install minimal dependencies manually**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm joblib
```

### Automatic Setup
The script automatically handles all setup:
- ✅ Creates required directories (`results/figures/`, `results/tables/`, `results/tables_html/`)
- ✅ Runs baseline models if `baseline_results.csv` doesn't exist
- ✅ Runs ML models if `ml_results.csv` doesn't exist  
- ✅ Converts tables to HTML format
- ✅ Generates all 33 figures and 11 tables in one run

## Quick Start

### Single Command (Recommended)
```bash
# This one command does EVERYTHING:
python generate_all_visualizations.py
```

The script will automatically:
1. Create all required directories
2. Check for and generate missing baseline/ML results
3. Generate all 33 figures
4. Generate all 11 CSV tables  
5. Convert tables to HTML format

### Manual Step-by-Step (if needed)
```bash
# Only if you prefer manual control:
mkdir -p results/figures results/tables results/tables_html
python src/models/baseline_models.py
python src/models/ml_models.py
python generate_all_visualizations.py
python convert_tables_to_html.py
```

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'src.models.ml_models'"**
   - Solution: Ensure you're running from the project root directory (where `src/` folder is located)

2. **"FileNotFoundError: data/raw/orders_synthetic.csv"**
   - Solution: Ensure all required data files exist in the correct directory structure

3. **Script stops with subprocess error**
   - Solution: Run the individual scripts manually if automatic execution fails:
     ```bash
     python src/models/baseline_models.py
     python src/models/ml_models.py
     python generate_all_visualizations.py
     ```

### Verification
After successful run, you should have:
- 33 PNG files in `results/figures/`
- 11 CSV files in `results/tables/`
- 8 HTML files in `results/tables_html/`

The script will show ✅ success messages for each step and a final summary.

## Key Results

### Top 5 Models Performance

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|----|
| Hour-Day Average | 0.890 | 1.134 | 51.46% | 0.186 |
| Hourly Average | 0.907 | 1.153 | 52.61% | 0.157 |
| LightGBM (Time) | 0.968 | 1.208 | 61.82% | 0.076 |
| LightGBM (Demo) | 0.973 | 1.210 | 61.22% | 0.072 |
| Naive Forecast | 1.152 | 1.637 | 62.09% | -0.698 |

### Optimization Results

| Metric | Heuristic | Optimized | Improvement |
|--------|-----------|-----------|-------------|
| Weekly Cost | $10,605 | $6,675 | -37% |
| Cost/Order | $8.16 | $5.14 | -37% |
| Utilization | 27% | 58% | +31% |
| Service Level | 90% | 95% | +5% |
