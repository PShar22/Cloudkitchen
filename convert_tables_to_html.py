"""
Convert all CSV tables to professional HTML format
"""

import pandas as pd
import os

def csv_to_html(csv_path, html_path, table_title=""):
    """Convert CSV to styled HTML table"""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Create HTML with professional styling
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .table-container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 24px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 0 auto;
            font-size: 14px;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border: 1px solid #2980b9;
        }}
        td {{
            padding: 10px 12px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        .numeric {{
            text-align: right;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="table-container">
        <h2>{table_title}</h2>
        {df.to_html(index=False, classes='data-table', border=0)}
    </div>
</body>
</html>
"""
    
    # Save HTML
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Converted: {os.path.basename(csv_path)} → {os.path.basename(html_path)}")

# Create HTML tables directory
os.makedirs('results/tables_html', exist_ok=True)

print("="*70)
print("Converting CSV Tables to HTML")
print("="*70)

# Define tables with titles
tables = [
    ('results/tables/01_summary_statistics.csv', 
     'results/tables_html/01_summary_statistics.html',
     'Table 1: Summary Statistics of Cloud Kitchen Operations'),
    
    ('results/tables/02_model_performance_summary.csv',
     'results/tables_html/02_model_performance_summary.html',
     'Table 2: Model Performance Summary (All Models)'),
    
    ('results/tables/03_hourly_demand_statistics.csv',
     'results/tables_html/03_hourly_demand_statistics.html',
     'Table 3: Hourly Demand Statistics'),
    
    ('results/tables/04_day_of_week_statistics.csv',
     'results/tables_html/04_day_of_week_statistics.html',
     'Table 4: Day of Week Performance Statistics'),
    
    ('results/tables/05_menu_performance.csv',
     'results/tables_html/05_menu_performance.html',
     'Table 5: Menu Item Performance Analysis'),
    
    ('results/tables/06_demographics_summary.csv',
     'results/tables_html/06_demographics_summary.html',
     'Table 6: Demographic Profile of Zip Code 60642'),
    
    ('results/tables/baseline_results.csv',
     'results/tables_html/baseline_results.html',
     'Table 7: Baseline Model Results'),
    
    ('results/tables/ml_results.csv',
     'results/tables_html/ml_results.html',
     'Table 8: Machine Learning Model Results'),
    
    ('results/tables/all_models_comparison.csv',
     'results/tables_html/all_models_comparison.html',
     'Table 9: Complete Model Comparison (Baseline + ML)')
]

# Convert all tables
for csv_path, html_path, title in tables:
    if os.path.exists(csv_path):
        csv_to_html(csv_path, html_path, title)
    else:
        print(f"✗ Not found: {csv_path}")

print("\n" + "="*70)
print("✅ All tables converted to HTML!")
print("="*70)
print(f"\nHTML tables saved in: results/tables_html/")
print("\nYou can now copy the HTML code from these files")
print("and paste directly into your Word document.")
