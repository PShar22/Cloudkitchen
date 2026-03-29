"""
Compare all forecasting models (baseline + ML)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_all_models():
    """Load and compare all model results"""
    
    print("="*70)
    print("Complete Model Comparison")
    print("="*70)
    
    # Load results
    baseline_results = pd.read_csv('results/tables/baseline_results.csv')
    ml_results = pd.read_csv('results/tables/ml_results.csv')
    
    # Add model type
    baseline_results['Type'] = 'Baseline'
    baseline_results['Feature_Type'] = 'time_only'
    ml_results['Type'] = 'Machine Learning'
    
    # Combine
    all_results = pd.concat([
        baseline_results[['Model', 'Type', 'Feature_Type', 'MAE', 'RMSE', 'MAPE']],
        ml_results[['Model', 'Type', 'Feature_Type', 'MAE', 'RMSE', 'MAPE']]
    ], ignore_index=True)
    
    # Sort by MAE
    all_results = all_results.sort_values('MAE')
    
    print("\nAll Models Ranked by MAE:")
    print("-"*70)
    print(all_results.to_string(index=False))
    
    # Save combined results
    all_results.to_csv('results/tables/all_models_comparison.csv', index=False)
    print("\nSaved: results/tables/all_models_comparison.csv")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # MAE comparison
    top_models = all_results.head(10)
    axes[0].barh(range(len(top_models)), top_models['MAE'], 
                 color=['steelblue' if t == 'Baseline' else 'coral' for t in top_models['Type']])
    axes[0].set_yticks(range(len(top_models)))
    axes[0].set_yticklabels([f"{row['Model']}\n({row['Feature_Type']})" 
                              for _, row in top_models.iterrows()], fontsize=9)
    axes[0].set_xlabel('MAE (Mean Absolute Error)')
    axes[0].set_title('Top 10 Models by MAE', fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # RMSE comparison
    axes[1].barh(range(len(top_models)), top_models['RMSE'],
                 color=['steelblue' if t == 'Baseline' else 'coral' for t in top_models['Type']])
    axes[1].set_yticks(range(len(top_models)))
    axes[1].set_yticklabels([f"{row['Model']}\n({row['Feature_Type']})" 
                              for _, row in top_models.iterrows()], fontsize=9)
    axes[1].set_xlabel('RMSE (Root Mean Squared Error)')
    axes[1].set_title('Top 10 Models by RMSE', fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # MAPE comparison
    axes[2].barh(range(len(top_models)), top_models['MAPE'],
                 color=['steelblue' if t == 'Baseline' else 'coral' for t in top_models['Type']])
    axes[2].set_yticks(range(len(top_models)))
    axes[2].set_yticklabels([f"{row['Model']}\n({row['Feature_Type']})" 
                              for _, row in top_models.iterrows()], fontsize=9)
    axes[2].set_xlabel('MAPE (%)')
    axes[2].set_title('Top 10 Models by MAPE', fontweight='bold')
    axes[2].invert_yaxis()
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('results/figures/all_models_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/figures/all_models_comparison.png")
    plt.show()
    
    # Best model summary
    best = all_results.iloc[0]
    print("\n" + "="*70)
    print("BEST OVERALL MODEL")
    print("="*70)
    print(f"Model: {best['Model']}")
    print(f"Type: {best['Type']}")
    print(f"Features: {best['Feature_Type']}")
    print(f"MAE:  {best['MAE']:.4f}")
    print(f"RMSE: {best['RMSE']:.4f}")
    print(f"MAPE: {best['MAPE']:.2f}%")
    print("="*70)
    
    # Calculate improvement over baseline
    best_baseline = baseline_results.iloc[0]
    mae_improvement = ((best_baseline['MAE'] - best['MAE']) / best_baseline['MAE']) * 100
    rmse_improvement = ((best_baseline['RMSE'] - best['RMSE']) / best_baseline['RMSE']) * 100
    mape_improvement = ((best_baseline['MAPE'] - best['MAPE']) / best_baseline['MAPE']) * 100
    
    print(f"\nImprovement over best baseline ({best_baseline['Model']}):")
    print(f"  MAE:  {mae_improvement:+.2f}%")
    print(f"  RMSE: {rmse_improvement:+.2f}%")
    print(f"  MAPE: {mape_improvement:+.2f}%")
    
    return all_results

if __name__ == "__main__":
    results = compare_all_models()
