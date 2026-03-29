"""
Staffing Optimization Module
Mixed-Integer Linear Programming for cloud kitchen staffing decisions
"""

import pandas as pd
import numpy as np
from pulp import *
import json
from datetime import datetime, timedelta

class StaffingOptimizer:
    """Optimize staffing schedules based on demand forecasts"""
    
    def __init__(self, kitchen_config=None):
        """
        Initialize optimizer with kitchen configuration
        
        Parameters:
        - kitchen_config: dict with kitchen parameters
        """
        if kitchen_config is None:
            kitchen_config = {
                'capacity_per_chef': 3,  # orders per 30-min interval per chef
                'capacity_per_prep': 5,  # prep support capacity
                'chef_hourly_cost': 20,  # $/hour
                'prep_hourly_cost': 15,  # $/hour
                'overtime_multiplier': 1.5,
                'min_chefs': 1,
                'max_chefs': 5,
                'min_prep': 0,
                'max_prep': 3,
                'shift_length': 8,  # hours
                'max_hours_per_week': 40,
                'service_level_target': 0.95  # 95% on-time
            }
        
        self.config = kitchen_config
        self.solution = None
        
    def optimize_schedule(self, demand_forecast, time_periods, 
                         objective='minimize_cost', verbose=True):
        """
        Optimize staffing schedule
        
        Parameters:
        - demand_forecast: array of predicted order counts per interval
        - time_periods: list of time period identifiers
        - objective: 'minimize_cost' or 'maximize_service'
        - verbose: print solver output
        
        Returns:
        - solution dict with staffing decisions and metrics
        """
        
        n_periods = len(demand_forecast)
        
        if verbose:
            print("="*60)
            print("Staffing Optimization Problem")
            print("="*60)
            print(f"Time periods: {n_periods}")
            print(f"Total forecasted demand: {sum(demand_forecast):.0f} orders")
            print(f"Objective: {objective}")
        
        # Create optimization problem
        prob = LpProblem("Cloud_Kitchen_Staffing", LpMinimize)
        
        # Decision variables
        # Number of chefs in each period
        chefs = LpVariable.dicts("chefs", range(n_periods), 
                                lowBound=self.config['min_chefs'],
                                upBound=self.config['max_chefs'],
                                cat='Integer')
        
        # Number of prep staff in each period
        prep = LpVariable.dicts("prep", range(n_periods),
                               lowBound=self.config['min_prep'],
                               upBound=self.config['max_prep'],
                               cat='Integer')
        
        # Unmet demand (slack variable for service level)
        unmet = LpVariable.dicts("unmet", range(n_periods),
                                lowBound=0,
                                cat='Continuous')
        
        # Overtime hours
        overtime = LpVariable.dicts("overtime", range(n_periods),
                                   lowBound=0,
                                   cat='Continuous')
        
        # Objective function: Minimize total cost
        labor_cost = lpSum([
            chefs[t] * self.config['chef_hourly_cost'] * 0.5 +  # 0.5 hours per 30-min interval
            prep[t] * self.config['prep_hourly_cost'] * 0.5
            for t in range(n_periods)
        ])
        
        overtime_cost = lpSum([
            overtime[t] * self.config['chef_hourly_cost'] * self.config['overtime_multiplier']
            for t in range(n_periods)
        ])
        
        # Penalty for unmet demand (high cost to encourage meeting demand)
        unmet_penalty = lpSum([
            unmet[t] * 50  # $50 per unmet order (represents lost revenue + reputation)
            for t in range(n_periods)
        ])
        
        prob += labor_cost + overtime_cost + unmet_penalty, "Total_Cost"
        
        # Constraints
        for t in range(n_periods):
            # Capacity constraint: staff capacity must meet demand
            capacity = (chefs[t] * self.config['capacity_per_chef'] + 
                       prep[t] * self.config['capacity_per_prep'])
            
            prob += capacity + unmet[t] >= demand_forecast[t], f"Demand_Coverage_{t}"
            
            # Service level constraint: limit unmet demand
            prob += unmet[t] <= demand_forecast[t] * (1 - self.config['service_level_target']), f"Service_Level_{t}"
        
        # Solve
        if verbose:
            print("\nSolving optimization problem...")
        
        solver = PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        
        # Extract solution
        if LpStatus[prob.status] == 'Optimal':
            if verbose:
                print(f"Status: {LpStatus[prob.status]}")
                print(f"Optimal cost: ${value(prob.objective):,.2f}")
            
            solution = {
                'status': 'Optimal',
                'objective_value': value(prob.objective),
                'chefs': [int(chefs[t].varValue) for t in range(n_periods)],
                'prep': [int(prep[t].varValue) for t in range(n_periods)],
                'unmet_demand': [unmet[t].varValue for t in range(n_periods)],
                'demand_forecast': demand_forecast.tolist() if isinstance(demand_forecast, np.ndarray) else demand_forecast,
                'time_periods': time_periods,
                'total_labor_cost': value(labor_cost),
                'total_overtime_cost': value(overtime_cost),
                'total_unmet_penalty': value(unmet_penalty)
            }
            
            # Calculate metrics
            solution['metrics'] = self._calculate_metrics(solution)
            
            self.solution = solution
            
            if verbose:
                self._print_solution_summary(solution)
            
            return solution
        else:
            print(f"Optimization failed: {LpStatus[prob.status]}")
            return None
    
    def _calculate_metrics(self, solution):
        """Calculate operational metrics from solution"""
        
        total_demand = sum(solution['demand_forecast'])
        total_unmet = sum(solution['unmet_demand'])
        total_met = total_demand - total_unmet
        
        total_chef_hours = sum(solution['chefs']) * 0.5  # 30-min intervals
        total_prep_hours = sum(solution['prep']) * 0.5
        total_staff_hours = total_chef_hours + total_prep_hours
        
        # Calculate capacity
        total_capacity = sum([
            solution['chefs'][t] * self.config['capacity_per_chef'] +
            solution['prep'][t] * self.config['capacity_per_prep']
            for t in range(len(solution['chefs']))
        ])
        
        metrics = {
            'total_demand': total_demand,
            'total_met_demand': total_met,
            'total_unmet_demand': total_unmet,
            'service_level': (total_met / total_demand * 100) if total_demand > 0 else 100,
            'total_staff_hours': total_staff_hours,
            'total_chef_hours': total_chef_hours,
            'total_prep_hours': total_prep_hours,
            'avg_chefs_per_interval': np.mean(solution['chefs']),
            'avg_prep_per_interval': np.mean(solution['prep']),
            'total_capacity': total_capacity,
            'capacity_utilization': (total_demand / total_capacity * 100) if total_capacity > 0 else 0,
            'labor_cost_per_order': (solution['total_labor_cost'] / total_met) if total_met > 0 else 0,
            'total_cost': solution['objective_value']
        }
        
        return metrics
    
    def _print_solution_summary(self, solution):
        """Print solution summary"""
        metrics = solution['metrics']
        
        print("\n" + "="*60)
        print("OPTIMIZATION SOLUTION SUMMARY")
        print("="*60)
        
        print("\nStaffing:")
        print(f"  Average chefs per interval: {metrics['avg_chefs_per_interval']:.2f}")
        print(f"  Average prep staff per interval: {metrics['avg_prep_per_interval']:.2f}")
        print(f"  Total chef hours: {metrics['total_chef_hours']:.1f}")
        print(f"  Total prep hours: {metrics['total_prep_hours']:.1f}")
        
        print("\nDemand Coverage:")
        print(f"  Total demand: {metrics['total_demand']:.0f} orders")
        print(f"  Met demand: {metrics['total_met_demand']:.0f} orders")
        print(f"  Unmet demand: {metrics['total_unmet_demand']:.0f} orders")
        print(f"  Service level: {metrics['service_level']:.2f}%")
        
        print("\nCapacity:")
        print(f"  Total capacity: {metrics['total_capacity']:.0f} orders")
        print(f"  Utilization: {metrics['capacity_utilization']:.2f}%")
        
        print("\nCosts:")
        print(f"  Labor cost: ${solution['total_labor_cost']:,.2f}")
        print(f"  Overtime cost: ${solution['total_overtime_cost']:,.2f}")
        print(f"  Unmet penalty: ${solution['total_unmet_penalty']:,.2f}")
        print(f"  Total cost: ${metrics['total_cost']:,.2f}")
        print(f"  Cost per order: ${metrics['labor_cost_per_order']:.2f}")
        
        print("="*60)
    
    def create_schedule_dataframe(self, solution=None):
        """Convert solution to readable schedule dataframe"""
        if solution is None:
            solution = self.solution
        
        if solution is None:
            print("No solution available. Run optimize_schedule first.")
            return None
        
        schedule = pd.DataFrame({
            'time_period': solution['time_periods'],
            'forecasted_demand': solution['demand_forecast'],
            'chefs': solution['chefs'],
            'prep_staff': solution['prep'],
            'total_staff': [c + p for c, p in zip(solution['chefs'], solution['prep'])],
            'capacity': [
                solution['chefs'][i] * self.config['capacity_per_chef'] +
                solution['prep'][i] * self.config['capacity_per_prep']
                for i in range(len(solution['chefs']))
            ],
            'unmet_demand': solution['unmet_demand']
        })
        
        schedule['utilization'] = (schedule['forecasted_demand'] / schedule['capacity'] * 100).round(2)
        
        return schedule
    
    def save_solution(self, filepath, solution=None):
        """Save solution to JSON file"""
        if solution is None:
            solution = self.solution
        
        if solution is None:
            print("No solution to save.")
            return
        
        with open(filepath, 'w') as f:
            json.dump(solution, f, indent=2)
        
        print(f"Solution saved: {filepath}")


def compare_heuristic_vs_optimized(test_data, forecast_col='order_count', 
                                   actual_col='order_count'):
    """
    Compare heuristic staffing vs optimized staffing
    
    Heuristic: Simple rule-based staffing (e.g., fixed staff or simple thresholds)
    Optimized: MILP-based optimal staffing
    """
    
    print("="*70)
    print("Comparing Heuristic vs Optimized Staffing")
    print("="*70)
    
    # Load test data
    test = pd.read_csv(test_data, parse_dates=['interval_start'])
    demand = test[forecast_col].values
    time_periods = test['interval_start'].astype(str).tolist()
    
    # Heuristic approach: Fixed staffing based on average demand
    avg_demand = demand.mean()
    heuristic_chefs = int(np.ceil(avg_demand / 3))  # 3 orders per chef
    heuristic_prep = 1  # Fixed 1 prep staff
    
    heuristic_cost = len(demand) * 0.5 * (heuristic_chefs * 20 + heuristic_prep * 15)
    heuristic_capacity = (heuristic_chefs * 3 + heuristic_prep * 5) * len(demand)
    heuristic_utilization = sum(demand) / heuristic_capacity * 100
    
    print(f"\nHeuristic Approach:")
    print(f"  Fixed staffing: {heuristic_chefs} chefs, {heuristic_prep} prep")
    print(f"  Total cost: ${heuristic_cost:,.2f}")
    print(f"  Utilization: {heuristic_utilization:.2f}%")
    print(f"  Cost per order: ${heuristic_cost / sum(demand):.2f}")
    
    # Optimized approach
    print(f"\nOptimized Approach:")
    optimizer = StaffingOptimizer()
    solution = optimizer.optimize_schedule(demand, time_periods, verbose=False)
    
    if solution:
        print(f"  Dynamic staffing: {solution['metrics']['avg_chefs_per_interval']:.2f} avg chefs, "
              f"{solution['metrics']['avg_prep_per_interval']:.2f} avg prep")
        print(f"  Total cost: ${solution['metrics']['total_cost']:,.2f}")
        print(f"  Utilization: {solution['metrics']['capacity_utilization']:.2f}%")
        print(f"  Cost per order: ${solution['metrics']['labor_cost_per_order']:.2f}")
        
        # Calculate savings
        cost_savings = heuristic_cost - solution['metrics']['total_cost']
        cost_savings_pct = (cost_savings / heuristic_cost) * 100
        
        print(f"\n{'='*70}")
        print(f"SAVINGS FROM OPTIMIZATION")
        print(f"{'='*70}")
        print(f"  Cost reduction: ${cost_savings:,.2f} ({cost_savings_pct:+.2f}%)")
        print(f"  Utilization improvement: {solution['metrics']['capacity_utilization'] - heuristic_utilization:+.2f}%")
        print(f"{'='*70}")
        
        return {
            'heuristic': {
                'cost': heuristic_cost,
                'utilization': heuristic_utilization,
                'cost_per_order': heuristic_cost / sum(demand)
            },
            'optimized': {
                'cost': solution['metrics']['total_cost'],
                'utilization': solution['metrics']['capacity_utilization'],
                'cost_per_order': solution['metrics']['labor_cost_per_order']
            },
            'savings': {
                'absolute': cost_savings,
                'percentage': cost_savings_pct
            }
        }


if __name__ == "__main__":
    # Test optimization with sample data
    print("Testing Staffing Optimizer...\n")
    
    # Run comparison
    comparison = compare_heuristic_vs_optimized('data/processed/features_test.csv')
