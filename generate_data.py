"""
Synthetic Data Generator for Cloud Kitchen Operations
Generates realistic order data based on real patterns from Vel's Kitchen Indian
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class CloudKitchenDataGenerator:
    """Generate synthetic cloud kitchen operational data"""
    
    def __init__(self, start_date='2025-08-01', months=6, zip_code='60642'):
        self.start_date = pd.to_datetime(start_date)
        self.months = months
        self.zip_code = zip_code
        self.end_date = self.start_date + pd.DateOffset(months=months)
        
        # Kitchen configuration
        self.kitchen_id = 39555939
        self.kitchen_name = "Vel's Kitchen Indian"
        self.kitchen_lat = 41.9007
        self.kitchen_long = -87.6741
        
        # Menu items based on real data
        self.menu_items = self._create_menu()
        
        # Demographic data for zip 60642
        self.demographics = {
            'zip_code': '60642',
            'population': 21124,
            'median_income': 141179,
            'median_age': 32.5,
            'poverty_rate': 5.2,
            'white_pct': 64.5,
            'hispanic_pct': 14.4,
            'black_pct': 8.0,
            'population_density': 'high',
            'urbanicity': 'urban'
        }
        
    def _create_menu(self):
        """Create menu items based on real orders"""
        menu = [
            {'item_id': 1, 'name': 'Idli (3 pcs)', 'category': 'Tiffin', 'price': 6.99, 'cost': 2.50, 'prep_time': 8, 'is_veg': True, 'popularity': 0.85},
            {'item_id': 2, 'name': 'Masala Dosa 2pc', 'category': 'Tiffin', 'price': 10.99, 'cost': 3.80, 'prep_time': 12, 'is_veg': True, 'popularity': 0.90},
            {'item_id': 3, 'name': 'Pongal', 'category': 'Tiffin', 'price': 8.99, 'cost': 3.00, 'prep_time': 10, 'is_veg': True, 'popularity': 0.70},
            {'item_id': 4, 'name': 'Punugulu', 'category': 'Snacks', 'price': 5.99, 'cost': 2.00, 'prep_time': 6, 'is_veg': True, 'popularity': 0.65},
            {'item_id': 5, 'name': 'Chettinaad Chicken Rice Bowl', 'category': 'Rice Bowls', 'price': 13.99, 'cost': 5.50, 'prep_time': 15, 'is_veg': False, 'popularity': 0.80},
            {'item_id': 6, 'name': 'Biryani - Chicken', 'category': 'Rice Bowls', 'price': 14.99, 'cost': 6.00, 'prep_time': 18, 'is_veg': False, 'popularity': 0.95},
            {'item_id': 7, 'name': 'Biryani - Veg', 'category': 'Rice Bowls', 'price': 12.99, 'cost': 4.50, 'prep_time': 16, 'is_veg': True, 'popularity': 0.75},
            {'item_id': 8, 'name': 'Madras Filter Coffee', 'category': 'Beverages', 'price': 4.49, 'cost': 1.20, 'prep_time': 3, 'is_veg': True, 'popularity': 0.60},
            {'item_id': 9, 'name': 'Mango Lassi', 'category': 'Beverages', 'price': 4.99, 'cost': 1.50, 'prep_time': 3, 'is_veg': True, 'popularity': 0.55},
            {'item_id': 10, 'name': 'Sambar Rice', 'category': 'Rice Bowls', 'price': 9.99, 'cost': 3.50, 'prep_time': 12, 'is_veg': True, 'popularity': 0.68},
        ]
        return pd.DataFrame(menu)
    
    def _get_hourly_demand_multiplier(self, hour, day_of_week):
        """Get demand multiplier based on hour and day"""
        # Lunch peak: 11am-2pm, Dinner peak: 6pm-9pm
        if 11 <= hour <= 13:  # Lunch
            base = 2.5
        elif 18 <= hour <= 20:  # Dinner
            base = 3.0
        elif 14 <= hour <= 17:  # Afternoon
            base = 1.2
        elif 21 <= hour <= 22:  # Late evening
            base = 1.5
        elif 10 <= hour < 11:  # Late breakfast
            base = 1.0
        else:
            base = 0.3
        
        # Weekend boost
        if day_of_week >= 5:  # Saturday, Sunday
            base *= 1.3
        
        return base
    
    def _get_weather_condition(self, date):
        """Simulate weather conditions"""
        month = date.month
        rand = random.random()
        
        # Seasonal patterns
        if month in [12, 1, 2]:  # Winter
            if rand < 0.4:
                return 'Snow', -2.0
            elif rand < 0.7:
                return 'Cloudy', 2.0
            else:
                return 'Clear', 0.0
        elif month in [6, 7, 8]:  # Summer
            if rand < 0.2:
                return 'Rain', 15.0
            elif rand < 0.4:
                return 'Cloudy', 25.0
            else:
                return 'Clear', 28.0
        else:  # Spring/Fall
            if rand < 0.3:
                return 'Rain', 12.0
            elif rand < 0.6:
                return 'Cloudy', 15.0
            else:
                return 'Clear', 18.0
    
    def _is_holiday(self, date):
        """Check if date is a holiday"""
        holidays = [
            (1, 1),   # New Year
            (7, 4),   # Independence Day
            (11, 28), # Thanksgiving (approximate)
            (12, 25), # Christmas
        ]
        return (date.month, date.day) in holidays
    
    def generate_orders(self):
        """Generate synthetic order data"""
        orders = []
        order_id = 1
        
        current_date = self.start_date
        
        while current_date < self.end_date:
            day_of_week = current_date.dayofweek
            is_holiday = self._is_holiday(current_date)
            weather, temp = self._get_weather_condition(current_date)
            
            # Operating hours: 10 AM to 11 PM
            for hour in range(10, 23):
                # Get base demand for this hour
                base_demand = self._get_hourly_demand_multiplier(hour, day_of_week)
                
                # Holiday boost
                if is_holiday:
                    base_demand *= 1.4
                
                # Weather impact
                if weather in ['Rain', 'Snow']:
                    base_demand *= 1.25  # Bad weather increases delivery orders
                
                # Generate orders for this hour (Poisson distribution)
                num_orders = np.random.poisson(base_demand * 1.5)
                
                for _ in range(num_orders):
                    # Random minute within the hour
                    minute = random.randint(0, 59)
                    second = random.randint(0, 59)
                    order_time = current_date.replace(hour=hour, minute=minute, second=second)
                    
                    # Select menu items (1-3 items per order)
                    num_items = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
                    selected_items = self.menu_items.sample(n=num_items, weights='popularity')
                    
                    order_subtotal = selected_items['price'].sum()
                    
                    # Preparation time (max of all items + buffer)
                    prep_time_min = int(selected_items['prep_time'].max()) + random.randint(2, 8)
                    
                    # Delivery time (15-45 minutes after prep)
                    delivery_time_min = random.randint(15, 45)
                    
                    # Calculate timestamps
                    confirmed_ready_time = order_time + timedelta(minutes=prep_time_min)
                    dasher_arrival_time = confirmed_ready_time + timedelta(minutes=random.randint(-5, 10))
                    pickup_time = dasher_arrival_time + timedelta(minutes=random.uniform(0.5, 5))
                    delivery_time = pickup_time + timedelta(minutes=delivery_time_min)
                    
                    # Avoidable wait time
                    avoidable_wait = max(0, (pickup_time - confirmed_ready_time).total_seconds() / 60)
                    
                    # Total delivery time
                    total_delivery_time = (delivery_time - order_time).total_seconds() / 60
                    
                    # Order status (95% completed, 5% cancelled)
                    status = 'Completed' if random.random() < 0.95 else 'Cancelled'
                    
                    # Net payout (70-90% of subtotal)
                    net_payout = order_subtotal * random.uniform(0.70, 0.90)
                    
                    # Rating (3-5 stars, weighted toward higher)
                    rating = np.random.choice([3, 4, 5], p=[0.1, 0.3, 0.6])
                    
                    # Discount applied (20% of orders)
                    discount_applied = random.random() < 0.2
                    
                    order = {
                        'order_id': order_id,
                        'order_datetime': order_time,
                        'order_date': order_time.date(),
                        'order_time': order_time.time(),
                        'order_hour': hour,
                        'order_dayofweek': day_of_week,
                        'day_name': order_time.strftime('%A'),
                        'is_weekend': day_of_week >= 5,
                        'is_holiday': is_holiday,
                        'kitchen_id': self.kitchen_id,
                        'kitchen_name': self.kitchen_name,
                        'zip_code': self.zip_code,
                        'num_items': num_items,
                        'order_subtotal': round(order_subtotal, 2),
                        'net_payout': round(net_payout, 2),
                        'discount_applied': discount_applied,
                        'prep_time_min': prep_time_min,
                        'delivery_time_min': delivery_time_min,
                        'total_delivery_time_min': round(total_delivery_time, 2),
                        'avoidable_wait_min': round(avoidable_wait, 2),
                        'confirmed_ready_time': confirmed_ready_time,
                        'dasher_arrival_time': dasher_arrival_time,
                        'pickup_time': pickup_time,
                        'delivery_time': delivery_time,
                        'order_status': status,
                        'rating': rating if status == 'Completed' else None,
                        'weather_condition': weather,
                        'temperature_c': temp,
                        'currency': 'USD'
                    }
                    
                    orders.append(order)
                    order_id += 1
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(orders)
    
    def save_data(self, output_dir='data/raw'):
        """Generate and save all datasets"""
        print("Generating synthetic order data...")
        orders_df = self.generate_orders()
        
        print(f"Generated {len(orders_df)} orders from {self.start_date.date()} to {self.end_date.date()}")
        print(f"Date range: {orders_df['order_date'].min()} to {orders_df['order_date'].max()}")
        print(f"Completed orders: {(orders_df['order_status'] == 'Completed').sum()}")
        print(f"Cancelled orders: {(orders_df['order_status'] == 'Cancelled').sum()}")
        
        # Save orders
        orders_df.to_csv(f'{output_dir}/orders_synthetic.csv', index=False)
        print(f"Saved: {output_dir}/orders_synthetic.csv")
        
        # Save menu items
        self.menu_items.to_csv(f'{output_dir}/menu_items.csv', index=False)
        print(f"Saved: {output_dir}/menu_items.csv")
        
        # Save demographics
        demo_df = pd.DataFrame([self.demographics])
        demo_df.to_csv('data/external/demographics_60642.csv', index=False)
        print(f"Saved: data/external/demographics_60642.csv")
        
        # Save kitchen info
        kitchen_info = {
            'kitchen_id': self.kitchen_id,
            'brand_name': self.kitchen_name,
            'location_lat': self.kitchen_lat,
            'location_long': self.kitchen_long,
            'zip_code': self.zip_code,
            'capacity_orders_per_hr': 8,
            'num_chefs': 3,
            'rating_avg': 4.5,
            'status': 'Open'
        }
        kitchen_df = pd.DataFrame([kitchen_info])
        kitchen_df.to_csv(f'{output_dir}/kitchen_info.csv', index=False)
        print(f"Saved: {output_dir}/kitchen_info.csv")
        
        return orders_df

if __name__ == "__main__":
    generator = CloudKitchenDataGenerator(start_date='2025-08-01', months=6)
    orders_df = generator.save_data()
    
    print("\n" + "="*50)
    print("Data Generation Complete!")
    print("="*50)
