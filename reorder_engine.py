"""
Reorder Recommendation Engine Module for Smart Reorder Predictor
This module handles the calculation of reorder points, quantities, and recommendations.
"""

import pandas as pd
import numpy as np
from scipy import stats


class ReorderEngine:
    """
    Handles the calculation of optimal reorder points and quantities.
    """
    
    def __init__(self, forecaster, lead_time_days=7, service_level=0.95):
        """
        Initialize the ReorderEngine.
        
        Args:
            forecaster: Trained SalesForecaster object
            lead_time_days (int): Average supplier lead time in days
            service_level (float): Desired service level (0-1), default 95%
        """
        self.forecaster = forecaster
        self.lead_time_days = lead_time_days
        self.service_level = service_level
        self.z_score = stats.norm.ppf(service_level)  # Z-score for service level
        
    def calculate_safety_stock(self, demand_mean, demand_std):
        """
        Calculate safety stock using the formula: Safety Stock = Z * σ * √(Lead Time)
        """
        safety_stock = self.z_score * demand_std * np.sqrt(self.lead_time_days)
        return max(0, float(safety_stock))
    
    def calculate_reorder_point(self, demand_mean, safety_stock):
        """
        Calculate reorder point using the formula: ROP = (Demand * Lead Time) + Safety Stock
        """
        reorder_point = (demand_mean * self.lead_time_days) + safety_stock
        return max(0, float(reorder_point))
    
    def calculate_economic_order_quantity(self, annual_demand, ordering_cost=50, holding_cost=2):
        """
        Calculate Economic Order Quantity (EOQ) using the formula:
        EOQ = √(2 * D * S / H)
        """
        if holding_cost == 0:
            return 0
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return max(1, float(eoq))
    
    def generate_recommendations(self, store_id):
        """
        Generate reorder recommendations for a specific store.
        
        Args:
            store_id (int): Store ID
            
        Returns:
            dict: Dictionary with reorder recommendations
        """
        from datetime import datetime, timedelta
        
        # 1. Get Forecasted Demand (next 30 days average)
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 31)]
        forecasts = [self.forecaster.predict_single(store_id, d) for d in future_dates]
        
        demand_mean = np.mean(forecasts)
        demand_std = np.std(forecasts) if np.std(forecasts) > 0 else demand_mean * 0.15
        
        # 2. Calculate Metrics
        safety_stock = self.calculate_safety_stock(demand_mean, demand_std)
        reorder_point = self.calculate_reorder_point(demand_mean, safety_stock)
        
        # 3. Calculate EOQ
        annual_demand = demand_mean * 365
        eoq = self.calculate_economic_order_quantity(annual_demand)
        
        # 4. Mock Current Inventory (for prototype purposes)
        # In a real app, this would come from a database
        current_inventory = reorder_point * 0.8  # Assume we are at 80% of ROP
        
        reorder_needed = bool(current_inventory < reorder_point)
        recommended_quantity = max(eoq, reorder_point - current_inventory) if reorder_needed else 0
        
        # 5. Return clean dictionary with standard Python types
        return {
            'store_id': int(store_id),
            'avg_daily_forecast': round(float(demand_mean), 2),
            'safety_stock': round(float(safety_stock), 2),
            'reorder_point': round(float(reorder_point), 2),
            'eoq': round(float(eoq), 2),
            'current_inventory_estimate': round(float(current_inventory), 2),
            'reorder_needed': reorder_needed,
            'recommended_order_quantity': round(float(recommended_quantity), 2)
        }