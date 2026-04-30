"""
Scenario Simulator Module for Smart Reorder Predictor
This module enables "What-If" analysis by allowing users to modify input features
and observe the impact on sales forecasts and reorder recommendations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ScenarioSimulator:
    """
    Enables "What-If" scenario analysis by simulating different business conditions
    and their impact on sales forecasts and inventory recommendations.
    """
    
    def __init__(self, model, feature_names, historical_data):
        """
        Initialize the Scenario Simulator.
        
        Args:
            model: Trained XGBoost model
            feature_names (list): List of feature names used by the model
            historical_data (pd.DataFrame): Historical data for context
        """
        self.model = model
        self.feature_names = feature_names
        self.historical_data = historical_data
        
    def create_baseline_scenario(self, store_id, days_ahead=30):
        """
        Create a baseline scenario using recent historical patterns.
        """
        # Get recent data for the store
        store_data = self.historical_data[
            self.historical_data['Store'] == store_id
        ].tail(30).copy()
        
        if len(store_data) == 0:
            raise ValueError(f"No data found for Store {store_id}")
        
        # Create future dates
        last_date = pd.to_datetime(store_data['Date'].max())
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
        
        # Create baseline scenario
        baseline = pd.DataFrame({
            'Date': future_dates,
            'Store': store_id,
            'Promo': 0,
            'StateHoliday': '0',
            'SchoolHoliday': 0,
            'DayOfWeek': [d.dayofweek for d in future_dates],
            'Month': [d.month for d in future_dates],
            'Day': [d.day for d in future_dates],
            'Year': [d.year for d in future_dates],
            'Week': [d.isocalendar()[1] for d in future_dates],
            'Quarter': [d.quarter for d in future_dates],
        })
        
        # Add store-level features from historical data
        store_info = self.historical_data[
            self.historical_data['Store'] == store_id
        ].iloc[0]
        
        # Only add features that the model expects
        for col in self.feature_names:
            if col in store_info.index:
                baseline[col] = store_info[col]
            elif col not in baseline.columns:
                baseline[col] = 0
        
        # Add lag features based on recent average
        baseline['Sales_Lag1'] = store_data['Sales'].iloc[-1]
        baseline['Sales_Lag7'] = store_data['Sales'].tail(7).mean()
        baseline['Sales_Lag30'] = store_data['Sales'].mean()
        baseline['Sales_RollingMean7'] = store_data['Sales'].tail(7).mean()
        baseline['Sales_RollingMean30'] = store_data['Sales'].mean()
        baseline['Open'] = 1
        
        return baseline
    
    def apply_promotion_scenario(self, baseline_scenario, promo_dates):
        scenario = baseline_scenario.copy()
        scenario['Promo'] = scenario['Date'].isin(promo_dates).astype(int)
        return scenario
    
    def apply_holiday_scenario(self, baseline_scenario, holiday_dates, holiday_type='StateHoliday'):
        scenario = baseline_scenario.copy()
        if holiday_type == 'StateHoliday':
            scenario['StateHoliday'] = scenario['Date'].isin(holiday_dates).astype(int)
        elif holiday_type == 'SchoolHoliday':
            scenario['SchoolHoliday'] = scenario['Date'].isin(holiday_dates).astype(int)
        return scenario
    
    def apply_competition_scenario(self, baseline_scenario, competition_distance):
        scenario = baseline_scenario.copy()
        scenario['CompetitionDistance'] = competition_distance
        return scenario
    
    def forecast_scenario(self, scenario_df):
        """
        Generate sales forecasts for a given scenario.
        """
        # Prepare features for prediction
        X_scenario = scenario_df.copy()
        
        # Handle categorical data - convert to codes if they are objects
        for col in X_scenario.columns:
            if X_scenario[col].dtype == 'object':
                X_scenario[col] = X_scenario[col].astype('category').cat.codes
        
        # Ensure all required features are present and in correct order
        for col in self.feature_names:
            if col not in X_scenario.columns:
                X_scenario[col] = 0
        
        X_input = X_scenario[self.feature_names]
        
        # Make predictions
        predictions = self.model.predict(X_input)
        
        # Create results dataframe
        results = pd.DataFrame({
            'Date': scenario_df['Date'].values,
            'Forecasted_Sales': predictions,
            'Promo': scenario_df['Promo'].values if 'Promo' in scenario_df.columns else 0,
        })
        
        # Calculate statistics
        stats = {
            'Average_Sales': predictions.mean(),
            'Max_Sales': predictions.max(),
            'Min_Sales': predictions.min(),
            'Std_Dev': predictions.std(),
            'Total_Sales': predictions.sum()
        }
        
        return {
            'forecasts': results,
            'statistics': stats,
            'scenario': scenario_df
        }
    
    def compare_scenarios(self, baseline_result, modified_result):
        baseline_stats = baseline_result['statistics']
        modified_stats = modified_result['statistics']
        
        comparison = {
            'Baseline_Avg_Sales': baseline_stats['Average_Sales'],
            'Modified_Avg_Sales': modified_stats['Average_Sales'],
            'Sales_Change': modified_stats['Average_Sales'] - baseline_stats['Average_Sales'],
            'Sales_Change_Percent': (
                (modified_stats['Average_Sales'] - baseline_stats['Average_Sales']) / 
                baseline_stats['Average_Sales'] * 100
            ) if baseline_stats['Average_Sales'] > 0 else 0,
            'Total_Sales_Impact': modified_stats['Total_Sales'] - baseline_stats['Total_Sales'],
            'Volatility_Change': modified_stats['Std_Dev'] - baseline_stats['Std_Dev'],
        }
        
        return comparison
    
    def generate_scenario_report(self, scenario_name, baseline_result, modified_result=None):
        report = f"\n{'='*60}\n"
        report += f"SCENARIO ANALYSIS REPORT: {scenario_name}\n"
        report += f"{'='*60}\n\n"
        
        baseline_stats = baseline_result['statistics']
        report += "BASELINE SCENARIO:\n"
        report += f"  Average Daily Sales: {baseline_stats['Average_Sales']:.2f} units\n"
        report += f"  Total Forecasted Sales: {baseline_stats['Total_Sales']:.2f} units\n"
        report += f"  Sales Volatility (Std Dev): {baseline_stats['Std_Dev']:.2f}\n"
        
        if modified_result is not None:
            report += "\nMODIFIED SCENARIO:\n"
            modified_stats = modified_result['statistics']
            report += f"  Average Daily Sales: {modified_stats['Average_Sales']:.2f} units\n"
            report += f"  Total Forecasted Sales: {modified_stats['Total_Sales']:.2f} units\n"
            
            comparison = self.compare_scenarios(baseline_result, modified_result)
            report += "\nIMPACT ANALYSIS:\n"
            report += f"  Sales Change: {comparison['Sales_Change']:.2f} units ({comparison['Sales_Change_Percent']:.2f}%)\n"
            report += f"  Total Impact: {comparison['Total_Sales_Impact']:.2f} units\n"
        
        report += f"\n{'='*60}\n"
        return report