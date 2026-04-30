"""
XAI (Explainable AI) Module for Smart Reorder Predictor
This module provides interpretability and explainability for the ML model using SHAP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')


class XAIExplainer:
    """
    Provides explainability for the sales forecasting model using SHAP (SHapley Additive exPlanations).
    """
    
    def __init__(self, model, X_train=None, feature_names=None):
        """
        Initialize the XAI Explainer.
        
        Args:
            model: Trained XGBoost model
            X_train (pd.DataFrame): Training data used for SHAP background
            feature_names (list): List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        
    def initialize_explainer(self):
        """
        Initialize SHAP explainer.
        """
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
    
    def explain_prediction(self, store_id):
        """
        Explain a single prediction for a store.
        
        Args:
            store_id (int): Store ID to explain
            
        Returns:
            dict: Dictionary containing explanation data
        """
        if self.explainer is None:
            self.initialize_explainer()
            
        # Create a sample instance for the store (simplified for prototype)
        from datetime import datetime
        ts = pd.Timestamp(datetime.now())
        
        # This must match the feature preparation in model_training.py
        instance_data = {
            'Store': [store_id],
            'Promo': [1], # Assume promo for explanation variety
            'DayOfWeek': [ts.dayofweek],
            'Month': [ts.month],
            'Year': [ts.year],
            'Day': [ts.day],
            'Week': [ts.isocalendar()[1]],
            'Quarter': [ts.quarter],
            'Open': [1]
        }
        
        # Fill in other features if they exist in feature_names
        if self.feature_names:
            for col in self.feature_names:
                if col not in instance_data:
                    instance_data[col] = [0]
            
            # Ensure correct order
            X_instance = pd.DataFrame(instance_data)[self.feature_names]
        else:
            X_instance = pd.DataFrame(instance_data)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_instance)
        base_value = self.explainer.expected_value
        
        # Create a simple dictionary of feature importance for this prediction
        explanation = {}
        cols = X_instance.columns
        vals = shap_values[0]
        
        for col, val in zip(cols, vals):
            explanation[col] = float(val)
            
        # Sort by absolute value
        explanation = dict(sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True))
        
        return {
            'base_value': float(base_value),
            'prediction': float(self.model.predict(X_instance)[0]),
            'contributions': explanation,
            'feature_names': list(cols),
            'feature_values': X_instance.iloc[0].to_dict()
        }