"""
Model Training Module for Smart Reorder Predictor
This module handles training and evaluation of the XGBoost sales forecasting model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')


class SalesForecaster:
    """
    Handles training and evaluation of the XGBoost sales forecasting model.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the SalesForecaster.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.model = None
        self.random_state = random_state
        self.feature_names = None
        self.train_features = None
        self.train_target = None
        
    def prepare_features(self, df):
        """
        Prepare features for model training by selecting relevant columns.
        
        Args:
            df (pd.DataFrame): Input dataframe with all features
            
        Returns:
            tuple: (features_df, target_series)
        """
        # Define features to use for the model
        feature_cols = [
            'Store', 'Promo', 'DayOfWeek', 'Month', 'Year', 'Day', 'Week', 'Quarter',
            'Open', 'StoreType', 'Assortment', 'CompetitionDistance',
            'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
            'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear',
            'Sales_Lag1', 'Sales_Lag7', 'Sales_RollingMean7'
        ]
        
        # Filter to only available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols].copy()
        y = df['Sales'].copy()
        
        # Handle any remaining NaN values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        
        self.feature_names = available_cols
        return X, y
    
    def train(self, df):
        """
        Train the XGBoost model using a single dataframe.
        
        Args:
            df (pd.DataFrame): Processed dataframe containing features and target
        """
        # Prepare features and target
        X, y = self.prepare_features(df)
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Define XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'n_estimators': 100,
            'verbosity': 0
        }
        
        self.model = xgb.XGBRegressor(**params)
        
        # Train the model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.train_features = X_train
        self.train_target = y_train
        
    def predict_single(self, store_id, date):
        """
        Predict sales for a single store on a specific date.
        
        Args:
            store_id (int): Store ID
            date (datetime): Date to forecast
            
        Returns:
            float: Predicted sales value
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        # Convert to pandas Timestamp to ensure consistent attribute access
        ts = pd.Timestamp(date)
            
        # Create a single-row dataframe for prediction
        input_data = pd.DataFrame({
            'Store': [store_id],
            'Promo': [0],
            'DayOfWeek': [ts.dayofweek],
            'Month': [ts.month],
            'Year': [ts.year],
            'Day': [ts.day],
            'Week': [ts.isocalendar()[1]],
            'Quarter': [ts.quarter],
            'Open': [1]
        })
        
        # Add missing features with default values
        for col in self.feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
                
        # Ensure correct column order
        input_data = input_data[self.feature_names]
        
        prediction = self.model.predict(input_data)[0]
        return max(0, float(prediction))
    
    def get_feature_importance(self, top_n=10):
        """
        Get the top N most important features.
        """
        if self.model is None:
            return pd.DataFrame(columns=['Feature', 'Importance'])
            
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df.head(top_n)