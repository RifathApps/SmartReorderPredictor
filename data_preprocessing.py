"""
Data Preprocessing Module for Smart Reorder Predictor
Handles data cleaning, feature engineering, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Handles the cleaning and feature engineering for the Rossmann Store Sales dataset.
    """
    
    def __init__(self):
        """
        Initialize the preprocessor.
        """
        self.label_encoders = {}
        
    def clean_and_prepare_data(self, df):
        """
        Performs full cleaning and feature engineering on the input dataframe.
        
        Args:
            df (pd.DataFrame): The raw merged dataframe (train + store)
            
        Returns:
            pd.DataFrame: Cleaned and featured dataframe
        """
        data = df.copy()
        
        # 1. Basic Cleaning
        # Handle missing values in CompetitionDistance
        if 'CompetitionDistance' in data.columns:
            data['CompetitionDistance'].fillna(data['CompetitionDistance'].median(), inplace=True)
            
        # Handle other missing values
        cols_to_fix = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
                       'Promo2SinceWeek', 'Promo2SinceYear']
        for col in cols_to_fix:
            if col in data.columns:
                data[col].fillna(0, inplace=True)
        
        if 'PromoInterval' in data.columns:
            data['PromoInterval'].fillna('0', inplace=True)
        
        # 2. Feature Engineering
        # Extract date features
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data['Year'] = data['Date'].dt.year
            data['Month'] = data['Date'].dt.month
            data['Day'] = data['Date'].dt.day
            data['DayOfWeek'] = data['Date'].dt.dayofweek
            data['Week'] = data['Date'].dt.isocalendar().week.astype(int)
            data['Quarter'] = data['Date'].dt.quarter
            
        # 3. Categorical Encoding
        categorical_cols = ['StoreType', 'Assortment', 'PromoInterval', 'StateHoliday']
        
        for col in categorical_cols:
            if col in data.columns:
                try:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
                    self.label_encoders[col] = le
                except Exception as e:
                    print(f"Warning: Could not encode column {col}: {str(e)}")
            
        # 4. Lag Features
        # Sort by store and date for correct lag calculation
        data = data.sort_values(['Store', 'Date'])
        
        # Create lag features
        data['Sales_Lag1'] = data.groupby('Store')['Sales'].shift(1)
        data['Sales_Lag7'] = data.groupby('Store')['Sales'].shift(7)
        
        # Rolling mean features
        data['Sales_RollingMean7'] = data.groupby('Store')['Sales'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        # Fill NaN values created by lag features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col].fillna(data[col].mean(), inplace=True)
        
        # 5. Final Cleanup
        # Remove rows where sales are 0 (store closed) for training
        if 'Sales' in data.columns:
            data = data[data['Sales'] > 0]
        
        # Drop Date column as it's not used in the model directly
        if 'Date' in data.columns:
            data = data.drop(columns=['Date'])
            
        # Ensure all columns are numeric
        data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return data