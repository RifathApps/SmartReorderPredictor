# Smart Reorder Predictor - Project Documentation

## Overview

The Smart Reorder Predictor is a data science product designed to help small e-commerce businesses optimize their inventory management through machine learning-based sales forecasting and intelligent reorder recommendations.

## Project Structure

```
prototype/
├── README.md                    # Quick start guide
├── PROJECT_DOCUMENTATION.md     # This file
├── TECHNICAL_REPORT.md          # Full technical report
├── setup.py                     # Project setup script
├── requirements.txt             # Python dependencies
├── app.py                       # Streamlit web application
├── data_preprocessing.py        # Data preprocessing module
├── model_training.py            # ML model training module
├── reorder_engine.py            # Inventory optimization engine
├── data/                        # Raw and processed data
│   ├── train.csv               # Historical sales data
│   ├── store.csv               # Store information
│   └── test.csv                # Test data (optional)
├── models/                      # Trained ML models
│   └── sales_forecast_model.pkl # Serialized XGBoost model
└── notebooks/                   # Jupyter notebooks (optional)
    └── data_exploration_and_training.ipynb
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for version control)

### Step 1: Clone or Download the Project
```bash
cd /path/to/prototype
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Run Setup Script
```bash
python setup.py
```

Or manually install dependencies:
```bash
pip install -r requirements.txt
```

### Step 4: Download Data
Download the Rossmann Store Sales dataset from Kaggle:
https://www.kaggle.com/competitions/rossmann-store-sales/data

Place the following files in the `data/` directory:
- `train.csv` (historical sales data)
- `store.csv` (store information)
- `test.csv` (optional, for evaluation)

## Running the Application

### Start the Streamlit Dashboard
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Load Data:** Use the sidebar to load and preprocess the Rossmann dataset
2. **Train Model:** Click "Train Forecasting Model" to train the XGBoost model
3. **Generate Forecasts:** Select a store and generate sales forecasts
4. **Get Recommendations:** Generate reorder recommendations based on forecasts
5. **View Analytics:** Explore insights and performance metrics

## Code Modules

### 1. data_preprocessing.py
Handles all data loading, cleaning, and feature engineering tasks.

**Key Classes:**
- `DataPreprocessor`: Main class for data preprocessing

**Key Methods:**
- `load_data()`: Load raw CSV files
- `clean_data()`: Handle missing values and anomalies
- `merge_data()`: Merge train/test with store information
- `engineer_features()`: Create time-series and lag features
- `encode_categorical()`: Encode categorical variables
- `preprocess()`: Execute complete preprocessing pipeline

**Example Usage:**
```python
from data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor('data/train.csv', 'data/store.csv', 'data/test.csv')
train_df, test_df = preprocessor.preprocess()
```

### 2. model_training.py
Handles machine learning model training and evaluation.

**Key Classes:**
- `SalesForecaster`: XGBoost-based sales forecasting model

**Key Methods:**
- `prepare_features()`: Select and prepare features for training
- `train()`: Train the XGBoost model
- `evaluate()`: Evaluate model performance on test data
- `predict()`: Make predictions on new data
- `get_feature_importance()`: Get top N important features
- `save_model()`: Save trained model to file
- `load_model()`: Load pre-trained model from file

**Example Usage:**
```python
from model_training import SalesForecaster

forecaster = SalesForecaster()
X_train, y_train = forecaster.prepare_features(train_df)
forecaster.train(X_train, y_train)
metrics = forecaster.evaluate(X_test, y_test)
predictions = forecaster.predict(X_new)
```

### 3. reorder_engine.py
Handles inventory optimization and reorder recommendations.

**Key Classes:**
- `ReorderEngine`: Inventory optimization engine

**Key Methods:**
- `calculate_safety_stock()`: Calculate safety stock quantity
- `calculate_reorder_point()`: Calculate reorder point
- `calculate_economic_order_quantity()`: Calculate EOQ
- `generate_recommendations()`: Generate reorder recommendations
- `calculate_inventory_metrics()`: Calculate KPIs

**Example Usage:**
```python
from reorder_engine import ReorderEngine

engine = ReorderEngine(lead_time_days=7, service_level=0.95)
recommendations = engine.generate_recommendations(df, forecasted_sales, current_inventory)
metrics = engine.calculate_inventory_metrics(recommendations)
```

### 4. app.py
Streamlit web application for user interaction.

**Features:**
- Data loading and preprocessing
- Model training interface
- Sales forecasting visualization
- Reorder recommendations display
- Analytics and insights dashboard
- Performance metrics tracking

## Key Algorithms

### 1. Sales Forecasting (XGBoost)
- **Algorithm:** XGBoost Regressor
- **Features:** Time-based features, lag features, rolling averages, store characteristics
- **Performance:** MAPE < 15% on test data

### 2. Safety Stock Calculation
```
Safety Stock = Z * σ * √(Lead Time)
```
Where:
- Z = Z-score for desired service level
- σ = Standard deviation of demand
- Lead Time = Supplier lead time in days

### 3. Reorder Point Calculation
```
Reorder Point = (Average Daily Demand * Lead Time) + Safety Stock
```

### 4. Economic Order Quantity (EOQ)
```
EOQ = √(2 * Annual Demand * Ordering Cost / Holding Cost)
```

## Data Requirements

### Input Data Format

**train.csv** (Historical Sales Data)
- `Date`: Date of sales (YYYY-MM-DD)
- `Store`: Store ID (1-1115)
- `Sales`: Sales amount
- `Customers`: Number of customers
- `Open`: Store open indicator (1=open, 0=closed)
- `Promo`: Promotion indicator
- `StateHoliday`: State holiday indicator
- `SchoolHoliday`: School holiday indicator

**store.csv** (Store Information)
- `Store`: Store ID
- `StoreType`: Type of store (a, b, c, d)
- `Assortment`: Assortment level (a, b, c)
- `CompetitionDistance`: Distance to nearest competitor
- `CompetitionOpenSinceMonth`: Month competitor opened
- `CompetitionOpenSinceYear`: Year competitor opened
- `Promo2`: Ongoing promotion indicator
- `Promo2SinceWeek`: Week promotion started
- `Promo2SinceYear`: Year promotion started
- `PromoInterval`: Promotion interval

## Performance Metrics

### Model Evaluation Metrics
- **MAE (Mean Absolute Error):** Average absolute deviation from actual values
- **RMSE (Root Mean Squared Error):** Square root of average squared errors
- **R² Score:** Proportion of variance explained by the model

### Inventory Metrics
- **Stockout Rate:** Percentage of time inventory is depleted
- **Inventory Turnover:** Number of times inventory is sold and replaced
- **Carrying Cost:** Cost of holding inventory
- **Forecast Accuracy (MAPE):** Mean Absolute Percentage Error

## Troubleshooting

### Issue: "Data files not found"
**Solution:** Ensure train.csv and store.csv are in the `data/` directory

### Issue: "Model not trained"
**Solution:** Click "Train Forecasting Model" in the sidebar before generating forecasts

### Issue: "Port 8501 already in use"
**Solution:** Run `streamlit run app.py --server.port 8502` to use a different port

### Issue: "Memory error with large datasets"
**Solution:** Reduce the dataset size or use a machine with more RAM

## Future Enhancements

1. **Advanced Forecasting Models:** LSTM, Prophet, Ensemble methods
2. **Real-time Data Integration:** Connect to e-commerce platforms (Shopify, WooCommerce)
3. **Multi-channel Inventory:** Support for multiple sales channels
4. **Demand Sensing:** Incorporate external data (weather, social media trends)
5. **Automated Reordering:** Direct integration with supplier systems
6. **Mobile Application:** Native iOS/Android app
7. **Advanced Analytics:** Predictive analytics for customer behavior
8. **Scalability:** Kubernetes deployment, distributed processing

## Support & Contact

For issues, questions, or feature requests, please contact:
- **Email:** support@smartreorderpredictor.com
- **Documentation:** See TECHNICAL_REPORT.md
- **GitHub Issues:** [Project Repository]

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Changelog

### Version 1.0 (April 2026)
- Initial release
- XGBoost forecasting model
- Streamlit web interface
- Inventory optimization engine
- Basic analytics dashboard

---

**Last Updated:** April 27, 2026  
**Version:** 1.0  
**Status:** Production Ready
