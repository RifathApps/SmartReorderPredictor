"""
Smart Reorder Predictor - Streamlit Application
A data science product for enhanced inventory management in small e-commerce businesses.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from data_preprocessing import DataPreprocessor
from model_training import SalesForecaster
from reorder_engine import ReorderEngine
from xai_module import XAIExplainer
from scenario_simulator import ScenarioSimulator
import os
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit
st.set_page_config(
    page_title="Smart Reorder Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stDataFrame {
        border: 1px solid #e6e9ef;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Application Title and Description
st.title("📊 Smart Reorder Predictor")
st.markdown("**Intelligent Inventory Management for Small E-commerce Businesses**")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Initialize session state
if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = None
if 'reorder_engine' not in st.session_state:
    st.session_state.reorder_engine = None
if 'scenario_simulator' not in st.session_state:
    st.session_state.scenario_simulator = None

# Automatic Data Loading Section
st.sidebar.subheader("📁 Data Source")
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
STORE_FILE = os.path.join(DATA_DIR, "store.csv")

def load_local_data():
    if os.path.exists(TRAIN_FILE):
        try:
            # Load train data
            train = pd.read_csv(TRAIN_FILE)
            train['Date'] = pd.to_datetime(train['Date'])
            
            # Load store data if exists and merge
            if os.path.exists(STORE_FILE):
                store = pd.read_csv(STORE_FILE)
                train = pd.merge(train, store, on='Store', how='left')
                st.sidebar.success("✅ Data & Store info loaded from /data")
            else:
                st.sidebar.success("✅ Train data loaded from /data")
            
            return train
        except Exception as e:
            st.sidebar.error(f"❌ Error loading local data: {str(e)}")
            return None
    else:
        st.sidebar.warning(f"⚠️ Data not found in {TRAIN_FILE}")
        return None

# Load data automatically if not already loaded
if st.session_state.train_df is None:
    st.session_state.train_df = load_local_data()

if st.session_state.train_df is not None:
    st.sidebar.write(f"**Dataset Status:** Active")
    st.sidebar.write(f"**Records:** {len(st.session_state.train_df):,}")
    st.sidebar.write(f"**Stores:** {st.session_state.train_df['Store'].nunique()}")
else:
    st.sidebar.error("Please ensure 'train.csv' is in the 'data/' folder.")

# Model Training Section
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Training")

if st.session_state.train_df is not None:
    if st.sidebar.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training model... This may take a moment."):
            try:
                # Preprocess data
                preprocessor = DataPreprocessor()
                processed_df = preprocessor.clean_and_prepare_data(st.session_state.train_df)
                
                # Train model
                st.session_state.forecaster = SalesForecaster()
                st.session_state.forecaster.train(processed_df)
                
                # Initialize reorder engine
                st.session_state.reorder_engine = ReorderEngine(st.session_state.forecaster)
                
                # Initialize scenario simulator
                st.session_state.scenario_simulator = ScenarioSimulator(
                    st.session_state.forecaster.model,
                    st.session_state.forecaster.feature_names,
                    st.session_state.train_df
                )
                
                st.sidebar.success("✅ Model trained successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Error training model: {str(e)}")
else:
    st.sidebar.info("⏳ Waiting for data...")

# Main Content Area
if st.session_state.train_df is not None:
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Dashboard",
        "🔮 Forecasting",
        "📦 Reorder Recommendations",
        "📈 Analytics",
        "🔍 XAI Insights",
        "📋 Use Cases",
        "📅 Custom Date Range",
        "🎯 What-If Scenarios"
    ])
    
    # Tab 1: Dashboard
    with tab1:
        st.subheader("📊 Dashboard Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(st.session_state.train_df):,}")
        with col2:
            st.metric("Number of Stores", st.session_state.train_df['Store'].nunique())
        with col3:
            st.metric("Date Range", f"{st.session_state.train_df['Date'].min().date()} to {st.session_state.train_df['Date'].max().date()}")
        with col4:
            st.metric("Total Sales", f"€{st.session_state.train_df['Sales'].sum():,.0f}")
        
        st.markdown("---")
        
        # Sales distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Sales Distribution**")
            fig, ax = plt.subplots(figsize=(10, 6))
            st.session_state.train_df['Sales'].hist(bins=50, ax=ax, color='steelblue', edgecolor='black')
            ax.set_title('Distribution of Daily Sales')
            ax.set_xlabel('Sales (€)')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
        with col2:
            st.write("**Top 10 Stores by Total Sales**")
            top_stores = st.session_state.train_df.groupby('Store')['Sales'].sum().nlargest(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            top_stores.plot(kind='barh', ax=ax, color='coral')
            ax.set_title('Top 10 Stores by Total Sales')
            ax.set_xlabel('Total Sales (€)')
            st.pyplot(fig)
    
    # Tab 2: Forecasting
    with tab2:
        st.subheader("🔮 Sales Forecasting")
        
        if st.session_state.forecaster is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_store = st.selectbox(
                    "Select Store",
                    options=sorted(st.session_state.train_df['Store'].unique()),
                    key="forecast_store"
                )
            
            with col2:
                forecast_days = st.slider(
                    "Days to Forecast",
                    min_value=7,
                    max_value=90,
                    value=30,
                    step=7,
                    key="forecast_days_slider"
                )
            
            if st.button("📊 Generate Forecast", use_container_width=True):
                try:
                    # Get recent data for the store
                    store_data = st.session_state.train_df[
                        st.session_state.train_df['Store'] == selected_store
                    ].tail(30).copy()
                    
                    if len(store_data) > 0:
                        # Create future dates
                        last_date = pd.to_datetime(store_data['Date'].max())
                        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
                        
                        # Make forecasts
                        forecasts = []
                        for date in future_dates:
                            forecast_value = st.session_state.forecaster.predict_single(selected_store, date)
                            forecasts.append(forecast_value)
                        
                        # Display results
                        st.success("✅ Forecast generated successfully!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Average Forecast", f"€{np.mean(forecasts):,.0f}")
                        with col2:
                            st.metric("Max Forecast", f"€{np.max(forecasts):,.0f}")
                        with col3:
                            st.metric("Min Forecast", f"€{np.min(forecasts):,.0f}")
                        with col4:
                            st.metric("Std Dev", f"€{np.std(forecasts):,.0f}")
                        
                        # Plot forecast
                        st.markdown("---")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(future_dates, forecasts, marker='o', linewidth=2, color='green', label='Forecast')
                        ax.fill_between(future_dates, forecasts, alpha=0.3, color='green')
                        ax.set_title(f'Sales Forecast - Store {selected_store}')
                        ax.set_ylabel('Forecasted Sales (€)')
                        ax.set_xlabel('Date')
                        ax.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display forecast table
                        forecast_df = pd.DataFrame({
                            'Date': future_dates,
                            'Forecasted_Sales': forecasts
                        })
                        st.dataframe(forecast_df, use_container_width=True)
                    else:
                        st.warning("⚠️ No data found for the selected store.")
                except Exception as e:
                    st.error(f"❌ Error generating forecast: {str(e)}")
        else:
            st.info("⏳ Please train the model first.")
    
    # Tab 3: Reorder Recommendations
    with tab3:
        st.subheader("📦 Reorder Recommendations")
        
        if st.session_state.reorder_engine is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_stores = st.multiselect(
                    "Select Stores for Analysis",
                    options=sorted(st.session_state.train_df['Store'].unique()),
                    default=sorted(st.session_state.train_df['Store'].unique())[:5],
                    key="reorder_stores_multi"
                )
            
            with col2:
                st.write("") # Spacer
                st.write("") # Spacer
                gen_button = st.button("🎯 Generate Recommendations", use_container_width=True)
            
            if gen_button:
                try:
                    all_reorders = []
                    with st.spinner("Calculating optimal reorder points..."):
                        for store_id in selected_stores:
                            rec = st.session_state.reorder_engine.generate_recommendations(store_id)
                            all_reorders.append(rec)
                    
                    reorder_df = pd.DataFrame(all_reorders)
                    
                    # Rename columns for better display
                    display_df = reorder_df.rename(columns={
                        'store_id': 'Store ID',
                        'avg_daily_forecast': 'Avg Daily Forecast (€)',
                        'safety_stock': 'Safety Stock (€)',
                        'reorder_point': 'Reorder Point (€)',
                        'eoq': 'EOQ (€)',
                        'current_inventory_estimate': 'Est. Inventory (€)',
                        'reorder_needed': 'Reorder Needed',
                        'recommended_order_quantity': 'Order Qty (€)'
                    })
                    
                    # Style the dataframe
                    def highlight_reorder(val):
                        color = 'background-color: #ffcccc' if val == True else ''
                        return color

                    st.success(f"✅ Recommendations generated for {len(selected_stores)} stores!")
                    
                    # Display metrics for the group
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Stores Needing Reorder", len(reorder_df[reorder_df['reorder_needed'] == True]))
                    with m2:
                        st.metric("Total Order Value", f"€{reorder_df['recommended_order_quantity'].sum():,.2f}")
                    with m3:
                        st.metric("Avg Reorder Point", f"€{reorder_df['reorder_point'].mean():,.2f}")

                    st.markdown("---")
                    st.write("**Inventory Optimization Table**")
                    st.dataframe(
                        display_df.style.applymap(highlight_reorder, subset=['Reorder Needed']),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.info("💡 **Tip:** Rows highlighted in red indicate stores where estimated inventory is below the calculated Reorder Point.")
                    
                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {str(e)}")
        else:
            st.info("⏳ Please train the model first.")
    
    # Tab 4: Analytics
    with tab4:
        st.subheader("📈 Advanced Data Science Analytics")
        
        # Row 1: Sales Distribution and Trends
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Sales Distribution by Store Type**")
            if 'StoreType' in st.session_state.train_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=st.session_state.train_df, x='StoreType', y='Sales', ax=ax, palette='viridis')
                ax.set_title('Sales Spread across Store Types')
                st.pyplot(fig)
        
        with col2:
            st.write("**Aggregate Daily Sales Trend**")
            daily_sales = st.session_state.train_df.groupby('Date')['Sales'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=daily_sales, x='Date', y='Sales', ax=ax, color='teal')
            ax.set_title('Total Daily Sales Over Time')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        st.markdown("---")
        
        # Row 2: Promotion Impact and Seasonality
        col3, col4 = st.columns(2)
        with col3:
            st.write("**Promotion Impact Analysis**")
            if 'Promo' in st.session_state.train_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.violinplot(data=st.session_state.train_df, x='Promo', y='Sales', ax=ax, palette='Set2')
                ax.set_title('Sales Distribution: Promo vs No Promo')
                ax.set_xticklabels(['No Promo', 'Promo'])
                st.pyplot(fig)
        
        with col4:
            st.write("**Day of Week Seasonality**")
            if 'DayOfWeek' in st.session_state.train_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=st.session_state.train_df, x='DayOfWeek', y='Sales', ax=ax, palette='magma')
                ax.set_title('Average Sales by Day of Week')
                ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
                st.pyplot(fig)
                
        st.markdown("---")
        
        # Row 3: Correlation and Feature Importance
        col5, col6 = st.columns(2)
        with col5:
            st.write("**Feature Correlation Matrix**")
            numeric_df = st.session_state.train_df.select_dtypes(include=[np.number])
            corr_cols = [c for c in numeric_df.columns if any(x in c for x in ['Sales', 'Promo', 'Day', 'Month', 'Customers'])]
            if len(corr_cols) > 1:
                corr = numeric_df[corr_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                ax.set_title('Correlation Matrix of Key Features')
                st.pyplot(fig)
        
        with col6:
            st.write("**Model Feature Importance**")
            if st.session_state.forecaster is not None:
                importance_df = st.session_state.forecaster.get_feature_importance(top_n=10)
                if not importance_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='rocket')
                    ax.set_title('Top 10 Drivers of Sales Forecast')
                    st.pyplot(fig)
                else:
                    st.info("Train the model to see feature importance.")
            else:
                st.info("Train the model to see feature importance.")

        st.markdown("---")
        
        # Row 4: Sales vs Customers (if available)
        if 'Customers' in st.session_state.train_df.columns:
            st.write("**Sales vs. Customer Traffic**")
            fig, ax = plt.subplots(figsize=(12, 6))
            # Sample data for better visualization if dataset is huge
            sample_df = st.session_state.train_df.sample(min(5000, len(st.session_state.train_df)))
            sns.scatterplot(data=sample_df, x='Customers', y='Sales', hue='Promo', alpha=0.5, ax=ax)
            ax.set_title('Relationship between Customer Count and Sales Revenue')
            st.pyplot(fig)
    
    # Tab 5: XAI Insights
    with tab5:
        st.subheader("🔍 Explainable AI (XAI) - Model Interpretability")
        
        if st.session_state.forecaster is not None:
            st.write("Understand why the model makes specific predictions using SHAP (SHapley Additive exPlanations).")
            
            selected_store = st.selectbox(
                "Select Store for XAI Analysis",
                options=sorted(st.session_state.train_df['Store'].unique()),
                key="xai_store"
            )
            
            if st.button("🔍 Analyze Prediction", use_container_width=True):
                with st.spinner("Generating XAI explanation..."):
                    try:
                        # Initialize explainer
                        xai = XAIExplainer(
                            st.session_state.forecaster.model,
                            feature_names=st.session_state.forecaster.feature_names
                        )
                        data = xai.explain_prediction(selected_store)
                        
                        st.success("✅ XAI Analysis Complete!")
                        
                        # Display high-level metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Base Value (Average)", f"€{data['base_value']:,.2f}")
                        with col2:
                            st.metric("Final Prediction", f"€{data['prediction']:,.2f}")
                        with col3:
                            impact = data['prediction'] - data['base_value']
                            st.metric("Total Feature Impact", f"€{impact:,.2f}", delta=f"{impact:,.2f}")
                        
                        st.markdown("---")
                        
                        # Display explanation
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Top Factors Increasing Sales:**")
                            pos_factors = {k: f"+€{v:,.2f}" for k, v in data['contributions'].items() if v > 0}
                            st.json(pos_factors)
                        
                        with col2:
                            st.write("**Top Factors Decreasing Sales:**")
                            neg_factors = {k: f"-€{abs(v):,.2f}" for k, v in data['contributions'].items() if v < 0}
                            st.json(neg_factors)
                            
                        # Plot SHAP values
                        st.markdown("---")
                        st.write("**Feature Contribution (SHAP Waterfall Analysis)**")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # Get top 10 contributions
                        top_items = list(data['contributions'].items())[:10]
                        features = [i[0] for i in top_items]
                        values = [i[1] for i in top_items]
                        
                        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
                        ax.barh(features, values, color=colors)
                        ax.set_title(f'How Features Pushed the Forecast for Store {selected_store}')
                        ax.set_xlabel('Impact on Sales Forecast (€)')
                        plt.gca().invert_yaxis()
                        
                        # Add a vertical line at 0
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
                        
                        st.pyplot(fig)
                        
                        st.info("💡 **Interpretation:** Green bars show features that increased the sales forecast above the average (Base Value), while red bars show features that pulled the forecast down.")
                        
                    except Exception as e:
                        st.error(f"❌ Error in XAI analysis: {str(e)}")
        else:
            st.info("⏳ Please train the model first.")
    
    # Tab 6: Use Cases
    with tab6:
        st.subheader("📋 Use Cases & Workflows")
        
        use_cases = [
            {
                "title": "1️⃣ Generate Sales Forecasts",
                "description": "Inventory managers can generate accurate sales forecasts for specific stores and time periods to plan inventory levels.",
                "steps": [
                    "Navigate to the 'Forecasting' tab",
                    "Select a store and forecast period",
                    "Click 'Generate Forecast' to see predictions",
                    "Use the forecast to adjust stock levels"
                ]
            },
            {
                "title": "2️⃣ Obtain Reorder Recommendations",
                "description": "Get automatic reorder suggestions based on forecasted demand, safety stock, and economic order quantities.",
                "steps": [
                    "Go to 'Reorder Recommendations' tab",
                    "Select a store",
                    "Click 'Generate Recommendations'",
                    "Review reorder point, safety stock, and EOQ"
                ]
            },
            {
                "title": "3️⃣ Monitor Inventory Performance",
                "description": "Track key performance indicators and trends across stores to identify optimization opportunities.",
                "steps": [
                    "Visit the 'Analytics' tab",
                    "Review sales distributions and trends",
                    "Compare store performance",
                    "Identify seasonal patterns"
                ]
            },
            {
                "title": "4️⃣ Understand Model Predictions (XAI)",
                "description": "Gain transparency into why the model makes specific forecasts using explainable AI techniques.",
                "steps": [
                    "Go to 'XAI Insights' tab",
                    "Select a store",
                    "Click 'Analyze Prediction'",
                    "Review feature importance and contributions"
                ]
            },
            {
                "title": "5️⃣ Simulate What-If Scenarios",
                "description": "Test business decisions by simulating different scenarios (promotions, holidays, competition) and their impact on sales.",
                "steps": [
                    "Navigate to 'What-If Scenarios' tab",
                    "Select a store and forecast period",
                    "Choose a scenario type (Promotion, Holiday, Competition)",
                    "Adjust parameters and compare results"
                ]
            }
        ]
        
        for use_case in use_cases:
            with st.expander(use_case["title"]):
                st.write(f"**Description:** {use_case['description']}")
                st.write("**Steps:**")
                for i, step in enumerate(use_case["steps"], 1):
                    st.write(f"{i}. {step}")
    
    # Tab 7: Custom Date Range Analysis
    with tab7:
        st.subheader("📅 Custom Date Range Analysis")
        st.write("Select a custom date range to analyze sales trends and generate forecasts for specific periods.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Select Start Date",
                value=pd.to_datetime(st.session_state.train_df['Date'].min()),
                min_value=pd.to_datetime(st.session_state.train_df['Date'].min()),
                max_value=pd.to_datetime(st.session_state.train_df['Date'].max()),
                key="custom_start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "Select End Date",
                value=pd.to_datetime(st.session_state.train_df['Date'].max()),
                min_value=pd.to_datetime(st.session_state.train_df['Date'].min()),
                max_value=pd.to_datetime(st.session_state.train_df['Date'].max()),
                key="custom_end_date"
            )
        
        if start_date <= end_date:
            filtered_df = st.session_state.train_df[
                (pd.to_datetime(st.session_state.train_df['Date']) >= pd.to_datetime(start_date)) &
                (pd.to_datetime(st.session_state.train_df['Date']) <= pd.to_datetime(end_date))
            ]
            
            if len(filtered_df) > 0:
                st.success(f"✅ Data loaded: {len(filtered_df):,} records from {start_date} to {end_date}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Sales", f"€{filtered_df['Sales'].sum():,.0f}")
                with col2:
                    st.metric("Avg Daily Sales", f"€{filtered_df['Sales'].mean():,.0f}")
                with col3:
                    st.metric("Total Customers", f"{filtered_df['Customers'].sum():,}" if 'Customers' in filtered_df.columns else "N/A")
                with col4:
                    st.metric("Days in Range", len(filtered_df))
                
                st.markdown("---")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                daily_sales = filtered_df.groupby('Date')['Sales'].sum()
                ax.plot(daily_sales.index, daily_sales.values, linewidth=2, color='steelblue', marker='o')
                ax.fill_between(daily_sales.index, daily_sales.values, alpha=0.3, color='steelblue')
                ax.set_title(f'Sales Trend: {start_date} to {end_date}')
                ax.set_ylabel('Daily Sales (€)')
                ax.set_xlabel('Date')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("Store Performance in Selected Range")
                store_performance = filtered_df.groupby('Store').agg({
                    'Sales': ['sum', 'mean', 'std'],
                    'Customers': 'sum' if 'Customers' in filtered_df.columns else 'count'
                }).round(2)
                st.dataframe(store_performance, use_container_width=True)
            else:
                st.warning("⚠️ No data found for the selected date range.")
        else:
            st.error("❌ Start date must be before end date.")
    
    # Tab 8: What-If Scenarios
    with tab8:
        st.subheader("🎯 What-If Scenario Simulator")
        st.write("Simulate different business scenarios and see how they impact sales forecasts and reorder recommendations.")
        
        if st.session_state.scenario_simulator is not None:
            try:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_store = st.selectbox(
                        "Select Store for Scenario",
                        options=sorted(st.session_state.train_df['Store'].unique()),
                        key="scenario_store_select"
                    )
                
                with col2:
                    forecast_days_scenario = st.slider(
                        "Days to Forecast",
                        min_value=7,
                        max_value=90,
                        value=30,
                        step=7,
                        key="scenario_days_slider"
                    )
                
                st.markdown("---")
                
                baseline_scenario = st.session_state.scenario_simulator.create_baseline_scenario(selected_store, forecast_days_scenario)
                baseline_result = st.session_state.scenario_simulator.forecast_scenario(baseline_scenario)
                
                st.subheader("📊 Baseline Scenario (No Changes)")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg Daily Sales", f"€{baseline_result['statistics']['Average_Sales']:,.0f}")
                with col2:
                    st.metric("Total Sales", f"€{baseline_result['statistics']['Total_Sales']:,.0f}")
                with col3:
                    st.metric("Max Daily Sales", f"€{baseline_result['statistics']['Max_Sales']:,.0f}")
                with col4:
                    st.metric("Volatility (Std Dev)", f"{baseline_result['statistics']['Std_Dev']:,.0f}")
                
                st.markdown("---")
                st.subheader("🎛️ Modify Scenario")
                
                scenario_type = st.radio(
                    "Select Scenario Type",
                    options=["Promotion", "Holiday", "Competition"],
                    horizontal=True,
                    key="scenario_type_radio"
                )
                
                modified_scenario = baseline_scenario.copy()
                modified_result = None
                
                if scenario_type == "Promotion":
                    st.write("**Simulate a promotional campaign**")
                    promo_intensity = st.slider(
                        "Promotion Days (out of next days)",
                        min_value=0,
                        max_value=forecast_days_scenario,
                        value=7,
                        step=1,
                        key="promo_intensity_slider"
                    )
                    
                    if promo_intensity > 0:
                        promo_dates = pd.to_datetime(baseline_scenario['Date'].sample(promo_intensity).values)
                        modified_scenario = st.session_state.scenario_simulator.apply_promotion_scenario(baseline_scenario, promo_dates)
                        modified_result = st.session_state.scenario_simulator.forecast_scenario(modified_scenario)
                        st.success(f"✅ Promotion applied to {promo_intensity} days")
                
                elif scenario_type == "Holiday":
                    st.write("**Simulate holiday periods**")
                    holiday_days = st.slider(
                        "Holiday Days",
                        min_value=0,
                        max_value=forecast_days_scenario,
                        value=3,
                        step=1,
                        key="holiday_days_slider"
                    )
                    
                    if holiday_days > 0:
                        holiday_dates = pd.to_datetime(baseline_scenario['Date'].sample(holiday_days).values)
                        modified_scenario = st.session_state.scenario_simulator.apply_holiday_scenario(baseline_scenario, holiday_dates, 'StateHoliday')
                        modified_result = st.session_state.scenario_simulator.forecast_scenario(modified_scenario)
                        st.success(f"✅ Holiday scenario applied to {holiday_days} days")
                
                elif scenario_type == "Competition":
                    st.write("**Simulate competition distance changes**")
                    new_distance = st.slider(
                        "Competition Distance (km)",
                        min_value=0.0,
                        max_value=50.0,
                        value=5.0,
                        step=0.5,
                        key="competition_distance_slider"
                    )
                    
                    modified_scenario = st.session_state.scenario_simulator.apply_competition_scenario(baseline_scenario, new_distance)
                    modified_result = st.session_state.scenario_simulator.forecast_scenario(modified_scenario)
                    st.success(f"✅ Competition distance set to {new_distance} km")
                
                if modified_result is not None:
                    st.markdown("---")
                    st.subheader("📊 Modified Scenario Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Avg Daily Sales", f"€{modified_result['statistics']['Average_Sales']:,.0f}")
                    with col2:
                        st.metric("Total Sales", f"€{modified_result['statistics']['Total_Sales']:,.0f}")
                    with col3:
                        st.metric("Max Daily Sales", f"€{modified_result['statistics']['Max_Sales']:,.0f}")
                    with col4:
                        st.metric("Volatility (Std Dev)", f"{modified_result['statistics']['Std_Dev']:,.0f}")
                    
                    comparison = st.session_state.scenario_simulator.compare_scenarios(baseline_result, modified_result)
                    
                    st.markdown("---")
                    st.subheader("📈 Impact Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Sales Impact",
                            f"€{comparison['Sales_Change']:,.0f}",
                            delta=f"{comparison['Sales_Change_Percent']:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Total Impact",
                            f"€{comparison['Total_Sales_Impact']:,.0f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Volatility Change",
                            f"{comparison['Volatility_Change']:,.0f}"
                        )
                    
                    st.markdown("---")
                    st.subheader("📊 Forecast Comparison")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(baseline_result['forecasts']['Date'], baseline_result['forecasts']['Forecasted_Sales'], 
                           label='Baseline', linewidth=2, marker='o')
                    ax.plot(modified_result['forecasts']['Date'], modified_result['forecasts']['Forecasted_Sales'], 
                           label='Modified Scenario', linewidth=2, marker='s', linestyle='--')
                    ax.set_title(f'Sales Forecast Comparison - Store {selected_store}')
                    ax.set_ylabel('Forecasted Sales (€)')
                    ax.set_xlabel('Date')
                    ax.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    report = st.session_state.scenario_simulator.generate_scenario_report(
                        f"{scenario_type} Scenario",
                        baseline_result,
                        modified_result
                    )
                    
                    st.text(report)
            
            except Exception as e:
                st.error(f"❌ Error in scenario simulation: {str(e)}")
        else:
            st.warning("⚠️ Please train the model first to use the scenario simulator.")

else:
    st.info("👈 Please ensure 'train.csv' and 'store.csv' are in the 'data/' folder.")