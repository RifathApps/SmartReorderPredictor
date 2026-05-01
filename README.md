# Smart Reorder Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python )
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=for-the-badge&logo=streamlit )
![XGBoost](https://img.shields.io/badge/XGBoost-v1.x-orange?style=for-the-badge&logo=xgboost )
![SHAP](https://img.shields.io/badge/SHAP-v0.x-green?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNTAwIDI1MDAiPjxwYXRoIGZpbGw9IiMwMDAwMDAiIGQ9Ik0xMjUwIDBDNTU5LjcgMCAwIDU1OS43IDAgMTI1MHM1NTkuNyAyNTAwIDEyNTAgMjUwMCAyNTAwLTU1OS43IDI1MDAtMTI1MFMxOTQwLjMgMCAxMjUwIDBabTAtMjAwYzY5MC4zIDAgMTI1MCA1NTkuNyAxMjUwIDEyNTBzLTU1OS43IDEyNTAtMTI1MCAxMjUwUzAgMTk0MC4zIDAgMTI1MCA1NTkuNyAwIDEyNTAgMFptMCAyMDAwYzU1Mi4zIDAgMTAwMC00NDcuNyAxMDAwLTEwMDBTMTgwMi4zIDI1MCAxMjUwIDI1MFMyNTAgNjk3LjcgMjUwIDEyNTBzNDQ3LjcgMTAwMCAxMDAwIDEwMDBabTAtMjAwYzMzMS40IDAgNjAwLTI2OC42IDYwMC02MDBTMTU4MS40IDY1MCAxMjUwIDY1MFM2NTAgOTgxLjQgNjUwIDEzNTBzMjY4LjYgNjAwIDYwMCA2MDBabTAtMjAwYzIyMS40IDAgNDAwLTE3OC42IDQwMC00MDBTMTQ3MS40IDg1MCAxMjUwIDg1MFM4NTAgMTAyMS40IDg1MCAxMjUwcDE3OC42IDQwMCA0MDAgNDAwWm0wLTIwMGMxMTAuNSAwIDIwMC04OS41IDIwMC0yMDBTMTM2MC41IDEwNTAgMTI1MCAxMDUwUzEwNTAgMTE0MC41IDEwNTAgMTI1MHwxMjUwIDEyNTB6Ii8+PC9zdmc+ ) 

## 🚀 Smart Reorder Predictor

A data science product prototype designed to empower small e-commerce businesses with intelligent inventory management. This project leverages advanced machine learning (XGBoost) for sales forecasting, integrates Explainable AI (SHAP) for transparency, and provides actionable reorder recommendations to minimize stockouts and optimize inventory costs.

--- 

## ✨ Features

-   **Automated Data Ingestion:** Seamlessly loads historical sales and store data from CSV files.
-   **Robust Data Preprocessing:** Handles missing values, engineers rich time-series features, and prepares data for modeling.
-   **XGBoost Sales Forecasting:** Utilizes a high-performance XGBoost model to predict future sales with high accuracy.
-   **Explainable AI (XAI) Insights:** Integrates **SHAP (SHapley Additive exPlanations)** to provide transparency into model predictions, showing *why* a specific sales forecast was made.
-   **Intelligent Reorder Recommendations:** Calculates optimal Safety Stock, Reorder Points, and Economic Order Quantities (EOQ) based on forecasts.
-   **Interactive Analytics Dashboard:** Visualizes sales trends, seasonality, promotional impact, and feature correlations.
-   **Custom Date Range Analysis:** Allows users to analyze sales performance over any selected period.
-   **"What-If" Scenario Simulator:** Enables business users to test the impact of hypothetical promotions, holidays, or competitor changes on future sales.
-   **User-Friendly Interface:** Built with Streamlit for an intuitive and interactive web application experience.

--- 

## 📂 Project Structure

--- 
prototype/
├── README.md                  # Project overview and instructions
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python dependencies
├── data_preprocessing.py      # Module for data cleaning and feature engineering
├── model_training.py          # Module for XGBoost model training and evaluation
├── reorder_engine.py          # Module for inventory optimization logic
├── xai_module.py              # Module for SHAP-based Explainable AI
├── scenario_simulator.py      # Module for What-If scenario generation
├── data/                      # Directory for raw and processed data (e.g., store.csv, train.csv)
├── models/                    # Directory for trained machine learning models (e.g., sales_forecast_model.pkl)
└── PROJECT_DOCUMENTATION.md   # Comprehensive project documentation



## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd prototype
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

--- 

## 📊 Data

The prototype utilizes the **Rossmann Store Sales** dataset. Please download the `store.csv` and `train.csv` files from the Kaggle competition page and place them into the `data/` directory:

[Rossmann Store Sales Kaggle Competition](https://www.kaggle.com/competitions/rossmann-store-sales/data )

--- 

## ▶️ Running the Application

To launch the Streamlit dashboard, navigate to the `prototype/` directory and run:

```bash
streamlit run app.py


---

<img width="1918" height="991" alt="image" src="https://github.com/user-attachments/assets/b4655a67-e818-4cc2-9dbc-393383bb6df6" />
