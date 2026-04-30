# Smart Reorder Predictor Prototype

This repository contains the proof-of-concept prototype for the 'Smart Reorder Predictor', a data science product designed to assist small e-commerce businesses with enhanced inventory management through sales forecasting and reorder recommendations.

## Project Structure

```
prototype/
├── README.md
├── app.py                # Streamlit application for the dashboard
├── requirements.txt      # Python dependencies
├── data/                 # Directory for raw and processed data
│   ├── store.csv
│   ├── train.csv
│   └── test.csv
├── models/               # Directory for trained machine learning models
│   └── sales_forecast_model.pkl
└── notebooks/            # (Optional) Jupyter notebooks for EDA and model training
    └── data_exploration_and_training.ipynb
```

## Setup and Installation

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

## Data

The prototype utilizes the **Rossmann Store Sales** dataset. Please download the `store.csv`, `train.csv`, and `test.csv` files from the Kaggle competition page and place them into the `data/` directory:

[Rossmann Store Sales Kaggle Competition](https://www.kaggle.com/competitions/rossmann-store-sales/data)

## Running the Application

To launch the Streamlit dashboard, navigate to the `prototype/` directory and run:

```bash
streamlit run app.py
```

This will open the application in your web browser, typically at `http://localhost:8501`.

## Features

*   **Sales Forecasting:** View predicted sales for stores/products.
*   **Reorder Recommendations:** Get suggestions for optimal stock levels.
*   **Interactive Dashboard:** Explore data and insights through a user-friendly interface.

## Model Training (Optional)

If you wish to retrain the model or explore the data, refer to the Jupyter notebooks in the `notebooks/` directory. The `sales_forecast_model.pkl` file in the `models/` directory is a pre-trained model for demonstration purposes.

## Technologies Used

*   Python
*   Streamlit
*   Pandas, NumPy
*   Scikit-learn, XGBoost

## License

[Specify your license here, e.g., MIT License]
