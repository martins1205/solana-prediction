
# Solana Price Prediction App

This project is designed to predict the future price of Solana (SOL) cryptocurrency using advanced machine learning techniques. The application integrates a LightGBM model trained on historical price data and several technical indicators to provide accurate predictions.

## Features
- Predicts Solana price movements based on historical data.
- Real-time cryptocurrency data fetching using yfinance.
- Visualizes price trends, RSI, volatility, and other technical indicators.
- Interactive user interface built with Streamlit.

## Deployment
This project includes a Streamlit app for deployment, which provides real-time updates, predictions, and visualizations.

## Installation Requirements
Clone the repository and ensure the following dependencies are installed:

```bash
pip install -r requirements.txt
```

## Contents
### Files
- `app.py`: Streamlit app for price prediction.
- `lightgbm_model.joblib`: Trained LightGBM model for predictions.
- `standard_scaler.joblib`: StandardScaler object for feature scaling.
- `feature_names.joblib`: List of features used for predictions.
- `solusd_dataset.csv`: Historical price data used for analysis and training. https://www.kaggle.com/datasets/itsmecevi/solusd-daily-ticker-price-weekly-update

### Prediction Pipeline
1. Fetch real-time or historical Solana price data.
2. Calculate technical indicators and prepare features.
3. Scale the features and apply the LightGBM model.
4. Display the prediction and visualization in the Streamlit app.

## Running the App
Run the Streamlit app using the following command:![Screenshot 2025-05-27 171250](https://github.com/user-attachments/assets/5556b624-85dc-4d09-b313-9308a16ab3a4)

```bash
streamlit run app.py
```

## Disclaimer
This project is for educational purposes only and does not constitute financial advice. Cryptocurrency trading is risky, and users should exercise caution.

---
Developed with ❤️ using Streamlit and LightGBM.

