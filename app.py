# solana_price_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import plotly.graph_objects as go

try:
    import yfinance as yf
except ModuleNotFoundError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance'])
    import yfinance as yf

try:
    import lightgbm
except ModuleNotFoundError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'lightgbm'])
    import lightgbm

class SolanaPricePredictor:
    def __init__(self, model_path='lightgbm_model.joblib', 
                 scaler_path='standard_scaler.joblib',
                 features_path='feature_names.joblib'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)

    def prepare_features(self, price_data):
        df = price_data.copy()
        df['Daily_Return'] = df['Close'].pct_change()
        df['Rolling_Volatility'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        return df.dropna()

    def predict(self, data):
        X = data[self.features]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_from_raw(self, price_data):
        prepared_data = self.prepare_features(price_data)
        return self.predict(prepared_data.iloc[[-1]])

def main():
    st.set_page_config(page_title="Solana Price Predictor", layout="wide", page_icon="🚀")
    st.title('Solana (SOL) Price Prediction App')
    st.markdown("""
    **Powered by Machine Learning** - Predicts Solana price movements using technical analysis indicators
    and a LightGBM regression model.
    """)

    with st.sidebar:
        st.header('Configuration')
        days_back = st.slider('Historical Data Window (days)', 30, 180, 90)
        st.markdown("---")
        st.caption("Model Information")
        st.write(f"Features used: {joblib.load('feature_names.joblib')}")

    predictor = SolanaPricePredictor()

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        sol_data = yf.download('SOL-USD', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)

        if sol_data.empty:
            st.error("Failed to fetch Solana price data")
            return

        current_price = sol_data['Close'].iloc[-1].item()
        prev_price = sol_data['Close'].iloc[-2].item()
        high_24h = sol_data['High'].iloc[-1].item()
        low_24h = sol_data['Low'].iloc[-1].item()

        cols = st.columns(3)
        cols[0].metric(
            "Current Price", 
            f"${current_price:.2f}",
            delta=f"${current_price - prev_price:.2f}",
            delta_color="normal" if current_price >= prev_price else "inverse"
        )
        cols[1].metric("24h High", f"${high_24h:.2f}")
        cols[2].metric("24h Low", f"${low_24h:.2f}")

        prediction = predictor.predict_from_raw(sol_data)
        predicted_price = current_price * (1 + prediction[0])

        st.markdown("## 📈 Price Prediction")
        pred_cols = st.columns(2)
        pred_cols[0].metric(
            "Next Day Prediction", 
            f"${predicted_price:.2f}",
            delta=f"{prediction[0]*100:.2f}%"
        )

        tech_data = predictor.prepare_features(sol_data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sol_data.index, y=sol_data['Close'], name='Price', line=dict(color='#636EFA')))
        fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA20'], name='20-day MA', line=dict(color='#FFA15A')))
        fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA50'], name='50-day MA', line=dict(color='#EF553B')))
        fig.update_layout(title='Price Analysis with Moving Averages', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## 📊 Technical Indicators")
        col1, col2 = st.columns(2)

        with col1:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=tech_data.index, y=tech_data['RSI'], name='RSI', line=dict(color='#00CC96')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(title='Relative Strength Index (RSI)', height=400)
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col2:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=tech_data.index, y=tech_data['Rolling_Volatility'], name='Volatility', line=dict(color='#AB63FA')))
            fig_vol.update_layout(title='30-day Rolling Volatility', height=400)
            st.plotly_chart(fig_vol, use_container_width=True)

    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.exception(e)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>💰 DISCLAIMER: Cryptocurrency predictions are for informational purposes only.</p>
        <p>⚙️ Model Version: 1.0.0 | Updated: 2025-05-27</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()