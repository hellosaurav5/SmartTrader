import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained Random Forest models and scalers
nvda_model_path = 'random_forest_nvda.pkl'
nvdq_model_path = 'random_forest_nvdq.pkl'
nvda_scaler_path = 'scaler_nvda.pkl'
nvdq_scaler_path = 'scaler_nvdq.pkl'

nvda_model = joblib.load(nvda_model_path)
nvdq_model = joblib.load(nvdq_model_path)
nvda_scaler = joblib.load(nvda_scaler_path)
nvdq_scaler = joblib.load(nvdq_scaler_path)

# Function to create lagged features
def create_lagged_features(data, n_lags=5):
    df = data.copy()
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

# Function to fetch historical data and predict future prices
def fetch_and_predict(ticker, model, scaler, end_date, n_days=5):
    # Fetch historical market data using yfinance
    start_date = '2021-11-18'  # Fixed start date for training data
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    
    # Ensure there is enough data for lagged features
    if len(data) < 10:
        raise ValueError("Not enough data available to create lagged features.")
    
    # Create lagged features for the last few rows of data
    specific_date_data = data.iloc[-10:]  # Ensure at least n_lags + 5 rows
    specific_date_features_with_lags = create_lagged_features(specific_date_data)
    
    # Align columns of prediction data with training data
    training_columns = ['Open', 'High', 'Low', 'Volume'] + [f'lag_{i}' for i in range(1, 6)]
    specific_date_features_with_lags = specific_date_features_with_lags[training_columns]
    
    # Scale the features using the loaded scaler
    specific_date_features_scaled = scaler.transform(specific_date_features_with_lags)
    
    # Predict prices for next n_days (business days only)
    predicted_next_days = []
    last_known_scaled_data = specific_date_features_scaled[-1].copy()
    
    for _ in range(n_days):
        next_day_prediction = model.predict(last_known_scaled_data.reshape(1, -1))[0]
        predicted_next_days.append(next_day_prediction)
        
        # Update last_known_scaled_data with new predictions (shift lagged features)
        last_known_scaled_data[-5:] = np.roll(last_known_scaled_data[-5:], shift=1)
        last_known_scaled_data[-5] = next_day_prediction[3]  # Update with predicted Close price
    
    # Generate future business dates
    last_date = pd.to_datetime(end_date)
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days).tolist()
    
    return future_dates, np.array(predicted_next_days)

# Function to determine trading actions and calculate net value
def determine_trading_actions(nvda_predictions, nvdq_predictions, initial_nvda_shares=10000, initial_nvdq_shares=100000):
    actions = []
    nvda_shares = initial_nvda_shares
    nvdq_shares = initial_nvdq_shares
    
    for i in range(len(nvda_predictions)):
        open_nvda = nvda_predictions[i][0]
        close_nvda = nvda_predictions[i][3]
        open_nvdq = nvdq_predictions[i][0]
        close_nvdq = nvdq_predictions[i][3]
        
        if close_nvda > open_nvda and close_nvdq < open_nvdq:
            action = 'BULLISH'
            nvda_shares += nvdq_shares / open_nvda
            nvdq_shares = 0
        elif close_nvda < open_nvda and close_nvdq > open_nvdq:
            action = 'BEARISH'
            nvdq_shares += nvda_shares * open_nvdq
            nvda_shares = 0
        else:
            action = 'IDLE'
        
        actions.append(action)
    
    final_value_nvda = nvda_shares * nvda_predictions[-1][3]
    final_value_nvdq = nvdq_shares * nvdq_predictions[-1][3]
    final_value = final_value_nvda + final_value_nvdq
    
    return actions, final_value

# Streamlit App Interface
st.title("SmartTrader Console")

# Input section: Date picker and Predict button
input_date = st.date_input("Select a Date", pd.to_datetime('2024-12-16'))
if st.button("Predict"):
    try:
        # Fetch and predict stock prices from the input date for both NVDA and NVDQ
        nvda_future_dates, nvda_predictions = fetch_and_predict('NVDA', nvda_model, nvda_scaler, input_date.strftime('%Y-%m-%d'), n_days=5)
        nvdq_future_dates, nvdq_predictions = fetch_and_predict('NVDQ', nvdq_model, nvdq_scaler, input_date.strftime('%Y-%m-%d'), n_days=5)
        
        # Round off predictions to 2 decimal places and format dates
        nvda_predictions_rounded = np.round(nvda_predictions, 2)
        nvdq_predictions_rounded = np.round(nvdq_predictions, 2)

        # Display NVDA predictions table
        st.write("NVDA Predicted Prices for next 5 business days in USD:")
        nvda_df = pd.DataFrame(nvda_predictions_rounded, columns=['Open', 'High', 'Low', 'Close'], index=[date.strftime('%Y-%m-%d') for date in nvda_future_dates])
        st.dataframe(nvda_df)

        # Determine trading actions and calculate net value at the end of 5 days
        actions, final_value = determine_trading_actions(nvda_predictions_rounded, nvdq_predictions_rounded)

        # Display trading actions and final net value
        st.write("Trading Actions for Each Day:")
        action_df = pd.DataFrame(actions, index=[date.strftime('%Y-%m-%d') for date in nvda_future_dates], columns=["Action"])
        st.table(action_df)

        st.write(f"Final Net Value after 5 Days: ${final_value:,.2f}")
        
    except Exception as e:
        st.error(f"Error: {e}")
