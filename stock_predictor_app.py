# stock_predictor_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model and scaler
model_save_path = 'random_forest_model.pkl'
scaler_save_path = 'scaler.pkl'

# Load the trained Random Forest model and scaler
rf_model = joblib.load(model_save_path)
scaler = joblib.load(scaler_save_path)

# Function to create lagged features
def create_lagged_features(data, n_lags=5):
    df = data.copy()
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

# Function to fetch historical data and predict future prices
def fetch_and_predict(ticker, end_date, n_days=5):
    # Fetch historical market data using yfinance
    start_date = '2019-11-18'  # Fixed start date for training data
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
    
    # Scale the features using the same scaler fitted on training data
    specific_date_features_scaled = scaler.transform(specific_date_features_with_lags)
    
    # Predict prices for next n_days (business days only)
    predicted_next_days = []
    last_known_scaled_data = specific_date_features_scaled[-1].copy()
    
    for _ in range(n_days):
        next_day_prediction = rf_model.predict(last_known_scaled_data.reshape(1, -1))[0]
        predicted_next_days.append(next_day_prediction)
        
        # Update last_known_scaled_data with new predictions (shift lagged features)
        last_known_scaled_data[-5:] = np.roll(last_known_scaled_data[-5:], shift=1)
        last_known_scaled_data[-5] = next_day_prediction[3]  # Update with predicted Close price
    
    # Generate future business dates
    last_date = pd.to_datetime(end_date)
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days).tolist()
    
    return future_dates, np.array(predicted_next_days)

# Streamlit App Interface
st.title("SmartTrader Console")

# Input section: Date picker and Predict button
input_date = st.date_input("Select a Date", pd.to_datetime('2024-11-18'))
if st.button("Predict"):
    try:
        # Fetch and predict stock prices from the input date
        future_dates, predictions = fetch_and_predict('NVDA', input_date.strftime('%Y-%m-%d'), n_days=5)
        
        # Round off predictions to 2 decimal places and format dates
        predictions_rounded = np.round(predictions, 2)
        future_dates_formatted = [date.strftime('%Y-%m-%d') for date in future_dates]
        
        # Create a DataFrame with formatted dates and rounded predictions
        prediction_df = pd.DataFrame(predictions_rounded, columns=['Open', 'High', 'Low', 'Close'], index=future_dates_formatted)
        prediction_df.index.name = "Date"
        
        # Explicitly format the DataFrame to ensure proper rounding display
        prediction_df = prediction_df.style.format("{:.2f}")
        
        # Display message before showing the table
        st.write("Predicted prices for the next 5 business days in USD are:")
        
        # Display predicted values in a tabular format with formatted dates and rounded prices
        st.dataframe(prediction_df)
        
    except Exception as e:
        st.error(f"Error: {e}")
