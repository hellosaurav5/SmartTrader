# SmartTrader Console

Welcome to **SmartTrader Console**, a powerful Streamlit application designed to predict stock prices and implement trading strategies for NVDA and NVDQ stocks. This app allows users to predict stock prices for the next 5 business days and determine optimal trading actions (IDLE, BULLISH, BEARISH) to maximize net equity value.

---

## Team Members

| Name                  | Email                        | Student ID   |
|-----------------------|------------------------------|--------------|
|   Saurav Kashyap      | saurav@sjsu.edu              | 018205655    |
|   Meet Hitesh Thakkar | meethitesh.thakkar@sjsu.edu  | 017792476   |

---

## App URL

You can access the deployed app at the following URL:

👉 **[SmartTrader Console](https://smarttrader-team14-saurav-meet.streamlit.app)** - https://smarttrader-team14-saurav-meet.streamlit.app

1. **App Functionality**:
   - The app predicts stock prices for NVDA and NVDQ for the next 5 business days.
   - It implements a trading strategy based on predicted prices:
     - **IDLE**: No action is taken.
     - **BULLISH**: Swaps all NVDQ shares for NVDA shares using NVDA's open price.
     - **BEARISH**: Swaps all NVDA shares for NVDQ shares using NVDQ's open price.
   - At the end of the 5th business day, the app calculates and displays the total net equity value.

2. **How to Use the App**:
   - Select a date using the date picker.
   - Click the "Predict" button to view:
     - Predicted stock prices for NVDA.
     - Trading actions for each business day.
     - Final net equity value after 5 business days.

3. **Expected Outputs**:
   - A table of predicted NVDA stock prices (Open, High, Low, Close).
   - A table of trading actions (IDLE, BULLISH, BEARISH) for each day.

---

## Build Instructions

Follow these steps to build and run the app locally:

### Prerequisites
- Python 3.8 or higher installed on your system.
- Required Python libraries: `streamlit`, `yfinance`, `pandas`, `numpy`, `scikit-learn`, `joblib`.

### Steps
1. Clone the repository or download the app code:
git clone https://github.com/hellosaurav5/SmartTrader.git

2. Install dependencies using `pip`:
    ```python
    pip install requirements.txt
    ```

3. Run the Streamlit app:
    ```python
    streamlit run app.py
    ```


4. Open your browser and navigate to `http://localhost:8501` to interact with the app.


---

## Contact Information

For any queries or issues related to this app, please contact:

- Saurav Kashyap: saurav@sjsu.edu
- Meet Hitesh Thakkar: meethitesh.thakkar@sjsu.edu
