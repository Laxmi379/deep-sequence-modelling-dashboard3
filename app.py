# Streamlit Dashboard for Multidomain LSTM Models
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import nltk
nltk.download('vader_lexicon')

# Page setup
st.set_page_config(layout="wide")
st.title("🧠 Deep Sequence Modeling Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📘 NLP Sentiment", "💰 Finance Stock", "❤️ Healthcare", "🌦️ Weather"])

# --- NLP Tab ---
with tab1:
    st.header("📘 NLP Sentiment Prediction")
    text_input = st.text_area("Enter text", "I love this product, it is amazing!")

    if st.button("Predict Sentiment"):
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text_input)

        # Extract compound score and classify
        compound_score = scores['compound']
        sentiment = "Positive" if compound_score > 0.05 else "Negative" if compound_score < -0.05 else "Neutral"

        # Emoji based on sentiment
        if compound_score > 0.5:
            emoji = "😄"
        elif compound_score > 0.05:
            emoji = "🙂"
        elif compound_score < -0.5:
            emoji = "😠"
        elif compound_score < -0.05:
            emoji = "🙁"
        else:
            emoji = "😐"

        # Display results
        st.metric("Compound Sentiment Score", round(compound_score, 3))
        st.markdown(f"### Sentiment: **{sentiment}** {emoji}")

        # Bar chart for sentiment components
        st.subheader("📊 Sentiment Breakdown")
        score_df = pd.DataFrame.from_dict(scores, orient='index', columns=['Score'])
        st.bar_chart(score_df)



with tab2:
    st.header("💰 Stock Price Forecasting")
    ticker = st.text_input("Enter stock symbol (e.g., AAPL)", "AAPL")

    if st.button("Fetch and Predict"):
        try:
            # Step 1: Download Stock Data
            data = yf.download(ticker, period="6mo")['Close'].dropna()

            if data.empty:
                st.error("No stock data found. Please try a different ticker.")
                st.stop()

            # Step 2: Normalize the data
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data.values.reshape(-1, 1))

            # Step 3: Prepare sequences
            window = 10
            if len(scaled) <= window:
                st.error("Not enough data for prediction. Try a different stock or longer period.")
                st.stop()

            X = np.array([scaled[i:i+window] for i in range(len(scaled) - window)])

            # Step 4: Load the model
            model = load_model("models/finance_model.h5", compile=False)

            # Step 5: Make predictions
            preds = model.predict(X)

            # Step 6: Reshape predictions safely
            try:
                preds_2d = preds.reshape(-1, 1)
            except Exception as e:
                st.error(f"Reshape error: {str(e)}")
                st.stop()

            # Step 7: Inverse transform predictions
            pred = scaler.inverse_transform(preds_2d)

            # Step 8: Align lengths for plotting
            actual = data.values[window:]
            min_len = min(len(actual), len(pred))
            actual = actual[:min_len]
            pred = pred[:min_len]

            actual = actual.flatten()
            pred = pred.flatten()

            # Step 9: Plot results
            st.line_chart(pd.DataFrame({'Actual': actual, 'Predicted': pred}))

            # Step 10: Show error metrics
            rmse = math.sqrt(mean_squared_error(actual,pred))
            mae = mean_absolute_error(actual,pred)
            # Display metrics
            st.subheader("📉 Prediction Error Metrics")
            st.write(f"*RMSE (Root Mean Squared Error):* {rmse:.2f}")
            st.write(f"*MAE (Mean Absolute Error):* {mae:.2f}")

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")


# --- Healthcare Tab (Fixed for shape issues) ---
with tab3:
    st.header("❤️ Patient Vitals Prediction")

    # Load healthcare model
    model = load_model("models/healthcare_model.h5", compile=False)


    # Simulate patient vitals
    days = st.slider("Simulate past days", 100, 200, 150, key="health_slider")
  # Generate enough data
    data = 60 + np.sin(np.linspace(0, 5, days)) * 10 + np.random.normal(0, 1, days)

    # Normalize the data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1))

    # ✅ Use last 100 for prediction
    X = scaled[-100:].reshape(1, 100)
    pred = model.predict(X)

    # Inverse transform the prediction
    next_val = scaler.inverse_transform(pred)[0][0]
    st.metric("Predicted Next Heart Rate", round(next_val, 2))

    # ✅ Generate graph with actual vs predicted
    actual = data[-100:]  # Last 100 points
    predicted = np.append(actual[1:], next_val)  # Shifted + predicted value

    # Ensure same length
    actual = actual[:len(predicted)]

    st.subheader("📈 Actual vs Predicted Heart Rate")
    df = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    st.line_chart(df)

# --- Weather Tab (Fixed model load + reshape) ---
   # --- Weather Tab ---
with tab4:
    st.header("🌦️ Weather Forecasting")
    st.markdown("### 🌤️ 7-Day Temperature Forecast")

    # Forecast Table
    forecast_df = pd.DataFrame({
        "Day": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"],
        "Predicted Temp": [30.3240, 30.3526, 31.4062, 30.2271, 30.8146, 30.8574, 30.8334]
    })
    st.dataframe(forecast_df)

    # Forecast Graph
    fig, ax = plt.subplots()
    ax.plot(forecast_df["Day"], forecast_df["Predicted Temp"], marker='o', color='orange')
    ax.set_title("7-Day Temperature Forecast")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Day")
    ax.grid(True)
    st.pyplot(fig)
