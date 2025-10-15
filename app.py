# Import necessary libraries
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import xgboost as xgb
import numpy as np
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- PAGE SETUP ---
st.set_page_config(page_title="Investment Dashboard", layout="wide")

# --- MODEL AND TOKENIZER LOADING (Cached) ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

# --- HELPER FUNCTIONS ---
def get_news(api_key, ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}&language=en&sortBy=publishedAt&pageSize=20"
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        return [article['title'] for article in articles]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return []

def analyze_sentiment(headlines):
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    df = pd.DataFrame({
        'Headline': headlines,
        'Positive': predictions[:, 0].tolist(),
        'Negative': predictions[:, 1].tolist(),
        'Neutral': predictions[:, 2].tolist()
    })
    return df

# --- APP TITLE ---
st.title("AI-Powered Quantitative Investment Strategy Dashboard 📈")

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Choose a tool", ["Single Stock Analysis", "Portfolio Optimization"])

# ==============================================================================
# --- SINGLE STOCK ANALYSIS MODE ---
# ==============================================================================
if app_mode == "Single Stock Analysis":
    st.sidebar.header("User Input")
    ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
    news_api_key = st.sidebar.text_input("Enter NewsAPI Key", "091192f036464a36bd3c00643d1d40d3", type="password") # API key: 091192f036464a36bd3c00643d1d40d3
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

    try:
        st.header(f"Displaying Data for: {ticker_symbol}")
        ticker_data = yf.Ticker(ticker_symbol)
        df = ticker_data.history(start=start_date, end=end_date)

        if df.empty:
            st.error("No data found. Please check the ticker symbol or date range.")
        else:
            # Company Info, Charts, News, and Prediction sections from before
            st.subheader("Company Information")
            st.markdown(f"**Name:** {ticker_data.info.get('longName', 'N/A')} | **Sector:** {ticker_data.info.get('sector', 'N/A')} | **Industry:** {ticker_data.info.get('industry', 'N/A')}")
            
            st.subheader("Historical Price Chart")
            fig = px.line(df, x=df.index, y='Close', title=f'{ticker_symbol} Close Price History')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📰 Latest News & Sentiment Analysis")
            if not news_api_key:
                st.warning("Please enter your NewsAPI key to fetch news.")
            else:
                headlines = get_news(news_api_key, ticker_symbol)
                if headlines:
                    sentiment_df = analyze_sentiment(headlines)
                    sentiment_df['Dominant'] = sentiment_df[['Positive', 'Negative', 'Neutral']].idxmax(axis=1)
                    st.dataframe(sentiment_df)
                    sentiment_counts = sentiment_df['Dominant'].value_counts()
                    fig_pie = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment Distribution')
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No recent news found.")
            
            st.subheader("🚀 Stock Price Prediction")
            n_days = st.slider("Select number of days to predict:", 1, 30, 5, key="days_slider")
            df_pred = df[['Close']].copy()
            for i in range(1, 4):
                df_pred[f'Lag_{i}'] = df_pred['Close'].shift(i)
            df_pred.dropna(inplace=True)
            X = df_pred.drop('Close', axis=1)
            y = df_pred['Close']
            model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
            model_xgb.fit(X, y)
            last_features = [X.iloc[-1].values]
            future_predictions = []
            for _ in range(n_days):
                next_pred = model_xgb.predict(np.array(last_features))[0]
                future_predictions.append(next_pred)
                new_features = np.roll(last_features[0], 1)
                new_features[0] = next_pred
                last_features = [new_features]
            last_date = df.index[-1]
            future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, n_days + 1)])
            predictions_df = pd.DataFrame(index=future_dates, data={'Prediction': future_predictions})
            plot_df = pd.concat([df['Close'], predictions_df['Prediction']], axis=1)
            fig_pred = px.line(plot_df, title=f'{ticker_symbol} Price Prediction')
            fig_pred.update_traces(selector=dict(name="Prediction"), line=dict(dash='dot', color='orange'))
            st.plotly_chart(fig_pred, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# ==============================================================================
# --- PORTFOLIO OPTIMIZATION MODE ---
# ==============================================================================
elif app_mode == "Portfolio Optimization":
    st.header("Portfolio Optimization 📊")
    st.sidebar.header("Portfolio Input")
    tickers_string = st.sidebar.text_area("Enter Tickers (comma-separated)", "AAPL,GOOG,MSFT,AMZN").upper()
    start_date_po = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"), key="po_start")
    end_date_po = st.sidebar.date_input("End Date", pd.to_datetime("today"), key="po_end")

    tickers = [t.strip() for t in tickers_string.split(',') if t.strip()]

    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        try:
            # Fetch historical data first
            all_prices_data = yf.download(tickers, start=start_date_po, end=end_date_po)

            if all_prices_data.empty:
                st.error("Could not fetch valid data. Check tickers and date range.")
            else:
                prices = pd.DataFrame()
                # --- THIS IS THE NEW, ROBUST LOGIC ---
                if len(tickers) == 1:
                    # If single ticker, data is a DataFrame with flat columns
                    prices = all_prices_data[['Adj Close']]
                    prices.columns = tickers
                else:
                    # If multiple tickers, data has multi-level columns
                    prices = all_prices_data['Adj Close']
                
                # Drop any rows with missing values that can occur if stocks have different trading days
                prices.dropna(inplace=True)

                if prices.empty:
                    st.error("No overlapping trading data found for the selected tickers. Try a different date range.")
                else:
                    st.subheader("Historical Performance")
                    st.dataframe(prices.tail())
                    fig_prices = px.line(prices, title="Portfolio Component Prices")
                    st.plotly_chart(fig_prices, use_container_width=True)

                    # Calculate expected returns and sample covariance
                    mu = expected_returns.mean_historical_return(prices)
                    S = risk_models.sample_cov(prices)

                    # Optimize for max Sharpe ratio
                    ef = EfficientFrontier(mu, S)
                    weights = ef.max_sharpe()
                    cleaned_weights = ef.clean_weights()
                    
                    # Display results
                    st.subheader("Optimal Portfolio Allocation (Max Sharpe Ratio)")
                    
                    weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight'])
                    fig_pie_po = px.pie(weights_df, values='Weight', names=weights_df.index, title='Optimal Asset Allocation')
                    st.plotly_chart(fig_pie_po, use_container_width=True)

                    st.subheader("Expected Portfolio Performance")
                    expected_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)
                    st.markdown(f"**Expected Annual Return:** `{expected_return*100:.2f}%`")
                    st.markdown(f"**Annual Volatility:** `{annual_volatility*100:.2f}%`")
                    st.markdown(f"**Sharpe Ratio:** `{sharpe_ratio:.2f}`")

        except Exception as e:
            st.error(f"An error occurred during portfolio optimization: {e}")

