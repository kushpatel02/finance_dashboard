AI-Powered Quantitative Investment Strategy Dashboard
📖 Overview
This project is an all-in-one, interactive web application for investment analysis, built with Streamlit. It serves as a powerful decision-support tool for investors by integrating real-time data fetching, AI-powered price prediction, news sentiment analysis, and sophisticated portfolio optimization into a single, user-friendly dashboard.

This tool is designed to demonstrate a full-stack approach to quantitative finance, combining skills in data engineering, machine learning, natural language processing (NLP), and financial theory.

✨ Key Features
This dashboard is divided into two main tools:

1. Single Stock Analysis
Historical Data: Fetches and displays historical stock prices, company information, and technical indicators (like Simple Moving Averages).

🚀 AI Price Prediction: Trains an XGBoost model on the fly to forecast future stock prices for a user-defined number of days.

📰 News Sentiment Analysis: Pulls the latest news headlines for a stock, and uses a pre-trained FinBERT model to analyze the sentiment (Positive, Negative, Neutral) of each headline, providing a high-level overview of market perception.

2. Portfolio Optimization
Multi-Asset Analysis: Accepts a list of multiple stock tickers for analysis.

Modern Portfolio Theory (MPT): Calculates the optimal asset allocation to achieve the highest possible risk-adjusted return.

Efficient Frontier: Utilizes the PyPortfolioOpt library to find the portfolio that maximizes the Sharpe Ratio.

Clear Recommendations: Displays the recommended portfolio weights in an intuitive pie chart, along with key performance metrics like Expected Annual Return, Volatility, and Sharpe Ratio.

🛠️ Technologies Used
Frontend & Dashboard: Streamlit

Data Manipulation & Analysis: Pandas, NumPy

Data Fetching: yfinance, requests (for NewsAPI)

Machine Learning (Prediction): Scikit-learn, XGBoost

NLP (Sentiment Analysis): Hugging Face transformers, PyTorch

Financial Analysis (Optimization): PyPortfolioOpt

Data Visualization: Plotly Express

