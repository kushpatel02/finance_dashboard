# AI-Powered Quantitative Investment Strategy Dashboard

An interactive, all-in-one web application for sophisticated investment analysis combining real-time market data, AI-powered price predictions, news sentiment analysis, and portfolio optimization. Built with Streamlit, this dashboard serves as a comprehensive decision-support tool for investors, demonstrating a full-stack approach to quantitative finance.

## Overview
This project bridges the gap between traditional financial analysis and modern machine learning techniques, providing investors with actionable insights through an intuitive web interface. The dashboard integrates multiple analytical approaches into a cohesive platform that supports both individual stock analysis and multi-asset portfolio construction.

## Key Capabilities:
- Real-time stock data analysis with technical indicators
- Machine learning-based price forecasting using XGBoost
- NLP-powered sentiment analysis on financial news
- Modern Portfolio Theory-based portfolio optimization
- Interactive visualizations for data-driven decision making

## Features:
### 1. Single Stock Analysis
#### 📊 Historical Data & Technical Analysis

- Fetch real-time and historical stock prices using yfinance API
- Display comprehensive company information (sector, industry, market cap)
- Calculate and visualize Simple Moving Averages (SMA) for trend analysis
- Interactive time-series charts with customizable date ranges

#### 🤖 AI-Powered Price Prediction

- On-the-fly XGBoost model training on historical price data
- User-defined forecast horizons (1-365 days)
- Feature engineering with technical indicators and lag features
- Visual comparison of predicted vs. historical prices
- Confidence metrics and model performance statistics

#### 📰 News Sentiment Analysis

- Real-time financial news aggregation via NewsAPI
- FinBERT-based sentiment classification (Positive, Negative, Neutral)
- Sentiment scoring for each headline
- Aggregated sentiment overview for market perception
- Source attribution and publication timestamps

### 2. Portfolio Optimization
#### 💼 Multi-Asset Portfolio Construction

- Simultaneous analysis of multiple stock tickers
- Historical return and volatility calculations
- Correlation matrix for diversification insights
- Risk-return profile visualization

#### 📈 Modern Portfolio Theory (MPT) Implementation

- Efficient Frontier calculation using PyPortfolioOpt
- Maximum Sharpe Ratio optimization
- Optimal asset allocation recommendations
- Risk-adjusted return metrics

#### 📊 Portfolio Performance Metrics

- Expected Annual Return
- Portfolio Volatility (Standard Deviation)
- Sharpe Ratio (risk-adjusted performance)
- Individual asset weight allocations
- Interactive pie charts for visual weight distribution


## Technology Stack
#### Frontend & User Interface
- Streamlit: Interactive web application framework with real-time updates

#### Data Handling & Analysis
- Pandas: Data manipulation and time-series analysis
- NumPy: Numerical computations and array operations
- yfinance: Real-time and historical financial data retrieval

#### Machine Learning & AI
- XGBoost: Gradient boosting for price prediction
- Scikit-learn: Model evaluation and preprocessing
- Hugging Face Transformers: Pre-trained FinBERT for sentiment analysis
- PyTorch: Deep learning backend for NLP models

#### Financial Analysis
- PyPortfolioOpt: Portfolio optimization and efficient frontier calculation
- Custom implementations of Modern Portfolio Theory

#### Visualization
- Plotly Express: Interactive, publication-quality charts and graphs

#### External APIs
- NewsAPI: Financial news headline aggregation
