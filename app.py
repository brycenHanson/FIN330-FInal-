import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Finance Dashboard", layout="wide")

st.title("📊 Stock & Portfolio Analytics Dashboard")

# -----------------------------
# Helper Functions
# -----------------------------
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_trend(row):
    if row['Close'] > row['20MA'] > row['50MA']:
        return "Strong Uptrend 📈"
    elif row['Close'] < row['20MA'] < row['50MA']:
        return "Strong Downtrend 📉"
    else:
        return "Mixed Trend ⚖️"

def get_recommendation(trend, rsi):
    if trend == "Strong Uptrend 📈" and rsi < 70:
        return "BUY ✅"
    elif trend == "Strong Downtrend 📉" and rsi > 30:
        return "SELL ❌"
    else:
        return "HOLD ⚖️"

# -----------------------------
# TABS
# -----------------------------
tab1, tab2 = st.tabs(["📈 Stock Analysis", "📁 Portfolio Dashboard"])

# =============================
# TAB 1: STOCK ANALYSIS
# =============================
with tab1:
    st.header("Individual Stock Analysis")

    ticker = st.text_input("Enter Stock Ticker", "AAPL")

    if ticker:
        try:
            data = yf.download(ticker, period="6mo")

            if data.empty:
                st.error("No data found.")
            else:
                data['20MA'] = data['Close'].rolling(20).mean()
                data['50MA'] = data['Close'].rolling(50).mean()
                data['RSI'] = calculate_rsi(data)

                latest = data.iloc[-1]

                trend = get_trend(latest)

                rsi_value = latest['RSI']
                if rsi_value > 70:
                    rsi_signal = "Overbought 🔴"
                elif rsi_value < 30:
                    rsi_signal = "Oversold 🟢"
                else:
                    rsi_signal = "Neutral ⚖️"

                returns = data['Close'].pct_change()
                volatility = returns.std() * np.sqrt(252)

                if volatility > 0.40:
                    vol_label = "High"
                elif volatility > 0.25:
                    vol_label = "Medium"
                else:
                    vol_label = "Low"

                recommendation = get_recommendation(trend, rsi_value)

                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${latest['Close']:.2f}")
                col2.metric("Trend", trend)
                col3.metric("RSI", f"{rsi_value:.2f} ({rsi_signal})")
                col4.metric("Volatility", f"{volatility:.2%} ({vol_label})")

                st.subheader("Recommendation")
                st.success(recommendation)

                # Chart
                st.subheader("Price & Moving Averages")
                fig, ax = plt.subplots()
                ax.plot(data['Close'], label="Close")
                ax.plot(data['20MA'], label="20MA")
                ax.plot(data['50MA'], label="50MA")
                ax.legend()
                st.pyplot(fig)

        except:
            st.error("Error loading data.")

# =============================
# TAB 2: PORTFOLIO
# =============================
with tab2:
    st.header("Portfolio Performance Dashboard")

    st.write("Enter 5 stocks and weights (must sum to 1.0)")

    tickers = []
    weights = []

    cols = st.columns(5)

    default_stocks = ["AAPL","MSFT","GOOGL","AMZN","TSLA"]

    for i in range(5):
        with cols[i]:
            t = st.text_input(f"Stock {i+1}", default_stocks[i])
            w = st.number_input(f"Weight {i+1}", 0.0, 1.0, 0.2)
            tickers.append(t)
            weights.append(w)

    if abs(sum(weights) - 1.0) > 0.01:
        st.warning("⚠️ Weights must sum to 1.0")
    else:
        try:
            prices = yf.download(tickers, period="1y")['Close']
            benchmark = yf.download("^GSPC", period="1y")['Close']

            returns = prices.pct_change().dropna()
            portfolio_returns = returns.dot(weights)

            benchmark_returns = benchmark.pct_change().dropna()

            # Align dates (important fix)
            portfolio_returns, benchmark_returns = portfolio_returns.align(
                benchmark_returns, join='inner'
            )

            total_return = (1 + portfolio_returns).prod() - 1
            benchmark_return = (1 + benchmark_returns).prod() - 1

            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Portfolio Return", f"{total_return:.2%}")
            col2.metric("Benchmark (S&P 500)", f"{benchmark_return:.2%}")
            col3.metric("Volatility", f"{volatility:.2%}")
            col4.metric("Sharpe Ratio", f"{sharpe:.2f}")

            # Performance result
            if total_return > benchmark_return:
                st.success("Portfolio OUTPERFORMED the S&P 500 ✅")
            else:
                st.error("Portfolio UNDERPERFORMED the S&P 500 ❌")

            # Chart
            st.subheader("Portfolio vs S&P 500 (^GSPC)")
            cumulative_portfolio = (1 + portfolio_returns).cumprod()
            cumulative_benchmark = (1 + benchmark_returns).cumprod()

            fig, ax = plt.subplots()
            ax.plot(cumulative_portfolio, label="Portfolio")
            ax.plot(cumulative_benchmark, label="S&P 500 (^GSPC)")
            ax.legend()
            st.pyplot(fig)

        except:
            st.error("Error loading portfolio data.")
