
 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
 
# ---------- Config ----------
st.set_page_config(page_title="FIN 330 — Portfolio Dashboard", layout="wide")
 
DEFAULT_TICKERS = ["TSLA", "AMZN", "APLD", "GOOGL", "MSFT"]
DEFAULT_WEIGHTS = [0.20, 0.25, 0.15, 0.25, 0.15]
TRADING_DAYS = 252
RISK_FREE_ANNUAL = 0.045  # ~current 3-mo T-bill ballpark; user can override
 
 
# ---------- Data ----------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_close(tickers: list[str], period: str) -> pd.DataFrame:
    """Download adjusted close prices. Returns DataFrame indexed by date."""
    data = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if data.empty:
        return pd.DataFrame()
 
    # Normalize: yfinance returns MultiIndex for multiple tickers, flat for one
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
 
    # Reorder to requested ticker order, drop any that failed
    close = close[[t for t in tickers if t in close.columns]]
    return close.dropna(how="all")
 
 
# ---------- Part 1 helpers ----------
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
 
 
def classify_trend(price: float, ma20: float, ma50: float) -> str:
    if any(pd.isna(x) for x in (price, ma20, ma50)):
        return "Insufficient Data"
    if price > ma20 > ma50:
        return "Strong Uptrend"
    if price < ma20 < ma50:
        return "Strong Downtrend"
    return "Mixed Trend"
 
 
def classify_rsi(rsi: float) -> str:
    if pd.isna(rsi):
        return "Insufficient Data"
    if rsi > 70:
        return "Overbought (Possible Sell Signal)"
    if rsi < 30:
        return "Oversold (Possible Buy Signal)"
    return "Neutral"
 
 
def classify_vol(annualized_vol: float) -> str:
    if pd.isna(annualized_vol):
        return "Insufficient Data"
    if annualized_vol > 0.40:
        return "High"
    if annualized_vol >= 0.25:
        return "Medium"
    return "Low"
 
 
def make_recommendation(trend: str, rsi_label: str, vol_label: str) -> tuple[str, str]:
    """Simple rule-based recommendation. Returns (action, rationale)."""
    bullish = trend == "Strong Uptrend"
    bearish = trend == "Strong Downtrend"
    overbought = "Overbought" in rsi_label
    oversold = "Oversold" in rsi_label
 
    if bullish and oversold:
        return "Buy", "Uptrend with a temporary pullback (oversold RSI) — favorable entry."
    if bullish and not overbought:
        return "Buy / Hold", "Trend is up and momentum is not stretched."
    if bullish and overbought:
        return "Hold", "Trend is up but momentum is extended — risk of short-term pullback."
    if bearish and overbought:
        return "Sell", "Downtrend with a relief rally fading — momentum confirms weakness."
    if bearish:
        return "Sell / Avoid", "Price below both MAs; trend is unfavorable."
    if oversold:
        return "Watch / Speculative Buy", "Mixed trend but oversold — wait for confirmation."
    if overbought:
        return "Hold / Trim", "Mixed trend with stretched momentum."
    return "Hold", "No clear directional edge from trend or momentum."
 
 
# ---------- Part 2 helpers ----------
def portfolio_metrics(returns: pd.Series, rf_annual: float) -> dict:
    total_return = (1 + returns).prod() - 1
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    n_days = len(returns)
    ann_return = (1 + total_return) ** (TRADING_DAYS / n_days) - 1 if n_days else np.nan
    sharpe = (ann_return - rf_annual) / ann_vol if ann_vol and ann_vol > 0 else np.nan
    return {
        "Total Return": total_return,
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
    }
 
 
# ---------- UI ----------
st.title("FIN 330 — Stock & Portfolio Dashboard")
st.caption("Part 1: individual stock analysis. Part 2: portfolio vs. benchmark.")
 
with st.sidebar:
    st.header("Portfolio")
    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        value=",".join(DEFAULT_TICKERS),
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
 
    st.markdown("**Weights**")
    weights = []
    default_map = dict(zip(DEFAULT_TICKERS, DEFAULT_WEIGHTS))
    for t in tickers:
        w = st.number_input(
            f"{t}", min_value=0.0, max_value=1.0,
            value=float(default_map.get(t, round(1 / len(tickers), 2))),
            step=0.05, format="%.2f", key=f"w_{t}",
        )
        weights.append(w)
 
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-6:
        st.warning(f"Weights sum to {weight_sum:.2f} — they should sum to 1.00.")
 
    st.divider()
    st.header("Benchmark")
    benchmark = st.selectbox(
        "Benchmark ticker",
        options=["^GSPC", "SPY", "VOO", "IVV"],
        index=0,
        help="^GSPC is the S&P 500 index. SPY/VOO/IVV are ETFs that track it.",
    )
 
    st.divider()
    st.header("Settings")
    rf = st.number_input(
        "Risk-free rate (annual)", min_value=0.0, max_value=0.15,
        value=RISK_FREE_ANNUAL, step=0.005, format="%.3f",
    )
    part1_lookback = st.selectbox("Part 1 lookback", ["6mo", "1y"], index=0)
    part2_lookback = st.selectbox("Part 2 lookback", ["1y", "2y", "6mo"], index=0)
 
# ---------- Tabs ----------
tab1, tab2 = st.tabs(["Part 1 — Stock Analysis", "Part 2 — Portfolio vs. Benchmark"])
 
# =====================================================
# PART 1
# =====================================================
with tab1:
    st.subheader("Individual Stock Analysis")
 
    if not tickers:
        st.info("Add at least one ticker in the sidebar.")
        st.stop()
 
    selected = st.selectbox("Pick a stock to analyze", tickers, index=0)
 
    with st.spinner(f"Fetching {part1_lookback} of {selected}..."):
        prices = fetch_close([selected], part1_lookback)
 
    if prices.empty or selected not in prices.columns:
        st.error(f"Could not fetch data for {selected}.")
        st.stop()
 
    s = prices[selected].dropna()
    ma20 = s.rolling(20).mean()
    ma50 = s.rolling(50).mean()
    rsi = compute_rsi(s, 14)
    daily_ret = s.pct_change()
    ann_vol_20d = daily_ret.rolling(20).std().iloc[-1] * np.sqrt(TRADING_DAYS)
 
    current_price = s.iloc[-1]
    trend = classify_trend(current_price, ma20.iloc[-1], ma50.iloc[-1])
    rsi_label = classify_rsi(rsi.iloc[-1])
    vol_label = classify_vol(ann_vol_20d)
    action, rationale = make_recommendation(trend, rsi_label, vol_label)
 
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${current_price:,.2f}")
    c2.metric("20-day MA", f"${ma20.iloc[-1]:,.2f}" if pd.notna(ma20.iloc[-1]) else "—")
    c3.metric("50-day MA", f"${ma50.iloc[-1]:,.2f}" if pd.notna(ma50.iloc[-1]) else "—")
    c4.metric("14-day RSI", f"{rsi.iloc[-1]:.1f}" if pd.notna(rsi.iloc[-1]) else "—")
 
    c5, c6, c7 = st.columns(3)
    c5.metric("Trend", trend)
    c6.metric("Momentum", rsi_label.split(" (")[0])
    c7.metric("Volatility (20d ann.)",
              f"{ann_vol_20d*100:.1f}% ({vol_label})" if pd.notna(ann_vol_20d) else "—")
 
    st.success(f"**Recommendation: {action}** — {rationale}")
 
    # Price + MAs chart
    st.markdown("**Price with 20/50-day Moving Averages**")
    price_df = pd.DataFrame({
        "Price": s, "MA20": ma20, "MA50": ma50,
    })
    fig = px.line(price_df, labels={"value": "Price ($)", "index": "Date"})
    fig.update_layout(height=400, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
 
    # RSI chart
    st.markdown("**14-day RSI**")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#6366f1")))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red",
                      annotation_text="Overbought (70)")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green",
                      annotation_text="Oversold (30)")
    rsi_fig.update_layout(height=300, yaxis_range=[0, 100], yaxis_title="RSI")
    st.plotly_chart(rsi_fig, use_container_width=True)
 
    with st.expander("Show raw data"):
        st.dataframe(price_df.tail(30).round(2))
 
# =====================================================
# PART 2
# =====================================================
with tab2:
    st.subheader("Portfolio Performance vs. Benchmark")
 
    if abs(weight_sum - 1.0) > 1e-6:
        st.error(f"Weights must sum to 1.00 (currently {weight_sum:.2f}).")
        st.stop()
 
    all_tickers = list(dict.fromkeys(tickers + [benchmark]))  # preserve order, dedupe
 
    with st.spinner(f"Fetching {part2_lookback} of price data..."):
        px_df = fetch_close(all_tickers, part2_lookback)
 
    missing = [t for t in all_tickers if t not in px_df.columns]
    if missing:
        st.warning(f"No data returned for: {', '.join(missing)}")
 
    available_tickers = [t for t in tickers if t in px_df.columns]
    if not available_tickers or benchmark not in px_df.columns:
        st.error("Insufficient data to build portfolio or benchmark.")
        st.stop()
 
    # Re-normalize weights if any ticker dropped out
    w = np.array([weights[tickers.index(t)] for t in available_tickers], dtype=float)
    if not np.isclose(w.sum(), 1.0):
        w = w / w.sum()
        st.info("Weights re-normalized after dropping unavailable tickers.")
 
    # Returns
    stock_prices = px_df[available_tickers].dropna()
    bench_prices = px_df[benchmark].dropna()
    common_idx = stock_prices.index.intersection(bench_prices.index)
    stock_prices = stock_prices.loc[common_idx]
    bench_prices = bench_prices.loc[common_idx]
 
    stock_rets = stock_prices.pct_change().dropna()
    bench_rets = bench_prices.pct_change().dropna()
    portfolio_rets = stock_rets.dot(w)
 
    # Metrics
    p_metrics = portfolio_metrics(portfolio_rets, rf)
    b_metrics = portfolio_metrics(bench_rets, rf)
    outperf = p_metrics["Total Return"] - b_metrics["Total Return"]
 
    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Portfolio Total Return", f"{p_metrics['Total Return']*100:.2f}%")
    k2.metric(f"{benchmark} Total Return", f"{b_metrics['Total Return']*100:.2f}%")
    k3.metric("Outperformance", f"{outperf*100:+.2f}%")
    k4.metric("Portfolio Sharpe", f"{p_metrics['Sharpe Ratio']:.2f}")
 
    # Comparison table
    st.markdown("**Performance Metrics**")
    comp = pd.DataFrame({
        "Portfolio": p_metrics,
        f"{benchmark}": b_metrics,
    }).T
    fmt = {
        "Total Return": "{:.2%}".format,
        "Annualized Return": "{:.2%}".format,
        "Annualized Volatility": "{:.2%}".format,
        "Sharpe Ratio": "{:.2f}".format,
    }
    st.dataframe(comp.style.format(fmt))
 
    # Cumulative growth chart
    st.markdown(f"**Cumulative Growth of $1 — Portfolio vs. {benchmark}**")
    growth = pd.DataFrame({
        "Portfolio": (1 + portfolio_rets).cumprod(),
        benchmark: (1 + bench_rets).cumprod(),
    })
    fig2 = px.line(growth, labels={"value": "Growth of $1", "index": "Date"})
    fig2.update_layout(height=420, legend_title_text="")
    st.plotly_chart(fig2, use_container_width=True)
 
    # Per-stock contribution
    st.markdown("**Individual Stock Performance**")
    per_stock = pd.DataFrame({
        "Weight": w,
        "Total Return": [(1 + stock_rets[t]).prod() - 1 for t in available_tickers],
        "Ann. Volatility": [stock_rets[t].std() * np.sqrt(TRADING_DAYS)
                            for t in available_tickers],
    }, index=available_tickers)
    per_stock["Weighted Contribution"] = per_stock["Weight"] * per_stock["Total Return"]
    st.dataframe(per_stock.style.format({
        "Weight": "{:.0%}".format,
        "Total Return": "{:.2%}".format,
        "Ann. Volatility": "{:.2%}".format,
        "Weighted Contribution": "{:.2%}".format,
    }))
 
    # Allocation pie
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Allocation**")
        alloc_fig = px.pie(values=w, names=available_tickers, hole=0.45)
        alloc_fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(alloc_fig, use_container_width=True)
    with col_b:
        st.markdown("**Daily Return Distribution**")
        dist_df = pd.DataFrame({
            "Portfolio": portfolio_rets,
            benchmark: bench_rets,
        }).melt(var_name="Series", value_name="Daily Return")
        dist_fig = px.histogram(dist_df, x="Daily Return", color="Series",
                                barmode="overlay", nbins=50, opacity=0.6)
        dist_fig.update_layout(height=350)
        st.plotly_chart(dist_fig, use_container_width=True)
 
    # Interpretation block
    st.markdown("### Interpretation")
    perf_word = "outperformed" if outperf > 0 else "underperformed"
    risk_word = (
        "more risky" if p_metrics["Annualized Volatility"] > b_metrics["Annualized Volatility"]
        else "less risky"
    )
    eff_word = (
        "more efficient" if p_metrics["Sharpe Ratio"] > b_metrics["Sharpe Ratio"]
        else "less efficient"
    )
    st.write(
        f"Over the {part2_lookback} window, the portfolio **{perf_word}** "
        f"{benchmark} by **{abs(outperf)*100:.2f}%** in total return. "
        f"Annualized volatility was **{p_metrics['Annualized Volatility']*100:.2f}%** "
        f"vs. **{b_metrics['Annualized Volatility']*100:.2f}%** for the benchmark — "
        f"i.e., the portfolio was **{risk_word}**. "
        f"On a risk-adjusted basis (Sharpe), the portfolio was **{eff_word}** "
        f"({p_metrics['Sharpe Ratio']:.2f} vs. {b_metrics['Sharpe Ratio']:.2f}). "
       
    )
 
st.caption("Data: Yahoo Finance via yfinance. Educational use only — not investment advice.")
