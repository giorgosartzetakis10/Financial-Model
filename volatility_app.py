import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Volatility Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #e9ecef;
        margin-bottom: 8px;
    }
    .metric-label { font-size: 12px; color: #6c757d; margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: 600; color: #212529; }
    .regime-low    { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .regime-medium { background:#fef3c7; color:#92400e; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .regime-high   { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .driver-box {
        background:#fffbeb; border-left:3px solid #f59e0b;
        border-radius:0 8px 8px 0; padding:10px 14px; margin-top:8px;
    }
    .driver-title { font-size:12px; font-weight:600; color:#92400e; margin-bottom:6px; }
    .driver-row { display:flex; justify-content:space-between; font-size:12px; margin-bottom:3px; }
    .interp-box {
        background:#f1f5f9; border-radius:8px;
        padding:12px 14px; font-size:13px; color:#475569;
        line-height:1.6; margin-top:10px;
    }
    .macro-banner {
        background:#f8fafc; border:1px solid #e2e8f0;
        border-radius:10px; padding:12px 16px; margin-bottom:1rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; }
    h1 { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Macro-geopolitical context (March 2026) ────────────────────────────────────
MACRO_CONTEXT = {
    "Fed rate": "3.5–3.75% (on hold)",
    "CPI": "~2.7% (above 2% target)",
    "GDP growth": "~3.4% annualized",
    "Fed chair": "Transition due May 2026",
    "Iran–US conflict": "Military ops active (Feb–Mar 2026)",
    "Strait of Hormuz": "Energy corridor at risk",
    "Russia–Ukraine": "Year 4, no ceasefire",
    "US tariffs": "Supreme Court ruling pending",
    "S&P 500": "~5% below recent peak",
}

# Sector → macro-geo premium (annualized vol %)
SECTOR_PREMIUMS = {
    "Energy":      {"premium": 7.5, "drivers": [
        ("Iran/Hormuz energy risk", "+5.0%"),
        ("Russia–Ukraine supply disruption", "+1.5%"),
        ("Tariff-driven demand uncertainty", "+1.0%"),
    ]},
    "Financial":   {"premium": 3.0, "drivers": [
        ("Fed chair transition uncertainty", "+1.5%"),
        ("Yield curve volatility", "+1.0%"),
        ("Credit spread widening", "+0.5%"),
    ]},
    "Technology":  {"premium": 3.5, "drivers": [
        ("US–China chip export controls", "+1.5%"),
        ("AI valuation concerns", "+1.0%"),
        ("Tariff hardware exposure", "+1.0%"),
    ]},
    "Consumer":    {"premium": 2.5, "drivers": [
        ("Tariff pass-through inflation", "+1.2%"),
        ("Consumer confidence declining", "+0.8%"),
        ("Fed rate sensitivity", "+0.5%"),
    ]},
    "Healthcare":  {"premium": 2.0, "drivers": [
        ("Policy/regulatory uncertainty", "+1.0%"),
        ("Elevated market vol environment", "+1.0%"),
    ]},
    "Industrial":  {"premium": 2.5, "drivers": [
        ("Tariff supply chain disruption", "+1.5%"),
        ("Global trade fragmentation", "+1.0%"),
    ]},
    "Defense":     {"premium": 0.5, "drivers": [
        ("European defense spending surge (positive)", "-1.5%"),
        ("Elevated geopolitical risk environment", "+2.0%"),
    ]},
    "Unknown":     {"premium": 2.5, "drivers": [
        ("Elevated geopolitical risk (Iran, Ukraine)", "+1.5%"),
        ("Fed policy uncertainty", "+1.0%"),
    ]},
}

# Ticker → sector mapping (common names; fallback to info lookup)
KNOWN_SECTORS = {
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "JPM": "Financial", "BAC": "Financial", "GS": "Financial", "MS": "Financial",
    "WFC": "Financial", "C": "Financial", "BRK-B": "Financial",
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "GOOG": "Technology", "META": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "INTC": "Technology", "TSLA": "Technology",
    "AMZN": "Technology", "NFLX": "Technology",
    "WMT": "Consumer", "COST": "Consumer", "TGT": "Consumer",
    "MCD": "Consumer", "KO": "Consumer", "PEP": "Consumer",
    "JNJ": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare",
    "UNH": "Healthcare", "ABBV": "Healthcare",
    "CAT": "Industrial", "DE": "Industrial", "GE": "Industrial",
    "LMT": "Defense", "RTX": "Defense", "NOC": "Defense", "BA": "Defense",
    "SPY": "Unknown", "QQQ": "Unknown", "IWM": "Unknown",
}


# ── Core model functions ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker: str, years: int = 2) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df = df[["Close"]].copy()
    df.columns = ["close"]
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df.dropna(inplace=True)
    return df


def compute_realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualized rolling realized volatility."""
    return returns.rolling(window).std() * np.sqrt(252) * 100


def build_har_features(rv_daily: pd.Series) -> pd.DataFrame:
    """Build HAR predictors: RV_d, RV_w (5d avg), RV_m (21d avg)."""
    df = pd.DataFrame({"RV_d": rv_daily})
    df["RV_w"] = rv_daily.rolling(5).mean()
    df["RV_m"] = rv_daily.rolling(21).mean()
    return df


def fit_har(rv_daily: pd.Series, horizon: int = 21) -> dict:
    """
    Fit HAR-RV and forecast h-step ahead RV.
    Returns model coefficients, R², and point forecast.
    """
    feats = build_har_features(rv_daily)
    # target: future RV averaged over [t+1, t+horizon]
    target = rv_daily.rolling(horizon).mean().shift(-horizon)

    df = pd.concat([feats, target.rename("target")], axis=1).dropna()
    X = df[["RV_d", "RV_w", "RV_m"]].values
    y = df["target"].values

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Forecast using latest available features
    last = feats.dropna().iloc[-1]
    x_new = np.array([[last["RV_d"], last["RV_w"], last["RV_m"]]])
    forecast = float(model.predict(x_new)[0])

    # Bootstrap CI (500 samples)
    residuals = y - y_pred
    boots = []
    rng = np.random.default_rng(42)
    for _ in range(500):
        idx = rng.integers(0, len(residuals), len(residuals))
        y_boot = y_pred + residuals[idx]
        m_boot = LinearRegression().fit(X, y_boot)
        boots.append(float(m_boot.predict(x_new)[0]))
    boots = np.array(boots)

    return {
        "coef_daily":   round(float(model.coef_[0]), 4),
        "coef_weekly":  round(float(model.coef_[1]), 4),
        "coef_monthly": round(float(model.coef_[2]), 4),
        "intercept":    round(float(model.intercept_), 4),
        "r_squared":    round(r2, 4),
        "forecast":     round(max(forecast, 1.0), 2),
        "ci_80_low":    round(max(float(np.percentile(boots, 10)), 1.0), 2),
        "ci_80_high":   round(float(np.percentile(boots, 90)), 2),
        "ci_95_low":    round(max(float(np.percentile(boots, 2.5)), 1.0), 2),
        "ci_95_high":   round(float(np.percentile(boots, 97.5)), 2),
    }


def get_sector(ticker: str) -> str:
    if ticker.upper() in KNOWN_SECTORS:
        return KNOWN_SECTORS[ticker.upper()]
    try:
        info = yf.Ticker(ticker).info
        raw = info.get("sector", "Unknown")
        for key, val in SECTOR_PREMIUMS.items():
            if key.lower() in raw.lower():
                return key
    except Exception:
        pass
    return "Unknown"


def classify_regime(vol: float) -> str:
    if vol < 15:
        return "Low"
    elif vol < 30:
        return "Medium"
    return "High"


def vol_trend(rv_series: pd.Series, lookback: int = 21) -> str:
    recent = rv_series.dropna().iloc[-lookback:]
    if len(recent) < 5:
        return "stable"
    slope = np.polyfit(range(len(recent)), recent.values, 1)[0]
    if slope > 0.05:
        return "rising"
    elif slope < -0.05:
        return "falling"
    return "stable"


def analyze_ticker(ticker: str, horizon: int) -> dict:
    df = fetch_data(ticker)
    rv_daily = compute_realized_vol(df["log_ret"], window=1) * np.sqrt(252 / 252)
    # Use proper daily RV (std of single day annualized)
    rv_1d = df["log_ret"].abs() * np.sqrt(252) * 100
    rv_rolling = compute_realized_vol(df["log_ret"], window=21)

    # Realized vol metrics
    rv_1m  = round(float(rv_rolling.dropna().iloc[-21:].mean()), 2)
    rv_3m  = round(float(rv_rolling.dropna().iloc[-63:].mean()), 2)
    ytd_start = datetime(datetime.today().year, 1, 1)
    ytd_mask = df.index >= pd.Timestamp(ytd_start)
    rv_ytd = round(float(rv_rolling[ytd_mask].dropna().mean()) if ytd_mask.sum() > 0 else rv_3m, 2)

    # Monthly realized vols (past 12 months)
    monthly_vols = []
    rv_vals = rv_rolling.dropna()
    for i in range(11, -1, -1):
        end_idx = len(rv_vals) - i * 21
        start_idx = end_idx - 21
        if start_idx >= 0 and end_idx > start_idx:
            monthly_vols.append(round(float(rv_vals.iloc[start_idx:end_idx].mean()), 2))
        else:
            monthly_vols.append(rv_1m)

    # HAR model
    har = fit_har(rv_1d.dropna(), horizon=horizon)

    # Macro-geo adjustment
    sector = get_sector(ticker)
    macro = SECTOR_PREMIUMS.get(sector, SECTOR_PREMIUMS["Unknown"])
    premium = macro["premium"]
    adj_forecast = round(har["forecast"] + premium, 2)
    adj_ci80_low  = round(har["ci_80_low"]  + premium * 0.7, 2)
    adj_ci80_high = round(har["ci_80_high"] + premium * 1.3, 2)
    adj_ci95_low  = round(har["ci_95_low"]  + premium * 0.5, 2)
    adj_ci95_high = round(har["ci_95_high"] + premium * 1.5, 2)

    current_price = round(float(df["close"].iloc[-1]), 2)
    daily_vol = round(float(rv_1d.dropna().iloc[-1]), 2)
    regime = classify_regime(rv_1m)
    trend = vol_trend(rv_rolling)

    return {
        "symbol":              ticker.upper(),
        "sector":              sector,
        "current_price":       current_price,
        "current_daily_vol":   daily_vol,
        "realized_vol_1m":     rv_1m,
        "realized_vol_3m":     rv_3m,
        "realized_vol_ytd":    rv_ytd,
        "har_base_forecast":   har["forecast"],
        "geo_macro_premium":   premium,
        "har_adjusted_forecast": adj_forecast,
        "ci_80_low":  adj_ci80_low,
        "ci_80_high": adj_ci80_high,
        "ci_95_low":  adj_ci95_low,
        "ci_95_high": adj_ci95_high,
        "coef_daily":   har["coef_daily"],
        "coef_weekly":  har["coef_weekly"],
        "coef_monthly": har["coef_monthly"],
        "r_squared":    har["r_squared"],
        "regime":       regime,
        "vol_trend":    trend,
        "macro_drivers": macro["drivers"],
        "monthly_vols":  monthly_vols,
        "rv_series":     rv_rolling,
        "price_series":  df["close"],
    }


# ── Plotly chart ───────────────────────────────────────────────────────────────
def make_chart(result: dict, hLabel: str) -> go.Figure:
    months = [f"{i}m ago" if i > 0 else "now" for i in range(12, 0, -1)] + [f"{hLabel} fcast"]
    hist_y = result["monthly_vols"]
    base_y = [None] * 12 + [result["har_base_forecast"]]
    adj_y  = [None] * 12 + [result["har_adjusted_forecast"]]
    ci80h  = [None] * 12 + [result["ci_80_high"]]
    ci80l  = [None] * 12 + [result["ci_80_low"]]
    ci95h  = [None] * 12 + [result["ci_95_high"]]
    ci95l  = [None] * 12 + [result["ci_95_low"]]

    fig = go.Figure()

    # 95% CI band
    fig.add_trace(go.Scatter(
        x=months, y=ci95h, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=months, y=ci95l, mode="lines", fill="tonexty",
        fillcolor="rgba(239,159,39,0.08)", line=dict(width=0),
        name="95% CI", hoverinfo="skip"
    ))
    # 80% CI band
    fig.add_trace(go.Scatter(
        x=months, y=ci80h, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=months, y=ci80l, mode="lines", fill="tonexty",
        fillcolor="rgba(239,159,39,0.15)", line=dict(width=0),
        name="80% CI", hoverinfo="skip"
    ))
    # Historical realized vol
    fig.add_trace(go.Scatter(
        x=months[:12], y=hist_y, mode="lines+markers",
        name="Realized vol (monthly)",
        line=dict(color="#378ADD", width=2.5),
        marker=dict(size=5),
    ))
    # HAR base forecast point
    fig.add_trace(go.Scatter(
        x=[months[-1]], y=[result["har_base_forecast"]],
        mode="markers", name="HAR base",
        marker=dict(color="#888780", size=9, symbol="circle"),
    ))
    # Adjusted forecast point
    fig.add_trace(go.Scatter(
        x=[months[-1]], y=[result["har_adjusted_forecast"]],
        mode="markers", name="Adjusted forecast",
        marker=dict(color="#EF9F27", size=13, symbol="triangle-up"),
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(
            gridcolor="#f0f0f0",
            ticksuffix="%",
            tickfont=dict(size=11),
            title="Annualized volatility (%)",
            titlefont=dict(size=11),
        ),
        hovermode="x unified",
    )
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📈 Volatility forecaster — HAR + macro-geopolitical model")

# Macro banner
with st.expander("🌍 Active macro & geopolitical context (March 2026)", expanded=False):
    cols = st.columns(3)
    items = list(MACRO_CONTEXT.items())
    for i, (k, v) in enumerate(items):
        with cols[i % 3]:
            st.markdown(f"**{k}**: {v}")

# Sidebar controls
with st.sidebar:
    st.header("Model settings")
    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, JPM, XOM",
        help="Up to 8 tickers. E.g. AAPL, MSFT, NVDA"
    )
    horizon_label = st.selectbox(
        "Forecast horizon",
        ["1 month (21 days)", "2 months (42 days)", "3 months (63 days)"],
        index=1,
    )
    horizon_map = {"1 month (21 days)": 21, "2 months (42 days)": 42, "3 months (63 days)": 63}
    horizon_days = horizon_map[horizon_label]
    h_label = horizon_label.split(" ")[0] + "-month"

    st.markdown("---")
    st.caption("**Model:** HAR-RV (Corsi, 2009) + macro-geopolitical risk premium layer. Data: Yahoo Finance via yfinance. CIs: 500-sample bootstrap.")
    st.caption("⚠️ Not financial advice.")

    run = st.button("▶ Run forecast", type="primary", use_container_width=True)

# Main content
if run:
    raw_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()][:8]
    if not raw_tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    results = {}
    progress = st.progress(0, text="Starting...")
    total = len(raw_tickers)

    for i, ticker in enumerate(raw_tickers):
        progress.progress((i) / total, text=f"Fetching data for {ticker}...")
        try:
            results[ticker] = analyze_ticker(ticker, horizon_days)
        except Exception as e:
            st.warning(f"⚠️ Could not fetch {ticker}: {e}")
        progress.progress((i + 1) / total, text=f"Done {ticker}")

    progress.empty()

    if not results:
        st.error("No data could be fetched. Check your ticker symbols and internet connection.")
        st.stop()

    # Summary table
    st.subheader("Summary")
    summary_rows = []
    for sym, r in results.items():
        trend_icon = "↑" if r["vol_trend"] == "rising" else "↓" if r["vol_trend"] == "falling" else "→"
        summary_rows.append({
            "Ticker":          sym,
            "Sector":          r["sector"],
            "Price":           f"${r['current_price']:.2f}",
            "1m Realized Vol": f"{r['realized_vol_1m']:.1f}%",
            "3m Realized Vol": f"{r['realized_vol_3m']:.1f}%",
            "HAR Base":        f"{r['har_base_forecast']:.1f}%",
            "Geo Premium":     f"+{r['geo_macro_premium']:.1f}%",
            "Adj. Forecast":   f"{r['har_adjusted_forecast']:.1f}%",
            "80% CI":          f"{r['ci_80_low']:.1f}–{r['ci_80_high']:.1f}%",
            "Regime":          r["regime"],
            "Trend":           trend_icon + " " + r["vol_trend"],
            "R²":              f"{r['r_squared']:.3f}",
        })
    st.dataframe(pd.DataFrame(summary_rows).set_index("Ticker"), use_container_width=True)

    # Per-stock detail tabs
    st.subheader("Detailed results")
    tabs = st.tabs([r["symbol"] for r in results.values()])

    for tab, (sym, r) in zip(tabs, results.items()):
        with tab:
            # Header row
            regime_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            col_h1, col_h2 = st.columns([3, 1])
            with col_h1:
                st.markdown(f"### {sym} &nbsp; <span style='font-size:14px;color:#6c757d'>{r['sector']} · ${r['current_price']:.2f}</span>", unsafe_allow_html=True)
            with col_h2:
                st.markdown(f"{regime_color.get(r['regime'], '⚪')} **{r['regime']} volatility regime**")

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("1m Realized Vol",  f"{r['realized_vol_1m']:.1f}%")
            m2.metric("3m Realized Vol",  f"{r['realized_vol_3m']:.1f}%")
            m3.metric("YTD Realized Vol", f"{r['realized_vol_ytd']:.1f}%")
            trend_delta = f"{r['vol_trend']}"
            m4.metric("Vol trend", trend_delta)

            # Chart
            st.plotly_chart(make_chart(r, h_label), use_container_width=True, config={"displayModeBar": False})

            # Forecast breakdown
            st.markdown(f"#### {h_label} forecast breakdown")
            f1, f2, f3 = st.columns(3)
            f1.metric("HAR base forecast",      f"{r['har_base_forecast']:.1f}%",  help="Pure statistical HAR-RV forecast")
            f2.metric("Macro-geo premium",       f"+{r['geo_macro_premium']:.1f}%", help="Risk premium based on current geopolitical environment")
            diff = r["har_adjusted_forecast"] - r["realized_vol_1m"]
            f3.metric("Adjusted forecast",       f"{r['har_adjusted_forecast']:.1f}%", delta=f"{diff:+.1f}% vs current 1m vol")

            c1, c2, c3 = st.columns(3)
            c1.metric("80% CI", f"{r['ci_80_low']:.1f}–{r['ci_80_high']:.1f}%")
            c2.metric("95% CI", f"{r['ci_95_low']:.1f}–{r['ci_95_high']:.1f}%")
            c3.metric("Model R²", f"{r['r_squared']:.3f}", help="In-sample fit of the HAR regression")

            # HAR coefficients
            with st.expander("HAR model coefficients"):
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Daily coef (β_d)",   f"{r['coef_daily']:.4f}")
                cc2.metric("Weekly coef (β_w)",  f"{r['coef_weekly']:.4f}")
                cc3.metric("Monthly coef (β_m)", f"{r['coef_monthly']:.4f}")
                st.caption("HAR-RV model: RV(h) = β₀ + β_d·RV_d + β_w·RV_w + β_m·RV_m + ε")

            # Macro drivers
            st.markdown("**Macro & geopolitical risk drivers**")
            driver_rows = "".join(
                f'<div class="driver-row"><span style="color:#6c757d">{d[0]}</span>'
                f'<span style="font-weight:600;color:{"#dc2626" if "+" in d[1] else "#16a34a"}">{d[1]}</span></div>'
                for d in r["macro_drivers"]
            )
            st.markdown(
                f'<div class="driver-box"><div class="driver-title">Active risk factors for {sym} ({r["sector"]} sector)</div>{driver_rows}</div>',
                unsafe_allow_html=True
            )

            # Price chart
            with st.expander("Price history (2 years)"):
                price_fig = go.Figure()
                price_fig.add_trace(go.Scatter(
                    x=r["price_series"].index,
                    y=r["price_series"].values,
                    mode="lines", name="Close price",
                    line=dict(color="#378ADD", width=1.5),
                ))
                price_fig.update_layout(
                    height=220, margin=dict(l=40, r=20, t=10, b=30),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(gridcolor="#f0f0f0", tickprefix="$"),
                )
                st.plotly_chart(price_fig, use_container_width=True, config={"displayModeBar": False})

else:
    st.info("👈 Configure your tickers and horizon in the sidebar, then click **Run forecast**.")
    st.markdown("""
    **What this model does:**
    - Downloads 2 years of daily price data from Yahoo Finance
    - Fits a **HAR-RV model** (Corsi, 2009) — the industry standard for multi-horizon volatility forecasting
    - Adds a **macro-geopolitical risk premium** based on current conditions (Iran–US conflict, Russia–Ukraine, Fed uncertainty, tariff risks)
    - Outputs point forecasts with **80% and 95% bootstrap confidence intervals**
    - Classifies each stock's volatility regime and trend
    """)
