import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Volatility & Returns Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .driver-box { background:#fffbeb; border-left:3px solid #f59e0b; border-radius:0 8px 8px 0; padding:10px 14px; margin-top:8px; }
    .driver-title { font-size:12px; font-weight:600; color:#92400e; margin-bottom:6px; }
    .driver-row { display:flex; justify-content:space-between; font-size:12px; margin-bottom:3px; }
    .rs-box { background:#f0f9ff; border-left:3px solid #0ea5e9; border-radius:0 8px 8px 0; padding:10px 14px; margin-top:8px; }
    .rs-title { font-size:12px; font-weight:600; color:#0369a1; margin-bottom:6px; }
    .ret-box { background:#f0fdf4; border-left:3px solid #22c55e; border-radius:0 8px 8px 0; padding:10px 14px; margin-top:8px; }
    .ret-title { font-size:12px; font-weight:600; color:#166534; margin-bottom:6px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    h1 { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MACRO_CONTEXT = {
    "Fed rate": "3.5-3.75% (on hold)",
    "CPI": "~2.7% (above 2% target)",
    "GDP growth": "~3.4% annualized",
    "Fed chair": "Transition due May 2026",
    "Iran-US conflict": "Military ops active (Feb-Mar 2026)",
    "Strait of Hormuz": "Energy corridor at risk",
    "Russia-Ukraine": "Year 4, no ceasefire",
    "US tariffs": "Supreme Court ruling pending",
    "S&P 500": "~5% below recent peak",
}

SECTOR_PREMIUMS = {
    "Energy":     {"premium": 7.5, "drivers": [("Iran/Hormuz energy risk", "+5.0%"), ("Russia-Ukraine supply disruption", "+1.5%"), ("Tariff-driven demand uncertainty", "+1.0%")]},
    "Financial":  {"premium": 3.0, "drivers": [("Fed chair transition uncertainty", "+1.5%"), ("Yield curve volatility", "+1.0%"), ("Credit spread widening", "+0.5%")]},
    "Technology": {"premium": 3.5, "drivers": [("US-China chip export controls", "+1.5%"), ("AI valuation concerns", "+1.0%"), ("Tariff hardware exposure", "+1.0%")]},
    "Consumer":   {"premium": 2.5, "drivers": [("Tariff pass-through inflation", "+1.2%"), ("Consumer confidence declining", "+0.8%"), ("Fed rate sensitivity", "+0.5%")]},
    "Healthcare": {"premium": 2.0, "drivers": [("Policy/regulatory uncertainty", "+1.0%"), ("Elevated market vol environment", "+1.0%")]},
    "Industrial": {"premium": 2.5, "drivers": [("Tariff supply chain disruption", "+1.5%"), ("Global trade fragmentation", "+1.0%")]},
    "Defense":    {"premium": 0.5, "drivers": [("European defense spending surge (positive)", "-1.5%"), ("Elevated geopolitical risk environment", "+2.0%")]},
    "Unknown":    {"premium": 2.5, "drivers": [("Elevated geopolitical risk (Iran, Ukraine)", "+1.5%"), ("Fed policy uncertainty", "+1.0%")]},
}

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

COLORS = ["#378ADD", "#E24B4A", "#1D9E75", "#EF9F27", "#7F77DD", "#D85A30", "#D4537E", "#639922"]


# ── Data ───────────────────────────────────────────────────────────────────────
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


# ── Returns calculations ───────────────────────────────────────────────────────
def compute_returns(df: pd.DataFrame) -> dict:
    prices = df["close"]
    log_ret = df["log_ret"]
    today = prices.index[-1]

    def pct(t0, t1):
        try:
            p0 = float(prices.asof(t0)) if t0 >= prices.index[0] else None
            p1 = float(prices.iloc[-1])
            return round((p1 / p0 - 1) * 100, 2) if p0 else None
        except Exception:
            return None

    # Period returns
    ret_1d  = round(float((prices.iloc[-1] / prices.iloc[-2] - 1) * 100), 2) if len(prices) >= 2 else None
    ret_1w  = pct(today - timedelta(days=7),   today)
    ret_1m  = pct(today - timedelta(days=30),  today)
    ret_3m  = pct(today - timedelta(days=91),  today)
    ret_6m  = pct(today - timedelta(days=182), today)
    ret_1y  = pct(today - timedelta(days=365), today)
    ytd_start = pd.Timestamp(datetime(today.year, 1, 1))
    ret_ytd = pct(ytd_start, today)

    # Annualised return (CAGR over available history)
    n_years = (prices.index[-1] - prices.index[0]).days / 365.25
    cagr = round((float(prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1) * 100, 2) if n_years > 0 else None

    # Best / worst single day
    best_day  = round(float(log_ret.apply(lambda x: (np.exp(x) - 1) * 100).max()), 2)
    worst_day = round(float(log_ret.apply(lambda x: (np.exp(x) - 1) * 100).min()), 2)

    # Win rate (% of days with positive return)
    win_rate = round(float((log_ret > 0).mean() * 100), 1)

    # Cumulative return series (rebased to 100)
    cumret = (1 + log_ret).cumprod() * 100
    cumret.name = "cumret"

    # Monthly returns heatmap data
    monthly_ret = log_ret.resample("ME").sum().apply(lambda x: round((np.exp(x) - 1) * 100, 2))
    monthly_df  = monthly_ret.to_frame("ret")
    monthly_df["year"]  = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month

    return {
        "ret_1d":   ret_1d,
        "ret_1w":   ret_1w,
        "ret_1m":   ret_1m,
        "ret_3m":   ret_3m,
        "ret_6m":   ret_6m,
        "ret_1y":   ret_1y,
        "ret_ytd":  ret_ytd,
        "cagr":     cagr,
        "best_day": best_day,
        "worst_day": worst_day,
        "win_rate": win_rate,
        "cumret":   cumret,
        "monthly_df": monthly_df,
    }


# ── Standard HAR ───────────────────────────────────────────────────────────────
def fit_har(rv_1d: pd.Series, horizon: int = 21) -> dict:
    feats = pd.DataFrame({"RV_d": rv_1d})
    feats["RV_w"] = rv_1d.rolling(5).mean()
    feats["RV_m"] = rv_1d.rolling(21).mean()
    target = rv_1d.rolling(horizon).mean().shift(-horizon)
    df = pd.concat([feats, target.rename("y")], axis=1).dropna()
    X, y = df[["RV_d", "RV_w", "RV_m"]].values, df["y"].values
    m = LinearRegression().fit(X, y)
    y_pred = m.predict(X)
    r2 = max(0.0, 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))
    last = feats.dropna().iloc[-1]
    x_new = np.array([[last["RV_d"], last["RV_w"], last["RV_m"]]])
    forecast = float(m.predict(x_new)[0])
    rng = np.random.default_rng(42)
    residuals = y - y_pred
    boots = [float(LinearRegression().fit(X, y_pred + residuals[rng.integers(0, len(residuals), len(residuals))]).predict(x_new)[0]) for _ in range(500)]
    boots = np.array(boots)
    return {
        "forecast":     round(max(forecast, 1.0), 2),
        "r_squared":    round(r2, 4),
        "coef_daily":   round(float(m.coef_[0]), 4),
        "coef_weekly":  round(float(m.coef_[1]), 4),
        "coef_monthly": round(float(m.coef_[2]), 4),
        "ci_80_low":    round(max(float(np.percentile(boots, 10)), 1.0), 2),
        "ci_80_high":   round(float(np.percentile(boots, 90)), 2),
        "ci_95_low":    round(max(float(np.percentile(boots, 2.5)), 1.0), 2),
        "ci_95_high":   round(float(np.percentile(boots, 97.5)), 2),
    }


# ── Regime-switching HAR ───────────────────────────────────────────────────────
def fit_regime_switching_har(rv_1d: pd.Series, horizon: int = 21) -> dict:
    feats = pd.DataFrame({"RV_d": rv_1d})
    feats["RV_w"] = rv_1d.rolling(5).mean()
    feats["RV_m"] = rv_1d.rolling(21).mean()
    target = rv_1d.rolling(horizon).mean().shift(-horizon)
    df = pd.concat([feats, target.rename("y")], axis=1).dropna()
    threshold = float(np.percentile(df["RV_d"].values, 75))
    df["regime"] = np.where(df["RV_d"] > threshold, "shock", "calm")
    regime_results = {}
    for name in ["calm", "shock"]:
        subset = df[df["regime"] == name]
        if len(subset) < 20:
            subset = df
        X = subset[["RV_d", "RV_w", "RV_m"]].values
        y = subset["y"].values
        m = LinearRegression().fit(X, y)
        y_pred = m.predict(X)
        r2 = max(0.0, 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))
        regime_results[name] = {
            "model": m, "X": X, "y": y, "y_pred": y_pred,
            "r2": round(r2, 4),
            "coef_d": round(float(m.coef_[0]), 4),
            "coef_w": round(float(m.coef_[1]), 4),
            "coef_m": round(float(m.coef_[2]), 4),
            "n_obs": len(subset),
        }
    last = feats.dropna().iloc[-1]
    current_regime = "shock" if float(last["RV_d"]) > threshold else "calm"
    active = regime_results[current_regime]
    x_new = np.array([[last["RV_d"], last["RV_w"], last["RV_m"]]])
    forecast = float(active["model"].predict(x_new)[0])
    rng = np.random.default_rng(42)
    residuals = active["y"] - active["y_pred"]
    boots = [float(LinearRegression().fit(active["X"], active["y_pred"] + residuals[rng.integers(0, len(residuals), len(residuals))]).predict(x_new)[0]) for _ in range(500)]
    boots = np.array(boots)
    std_r2 = fit_har(rv_1d, horizon)["r_squared"]
    return {
        "forecast":       round(max(forecast, 1.0), 2),
        "current_regime": current_regime,
        "threshold":      round(threshold, 2),
        "r2_calm":        regime_results["calm"]["r2"],
        "r2_shock":       regime_results["shock"]["r2"],
        "r2_active":      active["r2"],
        "r2_improvement": round(active["r2"] - std_r2, 4),
        "har_base_r2":    std_r2,
        "coef_d_calm":    regime_results["calm"]["coef_d"],
        "coef_w_calm":    regime_results["calm"]["coef_w"],
        "coef_m_calm":    regime_results["calm"]["coef_m"],
        "coef_d_shock":   regime_results["shock"]["coef_d"],
        "coef_w_shock":   regime_results["shock"]["coef_w"],
        "coef_m_shock":   regime_results["shock"]["coef_m"],
        "n_calm":         regime_results["calm"]["n_obs"],
        "n_shock":        regime_results["shock"]["n_obs"],
        "ci_80_low":      round(max(float(np.percentile(boots, 10)), 1.0), 2),
        "ci_80_high":     round(float(np.percentile(boots, 90)), 2),
        "ci_95_low":      round(max(float(np.percentile(boots, 2.5)), 1.0), 2),
        "ci_95_high":     round(float(np.percentile(boots, 97.5)), 2),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252) * 100

def get_sector(ticker: str) -> str:
    if ticker.upper() in KNOWN_SECTORS:
        return KNOWN_SECTORS[ticker.upper()]
    try:
        raw = yf.Ticker(ticker).info.get("sector", "Unknown")
        for key in SECTOR_PREMIUMS:
            if key.lower() in raw.lower():
                return key
    except Exception:
        pass
    return "Unknown"

def classify_regime(vol: float) -> str:
    return "Low" if vol < 15 else "High" if vol > 30 else "Medium"

def vol_trend(rv_series: pd.Series, lookback: int = 21) -> str:
    recent = rv_series.dropna().iloc[-lookback:]
    if len(recent) < 5:
        return "stable"
    slope = np.polyfit(range(len(recent)), recent.values, 1)[0]
    return "rising" if slope > 0.05 else "falling" if slope < -0.05 else "stable"

def fmt_ret(v):
    if v is None:
        return "n/a"
    color = "#16a34a" if v >= 0 else "#dc2626"
    sign  = "+" if v >= 0 else ""
    return f'<span style="color:{color};font-weight:600">{sign}{v:.2f}%</span>'


# ── Main analysis ──────────────────────────────────────────────────────────────
def analyze_ticker(ticker: str, horizon: int, use_rs: bool = True) -> dict:
    df = fetch_data(ticker)
    rv_1d      = df["log_ret"].abs() * np.sqrt(252) * 100
    rv_rolling = compute_realized_vol(df["log_ret"], window=21)

    rv_1m  = round(float(rv_rolling.dropna().iloc[-21:].mean()), 2)
    rv_3m  = round(float(rv_rolling.dropna().iloc[-63:].mean()), 2)
    ytd_mask = df.index >= pd.Timestamp(datetime(datetime.today().year, 1, 1))
    rv_ytd = round(float(rv_rolling[ytd_mask].dropna().mean()) if ytd_mask.sum() > 5 else rv_3m, 2)

    monthly_vols = []
    rv_vals = rv_rolling.dropna()
    for i in range(11, -1, -1):
        e = len(rv_vals) - i * 21
        s = e - 21
        monthly_vols.append(round(float(rv_vals.iloc[max(s, 0):e].mean()), 2) if e > s >= 0 else rv_1m)

    har = fit_har(rv_1d.dropna(), horizon)
    rs  = fit_regime_switching_har(rv_1d.dropna(), horizon) if use_rs else None

    active_forecast = rs["forecast"]   if (use_rs and rs) else har["forecast"]
    active_r2       = rs["r2_active"]  if (use_rs and rs) else har["r_squared"]
    active_ci80l    = rs["ci_80_low"]  if (use_rs and rs) else har["ci_80_low"]
    active_ci80h    = rs["ci_80_high"] if (use_rs and rs) else har["ci_80_high"]
    active_ci95l    = rs["ci_95_low"]  if (use_rs and rs) else har["ci_95_low"]
    active_ci95h    = rs["ci_95_high"] if (use_rs and rs) else har["ci_95_high"]

    sector  = get_sector(ticker)
    macro   = SECTOR_PREMIUMS.get(sector, SECTOR_PREMIUMS["Unknown"])
    premium = macro["premium"]

    returns = compute_returns(df)

    return {
        "symbol":                ticker.upper(),
        "sector":                sector,
        "current_price":         round(float(df["close"].iloc[-1]), 2),
        "current_daily_vol":     round(float(rv_1d.dropna().iloc[-1]), 2),
        "realized_vol_1m":       rv_1m,
        "realized_vol_3m":       rv_3m,
        "realized_vol_ytd":      rv_ytd,
        "har_base_forecast":     har["forecast"],
        "har_r_squared":         har["r_squared"],
        "har_coef_daily":        har["coef_daily"],
        "har_coef_weekly":       har["coef_weekly"],
        "har_coef_monthly":      har["coef_monthly"],
        "rs":                    rs,
        "rs_enabled":            use_rs and rs is not None,
        "active_forecast":       active_forecast,
        "active_r2":             active_r2,
        "geo_macro_premium":     premium,
        "har_adjusted_forecast": round(active_forecast + premium, 2),
        "ci_80_low":             round(active_ci80l + premium * 0.7, 2),
        "ci_80_high":            round(active_ci80h + premium * 1.3, 2),
        "ci_95_low":             round(active_ci95l + premium * 0.5, 2),
        "ci_95_high":            round(active_ci95h + premium * 1.5, 2),
        "regime":                classify_regime(rv_1m),
        "vol_trend":             vol_trend(rv_rolling),
        "macro_drivers":         macro["drivers"],
        "monthly_vols":          monthly_vols,
        "rv_series":             rv_rolling,
        "price_series":          df["close"],
        "returns":               returns,
    }


# ── Charts ─────────────────────────────────────────────────────────────────────
def make_forecast_chart(r: dict, h_label: str) -> go.Figure:
    months = [f"{i}m ago" for i in range(12, 0, -1)] + [f"{h_label} fcast"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=[None]*12+[r["ci_95_high"]], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=months, y=[None]*12+[r["ci_95_low"]],  fill="tonexty", fillcolor="rgba(239,159,39,0.08)", line=dict(width=0), name="95% CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=months, y=[None]*12+[r["ci_80_high"]], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=months, y=[None]*12+[r["ci_80_low"]],  fill="tonexty", fillcolor="rgba(239,159,39,0.15)", line=dict(width=0), name="80% CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=months[:12], y=r["monthly_vols"], mode="lines+markers", name="Realized vol",
                             line=dict(color="#378ADD", width=2.5), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=[months[-1]], y=[r["har_base_forecast"]], mode="markers", name="Standard HAR",
                             marker=dict(color="#888780", size=9, symbol="circle")))
    if r["rs_enabled"]:
        fig.add_trace(go.Scatter(x=[months[-1]], y=[r["rs"]["forecast"]], mode="markers", name="RS-HAR",
                                 marker=dict(color="#0ea5e9", size=10, symbol="diamond")))
    fig.add_trace(go.Scatter(x=[months[-1]], y=[r["har_adjusted_forecast"]], mode="markers", name="Adj. forecast",
                             marker=dict(color="#EF9F27", size=13, symbol="triangle-up")))
    fig.update_layout(
        height=300, margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", y=1.02, x=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#f0f0f0", ticksuffix="%", tickfont=dict(size=11),
                   title=dict(text="Annualized vol (%)", font=dict(size=11))),
        hovermode="x unified",
    )
    return fig


def make_regime_coef_chart(r: dict) -> go.Figure:
    rs = r["rs"]
    cats = ["beta daily", "beta weekly", "beta monthly"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=f"Calm (n={rs['n_calm']})", x=cats,
                         y=[rs["coef_d_calm"], rs["coef_w_calm"], rs["coef_m_calm"]],
                         marker_color="#378ADD", opacity=0.85))
    fig.add_trace(go.Bar(name=f"Shock (n={rs['n_shock']})", x=cats,
                         y=[rs["coef_d_shock"], rs["coef_w_shock"], rs["coef_m_shock"]],
                         marker_color="#E24B4A", opacity=0.85))
    fig.update_layout(
        height=240, barmode="group", margin=dict(l=40, r=20, t=10, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.02, x=0),
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(gridcolor="#f0f0f0", tickfont=dict(size=11),
                   title=dict(text="Coefficient value", font=dict(size=11))),
    )
    return fig


def make_cumret_chart(r: dict) -> go.Figure:
    cumret = r["returns"]["cumret"]
    fig = go.Figure()
    fig.add_hline(y=100, line=dict(color="#d1d5db", width=1, dash="dot"))
    color = "#1D9E75" if float(cumret.iloc[-1]) >= 100 else "#E24B4A"
    fig.add_trace(go.Scatter(
        x=cumret.index, y=cumret.values,
        mode="lines", name="Cumulative return",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({'29,158,117' if color=='#1D9E75' else '226,75,74'},0.08)",
    ))
    fig.update_layout(
        height=240, margin=dict(l=40, r=20, t=10, b=30),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#f0f0f0", tickfont=dict(size=11),
                   title=dict(text="Rebased (100 = start)", font=dict(size=11))),
        hovermode="x unified", showlegend=False,
    )
    return fig


def make_monthly_heatmap(r: dict) -> go.Figure:
    mdf = r["returns"]["monthly_df"].copy()
    years  = sorted(mdf["year"].unique())
    months = list(range(1, 13))
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    z = []
    text = []
    for yr in years:
        row_z, row_t = [], []
        for mo in months:
            val = mdf[(mdf["year"] == yr) & (mdf["month"] == mo)]["ret"]
            if len(val) > 0:
                v = float(val.iloc[0])
                row_z.append(v)
                row_t.append(f"{v:+.1f}%")
            else:
                row_z.append(None)
                row_t.append("")
        z.append(row_z)
        text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z, x=month_names, y=[str(y) for y in years],
        text=text, texttemplate="%{text}",
        colorscale=[[0,"#fee2e2"],[0.5,"#f3f4f6"],[1,"#d1fae5"]],
        zmid=0,
        showscale=False,
        textfont=dict(size=11),
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{text}<extra></extra>",
    ))
    fig.update_layout(
        height=max(160, len(years) * 36 + 60),
        margin=dict(l=50, r=20, t=10, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickfont=dict(size=11), side="bottom"),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
    )
    return fig


def make_risk_reward_scatter(results: dict, h_label: str) -> go.Figure:
    fig = go.Figure()

    # Quadrant shading
    all_vols = [r["har_adjusted_forecast"] for r in results.values()]
    all_rets = [r["returns"]["ret_1y"] or 0 for r in results.values()]
    mid_vol  = float(np.median(all_vols)) if all_vols else 25
    mid_ret  = float(np.median(all_rets)) if all_rets else 0

    # Quadrant labels
    for x, y, label, color in [
        (mid_vol * 0.5, mid_ret + abs(mid_ret) * 0.5 + 5, "High return / low risk", "#16a34a"),
        (mid_vol * 1.5, mid_ret + abs(mid_ret) * 0.5 + 5, "High return / high risk", "#ca8a04"),
        (mid_vol * 0.5, mid_ret - abs(mid_ret) * 0.5 - 5, "Low return / low risk",  "#6b7280"),
        (mid_vol * 1.5, mid_ret - abs(mid_ret) * 0.5 - 5, "Low return / high risk", "#dc2626"),
    ]:
        fig.add_annotation(x=x, y=y, text=label, showarrow=False,
                           font=dict(size=10, color=color), opacity=0.5)

    # Quadrant lines
    fig.add_hline(y=mid_ret, line=dict(color="#e5e7eb", width=1, dash="dot"))
    fig.add_vline(x=mid_vol, line=dict(color="#e5e7eb", width=1, dash="dot"))

    for i, (sym, r) in enumerate(results.items()):
        ret_1y = r["returns"]["ret_1y"]
        if ret_1y is None:
            continue
        adj_vol = r["har_adjusted_forecast"]
        color   = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=[adj_vol], y=[ret_1y],
            mode="markers+text",
            name=sym,
            text=[sym],
            textposition="top center",
            textfont=dict(size=11, color=color),
            marker=dict(size=14, color=color, opacity=0.85,
                        line=dict(width=1.5, color="white")),
            hovertemplate=(
                f"<b>{sym}</b><br>"
                f"1Y return: {ret_1y:+.1f}%<br>"
                f"{h_label} vol forecast: {adj_vol:.1f}%<br>"
                f"Sector: {r['sector']}<extra></extra>"
            ),
        ))

    fig.update_layout(
        height=400, margin=dict(l=60, r=20, t=20, b=60),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            gridcolor="#f0f0f0", ticksuffix="%", tickfont=dict(size=11),
            title=dict(text=f"Adjusted vol forecast ({h_label}, annualized)", font=dict(size=11)),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#f0f0f0", ticksuffix="%", tickfont=dict(size=11),
            title=dict(text="1-year return (%)", font=dict(size=11)),
            zeroline=True, zerolinecolor="#e5e7eb",
        ),
        hovermode="closest",
    )
    return fig


def make_returns_bar(r: dict) -> go.Figure:
    labels = ["1 day", "1 week", "1 month", "3 months", "6 months", "YTD", "1 year"]
    ret    = r["returns"]
    vals   = [ret["ret_1d"], ret["ret_1w"], ret["ret_1m"], ret["ret_3m"],
              ret["ret_6m"], ret["ret_ytd"], ret["ret_1y"]]
    colors = ["#1D9E75" if (v or 0) >= 0 else "#E24B4A" for v in vals]
    y_vals = [v if v is not None else 0 for v in vals]

    fig = go.Figure(go.Bar(
        x=labels, y=y_vals,
        marker_color=colors, opacity=0.85,
        text=[f"{v:+.2f}%" if v is not None else "n/a" for v in vals],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        height=260, margin=dict(l=40, r=20, t=10, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#f0f0f0", ticksuffix="%", tickfont=dict(size=11),
                   zeroline=True, zerolinecolor="#d1d5db"),
        showlegend=False,
    )
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("Volatility & returns analyser — HAR-RS + macro + returns")

with st.expander("Active macro & geopolitical context (March 2026)", expanded=False):
    cols = st.columns(3)
    for i, (k, v) in enumerate(MACRO_CONTEXT.items()):
        cols[i % 3].markdown(f"**{k}**: {v}")

with st.sidebar:
    st.header("Settings")
    tickers_input = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, GOOGL, JPM, XOM, NVDA")
    horizon_label = st.selectbox("Forecast horizon",
                                  ["1 month (21 days)", "2 months (42 days)", "3 months (63 days)"], index=1)
    horizon_days = {"1 month (21 days)": 21, "2 months (42 days)": 42, "3 months (63 days)": 63}[horizon_label]
    h_label      = horizon_label.split(" ")[0] + "-month"
    st.markdown("---")
    use_rs = st.toggle("Regime-switching HAR", value=True,
                        help="Fits separate HAR models for calm and shock vol regimes.")
    st.markdown("---")
    st.caption("Models: HAR-RV (Corsi 2009) + HAR-RS (regime-switching). "
               "Macro-geo premium applied on top. Data: Yahoo Finance. CIs: 500-sample bootstrap.")
    st.caption("Not financial advice.")
    run = st.button("Run analysis", type="primary", width="stretch")

if run:
    raw_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()][:8]
    if not raw_tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    results = {}
    progress = st.progress(0, text="Starting...")
    for i, ticker in enumerate(raw_tickers):
        progress.progress(i / len(raw_tickers), text=f"Analyzing {ticker}...")
        try:
            results[ticker] = analyze_ticker(ticker, horizon_days, use_rs=use_rs)
        except Exception as e:
            st.warning(f"Could not fetch {ticker}: {e}")
        progress.progress((i + 1) / len(raw_tickers))
    progress.empty()

    if not results:
        st.error("No data could be fetched.")
        st.stop()

    # ── Top-level tabs ─────────────────────────────────────────────────────────
    top_tab1, top_tab2, top_tab3 = st.tabs(["📊 Volatility forecast", "📈 Returns analysis", "⚖️ Risk / reward"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — VOLATILITY FORECAST (unchanged from previous version)
    # ══════════════════════════════════════════════════════════════════════════
    with top_tab1:
        st.subheader("Volatility summary")
        rows = []
        for sym, r in results.items():
            ti = "up" if r["vol_trend"] == "rising" else "down" if r["vol_trend"] == "falling" else "stable"
            row = {
                "Ticker":   sym, "Sector": r["sector"], "Price": f"${r['current_price']:.2f}",
                "1m Vol":   f"{r['realized_vol_1m']:.1f}%", "3m Vol": f"{r['realized_vol_3m']:.1f}%",
                "HAR base": f"{r['har_base_forecast']:.1f}%", "HAR R2": f"{r['har_r_squared']:.3f}",
            }
            if use_rs:
                rs = r["rs"]
                row["RS-HAR R2"] = f"{rs['r2_active']:.3f}"
                row["R2 impr."]  = f"{'+' if rs['r2_improvement']>=0 else ''}{rs['r2_improvement']:.3f}"
                row["Regime"]    = rs["current_regime"].upper()
            row.update({
                "Geo premium":   f"+{r['geo_macro_premium']:.1f}%",
                "Adj. forecast": f"{r['har_adjusted_forecast']:.1f}%",
                "80% CI":        f"{r['ci_80_low']:.1f}-{r['ci_80_high']:.1f}%",
                "Vol regime":    r["regime"], "Trend": ti,
            })
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), width="stretch")

        st.subheader("Detailed volatility")
        vtabs = st.tabs(list(results.keys()))
        for tab, (sym, r) in zip(vtabs, results.items()):
            with tab:
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"### {sym} &nbsp; <span style='font-size:14px;color:#6c757d'>{r['sector']} · ${r['current_price']:.2f}</span>", unsafe_allow_html=True)
                c2.markdown(f"**{r['regime']} vol regime**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("1m Realized Vol", f"{r['realized_vol_1m']:.1f}%")
                m2.metric("3m Realized Vol", f"{r['realized_vol_3m']:.1f}%")
                m3.metric("YTD Realized Vol", f"{r['realized_vol_ytd']:.1f}%")
                m4.metric("Vol trend", r["vol_trend"])
                st.plotly_chart(make_forecast_chart(r, h_label), width="stretch", config={"displayModeBar": False})
                st.markdown(f"#### {h_label} forecast breakdown")
                f1, f2, f3 = st.columns(3)
                f1.metric("HAR base forecast", f"{r['har_base_forecast']:.1f}%", help=f"Standard HAR-RV R2 = {r['har_r_squared']:.3f}")
                f2.metric("Macro-geo premium", f"+{r['geo_macro_premium']:.1f}%")
                diff = r["har_adjusted_forecast"] - r["realized_vol_1m"]
                f3.metric("Adjusted forecast", f"{r['har_adjusted_forecast']:.1f}%", delta=f"{diff:+.1f}% vs current 1m vol")
                c1, c2, c3 = st.columns(3)
                c1.metric("80% CI", f"{r['ci_80_low']:.1f}-{r['ci_80_high']:.1f}%")
                c2.metric("95% CI", f"{r['ci_95_low']:.1f}-{r['ci_95_high']:.1f}%")
                c3.metric("Active model R2", f"{r['active_r2']:.3f}")
                if r["rs_enabled"]:
                    rs = r["rs"]
                    st.markdown("---")
                    st.markdown("#### Regime-switching HAR (HAR-RS)")
                    rs1, rs2, rs3, rs4 = st.columns(4)
                    rs1.metric("Current regime", rs["current_regime"].upper(), help=f"Shock threshold: {rs['threshold']:.1f}%")
                    rs2.metric("RS-HAR forecast", f"{rs['forecast']:.1f}%", delta=f"{rs['forecast']-r['har_base_forecast']:+.1f}% vs standard HAR")
                    rs3.metric("R2 improvement", f"{'+' if rs['r2_improvement']>=0 else ''}{rs['r2_improvement']:.3f}", delta=f"from {rs['har_base_r2']:.3f} to {rs['r2_active']:.3f}")
                    rs4.metric("Observations", f"{rs['n_calm']} calm / {rs['n_shock']} shock")
                    st.plotly_chart(make_regime_coef_chart(r), width="stretch", config={"displayModeBar": False})
                    if rs["current_regime"] == "shock":
                        interp = (f"{sym} is in a shock regime (daily vol > {rs['threshold']:.1f}%). Beta_daily ({rs['coef_d_shock']:.3f}) dominates — short-term vol persistence is stronger. RS-HAR improves R2 from {rs['har_base_r2']:.3f} to {rs['r2_shock']:.3f}.")
                    else:
                        interp = (f"{sym} is in a calm regime (daily vol <= {rs['threshold']:.1f}%). Beta_monthly ({rs['coef_m_calm']:.3f}) dominates — long-horizon persistence leads. RS-HAR improves R2 from {rs['har_base_r2']:.3f} to {rs['r2_calm']:.3f}.")
                    st.markdown(f'<div class="rs-box"><div class="rs-title">Regime interpretation</div><div style="font-size:13px;color:#0c4a6e;line-height:1.6">{interp}</div></div>', unsafe_allow_html=True)
                    with st.expander("Full regime coefficient table"):
                        coef_df = pd.DataFrame({
                            "Coefficient": ["beta_daily","beta_weekly","beta_monthly","R2","Observations"],
                            "Calm":  [rs["coef_d_calm"], rs["coef_w_calm"], rs["coef_m_calm"], rs["r2_calm"], rs["n_calm"]],
                            "Shock": [rs["coef_d_shock"],rs["coef_w_shock"],rs["coef_m_shock"],rs["r2_shock"],rs["n_shock"]],
                        }).set_index("Coefficient")
                        st.dataframe(coef_df, width="stretch")
                with st.expander("Standard HAR coefficients"):
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("beta_daily",   f"{r['har_coef_daily']:.4f}")
                    cc2.metric("beta_weekly",  f"{r['har_coef_weekly']:.4f}")
                    cc3.metric("beta_monthly", f"{r['har_coef_monthly']:.4f}")
                    st.caption(f"RV(h) = b0 + bd*RV_d + bw*RV_w + bm*RV_m  |  R2 = {r['har_r_squared']:.3f}")
                st.markdown("**Macro & geopolitical risk drivers**")
                driver_rows = "".join(
                    f'<div class="driver-row"><span style="color:#6c757d">{d[0]}</span>'
                    f'<span style="font-weight:600;color:{"#dc2626" if "+" in d[1] else "#16a34a"}">{d[1]}</span></div>'
                    for d in r["macro_drivers"]
                )
                st.markdown(f'<div class="driver-box"><div class="driver-title">Active risk factors - {sym} ({r["sector"]})</div>{driver_rows}</div>', unsafe_allow_html=True)
                with st.expander("Price history (2 years)"):
                    pf = go.Figure()
                    pf.add_trace(go.Scatter(x=r["price_series"].index, y=r["price_series"].values, mode="lines", line=dict(color="#378ADD", width=1.5)))
                    pf.update_layout(height=220, margin=dict(l=40,r=20,t=10,b=30), plot_bgcolor="white", paper_bgcolor="white", xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0", tickprefix="$"))
                    st.plotly_chart(pf, width="stretch", config={"displayModeBar": False})

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — RETURNS ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with top_tab2:
        st.subheader("Returns summary")

        # Cross-ticker returns table
        ret_rows = []
        for sym, r in results.items():
            ret = r["returns"]
            ret_rows.append({
                "Ticker":   sym,
                "Sector":   r["sector"],
                "Price":    f"${r['current_price']:.2f}",
                "1 day":    f"{ret['ret_1d']:+.2f}%" if ret["ret_1d"] is not None else "n/a",
                "1 week":   f"{ret['ret_1w']:+.2f}%" if ret["ret_1w"] is not None else "n/a",
                "1 month":  f"{ret['ret_1m']:+.2f}%" if ret["ret_1m"] is not None else "n/a",
                "3 months": f"{ret['ret_3m']:+.2f}%" if ret["ret_3m"] is not None else "n/a",
                "6 months": f"{ret['ret_6m']:+.2f}%" if ret["ret_6m"] is not None else "n/a",
                "YTD":      f"{ret['ret_ytd']:+.2f}%" if ret["ret_ytd"] is not None else "n/a",
                "1 year":   f"{ret['ret_1y']:+.2f}%" if ret["ret_1y"] is not None else "n/a",
                "CAGR (2y)": f"{ret['cagr']:+.2f}%" if ret["cagr"] is not None else "n/a",
                "Win rate": f"{ret['win_rate']:.1f}%",
                "Best day": f"+{ret['best_day']:.2f}%",
                "Worst day": f"{ret['worst_day']:.2f}%",
            })
        st.dataframe(pd.DataFrame(ret_rows).set_index("Ticker"), width="stretch")

        # Cumulative returns comparison chart
        st.subheader("Cumulative returns — all tickers (rebased to 100)")
        fig_cum = go.Figure()
        fig_cum.add_hline(y=100, line=dict(color="#d1d5db", width=1, dash="dot"))
        for i, (sym, r) in enumerate(results.items()):
            cumret = r["returns"]["cumret"]
            color  = COLORS[i % len(COLORS)]
            fig_cum.add_trace(go.Scatter(
                x=cumret.index, y=cumret.values,
                mode="lines", name=sym,
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{sym}</b><br>%{{x|%b %d %Y}}<br>Value: %{{y:.1f}}<extra></extra>",
            ))
        fig_cum.update_layout(
            height=320, margin=dict(l=50, r=20, t=10, b=40),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.02, x=0),
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(gridcolor="#f0f0f0", tickfont=dict(size=11),
                       title=dict(text="Rebased to 100", font=dict(size=11))),
            hovermode="x unified",
        )
        st.plotly_chart(fig_cum, width="stretch", config={"displayModeBar": False})

        # Per-stock detail
        st.subheader("Per-stock returns detail")
        rtabs = st.tabs(list(results.keys()))
        for tab, (sym, r) in zip(rtabs, results.items()):
            with tab:
                ret = r["returns"]
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"### {sym} &nbsp; <span style='font-size:14px;color:#6c757d'>{r['sector']} · ${r['current_price']:.2f}</span>", unsafe_allow_html=True)

                # Key return metrics
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("1-day return",   f"{ret['ret_1d']:+.2f}%" if ret["ret_1d"] is not None else "n/a",
                          delta=None)
                k2.metric("YTD return",     f"{ret['ret_ytd']:+.2f}%" if ret["ret_ytd"] is not None else "n/a")
                k3.metric("1-year return",  f"{ret['ret_1y']:+.2f}%"  if ret["ret_1y"]  is not None else "n/a")
                k4.metric("2yr CAGR",       f"{ret['cagr']:+.2f}%"    if ret["cagr"]    is not None else "n/a")

                k5, k6, k7, k8 = st.columns(4)
                k5.metric("1-week return",  f"{ret['ret_1w']:+.2f}%" if ret["ret_1w"] is not None else "n/a")
                k6.metric("1-month return", f"{ret['ret_1m']:+.2f}%" if ret["ret_1m"] is not None else "n/a")
                k7.metric("Best single day",  f"+{ret['best_day']:.2f}%")
                k8.metric("Worst single day", f"{ret['worst_day']:.2f}%")

                # Returns bar chart
                st.markdown("**Returns across horizons**")
                st.plotly_chart(make_returns_bar(r), width="stretch", config={"displayModeBar": False})

                # Cumulative return for this stock
                st.markdown("**Cumulative return (2 years)**")
                st.plotly_chart(make_cumret_chart(r), width="stretch", config={"displayModeBar": False})

                # Monthly returns heatmap
                st.markdown("**Monthly returns heatmap**")
                st.plotly_chart(make_monthly_heatmap(r), width="stretch", config={"displayModeBar": False})
                st.caption("Green = positive month, red = negative month. Intensity reflects magnitude.")

                # Return summary box
                lines = (
                    f"Over the past year {sym} returned <b>{ret['ret_1y']:+.2f}%</b> " if ret["ret_1y"] is not None else ""
                    f"with a 2-year annualised CAGR of <b>{ret['cagr']:+.2f}%</b>. " if ret["cagr"] is not None else ""
                    f"The stock was up on <b>{ret['win_rate']:.1f}%</b> of trading days, "
                    f"with a best single-day gain of <b>+{ret['best_day']:.2f}%</b> and worst single-day loss of <b>{ret['worst_day']:.2f}%</b>. "
                    f"Current {h_label} vol forecast is <b>{r['har_adjusted_forecast']:.1f}%</b> (macro-adjusted)."
                )
                st.markdown(f'<div class="ret-box"><div class="ret-title">Returns summary — {sym}</div><div style="font-size:13px;color:#14532d;line-height:1.7">{lines}</div></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — RISK / REWARD SCATTER
    # ══════════════════════════════════════════════════════════════════════════
    with top_tab3:
        st.subheader("Risk / reward positioning")
        st.caption(f"X axis: {h_label} adjusted vol forecast (higher = more risk). Y axis: 1-year historical return. Quadrants reveal the risk/reward profile of each stock.")

        st.plotly_chart(make_risk_reward_scatter(results, h_label), width="stretch", config={"displayModeBar": False})

        # Interpretation table
        st.subheader("Risk / reward summary")
        rr_rows = []
        for sym, r in results.items():
            ret_1y   = r["returns"]["ret_1y"]
            adj_vol  = r["har_adjusted_forecast"]
            if ret_1y is None:
                quadrant = "n/a"
            else:
                high_ret = ret_1y >= 0
                high_vol = adj_vol >= r["har_adjusted_forecast"]
                all_vols_list = [x["har_adjusted_forecast"] for x in results.values()]
                median_vol = float(np.median(all_vols_list))
                all_rets_list = [x["returns"]["ret_1y"] or 0 for x in results.values()]
                median_ret = float(np.median(all_rets_list))
                if ret_1y >= median_ret and adj_vol <= median_vol:
                    quadrant = "High return / low risk"
                elif ret_1y >= median_ret and adj_vol > median_vol:
                    quadrant = "High return / high risk"
                elif ret_1y < median_ret and adj_vol <= median_vol:
                    quadrant = "Low return / low risk"
                else:
                    quadrant = "Low return / high risk"
            rr_rows.append({
                "Ticker":           sym,
                "Sector":           r["sector"],
                "1Y return":        f"{ret_1y:+.2f}%" if ret_1y is not None else "n/a",
                "Vol forecast":     f"{adj_vol:.1f}%",
                "Win rate":         f"{r['returns']['win_rate']:.1f}%",
                "Best day":         f"+{r['returns']['best_day']:.2f}%",
                "Worst day":        f"{r['returns']['worst_day']:.2f}%",
                "Risk/reward":      quadrant,
            })
        st.dataframe(pd.DataFrame(rr_rows).set_index("Ticker"), width="stretch")

else:
    st.info("Configure your tickers and horizon in the sidebar, then click Run analysis.")
    st.markdown("""
**What this app does:**

**Volatility forecast tab**
- Fits HAR-RV (Corsi 2009) and regime-switching HAR (HAR-RS) models on Yahoo Finance data
- Applies a macro-geopolitical risk premium (Iran conflict, Russia-Ukraine, Fed uncertainty, tariffs)
- Outputs 80% and 95% bootstrap confidence intervals

**Returns analysis tab**
- Historical returns across 7 horizons: 1 day, 1 week, 1 month, 3 months, 6 months, YTD, 1 year
- 2-year CAGR, daily win rate, best and worst single-day returns
- Cumulative return chart (rebased to 100) and monthly returns heatmap

**Risk / reward tab**
- Scatter plot: 1-year return vs forward vol forecast for all tickers simultaneously
- Quadrant classification (high return / low risk, etc.) for quick portfolio positioning
    """)
