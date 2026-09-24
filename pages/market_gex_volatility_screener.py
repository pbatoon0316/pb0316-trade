import math
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

APP_TITLE = "Market GEX & Volatility Screener"
MARKET_TIMEZONE = ZoneInfo("America/New_York")
METADATA_FILE = (
    Path(__file__).resolve().parents[1] / "nasdaq_screener_1779080945125.csv"
)
REQUEST_PAUSE_SECONDS = 0.15
MAX_CONSECUTIVE_PROVIDER_FAILURES = 5
GEX_DIVERGING_COLOR_SCALE = [
    [0.0, "#67001f"],
    [0.25, "#f4a582"],
    [0.5, "#3f3f46"],
    [0.75, "#92c5de"],
    [1.0, "#053061"],
]
SECTOR_LABELS = {"Telecommunications": "Communication Services"}
NON_EQUITY_PATTERNS = (
    r"\bEXCHANGE[- ]TRADED FUND\b",
    r"\bEXCHANGE[- ]TRADED NOTE\b",
    r"\bETF\b",
    r"\bFUND\b",
    r"\bWARRANTS?\b",
    r"\bUNITS?\b",
    r"\bPREFERRED\b",
    r"\bPFD\b",
    r"\bNOTES? DUE\b",
)
DISPLAY_COLUMNS = {
    "ticker": "Ticker",
    "Name": "Company",
    "Sector Display": "Sector",
    "Industry": "Industry",
    "market_cap_b": "Market Cap ($B)",
    "price": "Price",
    "daily_change_pct": "Day %",
    "weekly_ema50": "Weekly EMA50",
    "net_gex": "Net GEX ($)",
    "normalized_gex_bps": "GEX / Market Cap (bp)",
    "gex_regime": "GEX Regime",
    "iv_30_pct": "30D IV (%)",
    "hv_21_pct": "21D HV (%)",
    "iv_hv_spread_pct": "IV-HV (pp)",
    "risk_reversal_25d_pct": "25D RR (%)",
    "risk_reversal_dte": "RR DTE",
    "expiry_count": "Expiries",
}


def normalize_ticker(value: object) -> str:
    return str(value).strip().upper().replace("/", "-")


def equity_exclusion_reason(name: object, symbol: object = "") -> str | None:
    # Inspect the security name rather than the ticker: legitimate tickers such
    # as ETN otherwise look like exchange-traded-note abbreviations. ADRs and
    # REITs remain eligible per the screener's intended equity definition.
    text = str(name or "").upper()
    for pattern in NON_EQUITY_PATTERNS:
        if pd.Series([text]).str.contains(pattern, regex=True).iloc[0]:
            return pattern.replace(r"\b", "").replace("?", "")
    return None


@st.cache_data(ttl="12hr", show_spinner=False)
def load_metadata(path: str = str(METADATA_FILE)) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["ticker"] = frame["Symbol"].map(normalize_ticker)
    frame["Market Cap"] = pd.to_numeric(frame["Market Cap"], errors="coerce")
    frame["Sector"] = frame["Sector"].fillna("Unclassified").astype(str).str.strip()
    frame["Industry"] = frame["Industry"].fillna("Unclassified").astype(str).str.strip()
    frame["Sector Display"] = frame["Sector"].replace(SECTOR_LABELS)
    frame["exclusion_reason"] = frame.apply(
        lambda row: equity_exclusion_reason(row.get("Name"), row.get("Symbol")),
        axis=1,
    )
    frame = frame.dropna(subset=["Market Cap"])
    frame = frame[frame["ticker"].ne("")].drop_duplicates("ticker", keep="first")
    return frame.sort_values("Market Cap", ascending=False).reset_index(drop=True)


def filter_metadata(
    metadata: pd.DataFrame,
    minimum_market_cap: float,
    included_sectors: list[str],
    excluded_industries: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = metadata[
        (metadata["Market Cap"] >= float(minimum_market_cap))
        & metadata["Sector Display"].isin(included_sectors)
        & ~metadata["Industry"].isin(excluded_industries)
    ].copy()
    excluded_security_types = eligible[eligible["exclusion_reason"].notna()].copy()
    eligible = eligible[eligible["exclusion_reason"].isna()].copy()
    return eligible.sort_values("Market Cap", ascending=False), excluded_security_types


def normalize_download_columns(
    data: pd.DataFrame, tickers: tuple[str, ...]
) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        return data
    if len(tickers) != 1:
        return pd.DataFrame()
    normalized = data.copy()
    normalized.columns = pd.MultiIndex.from_product([normalized.columns, [tickers[0]]])
    return normalized


@st.cache_data(ttl="1hr", show_spinner=False)
def download_price_batch(tickers: tuple[str, ...]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        list(tickers),
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    return normalize_download_columns(data, tickers)


def download_price_history(
    tickers: list[str], batch_size: int = 200
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    frames = []
    failures = []
    for start in range(0, len(tickers), batch_size):
        batch = tuple(tickers[start : start + batch_size])
        try:
            frame = download_price_batch(batch)
        except Exception:  # noqa: BLE001 - one failed Yahoo batch should not end the scan
            frame = pd.DataFrame()
        if frame.empty:
            failures.append((start + 1, start + len(batch)))
        else:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), failures
    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined, failures


def ticker_close_series(data: pd.DataFrame, ticker: str) -> pd.Series:
    if data.empty or not isinstance(data.columns, pd.MultiIndex):
        return pd.Series(dtype=float)
    candidates = (("Close", ticker), (ticker, "Close"))
    for column in candidates:
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce").dropna().sort_index()
    return pd.Series(dtype=float)


def latest_price_metrics(close: pd.Series) -> dict | None:
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if len(close) < 252:
        return None
    current = float(close.iloc[-1])
    previous = float(close.iloc[-2])
    if current <= 0 or previous <= 0:
        return None
    weekly = close.resample("W-FRI").last().dropna()
    weekly_ema = weekly.ewm(span=50, adjust=False, min_periods=50).mean()
    if weekly_ema.empty or pd.isna(weekly_ema.iloc[-1]):
        return None
    log_returns = np.log(close / close.shift(1))
    hv = log_returns.tail(21).std(ddof=1) * math.sqrt(252.0) * 100.0
    return {
        "price": current,
        "previous_close": previous,
        "daily_change_pct": (current / previous - 1.0) * 100.0,
        "weekly_ema50": float(weekly_ema.iloc[-1]),
        "hv_21_pct": float(hv) if pd.notna(hv) else np.nan,
        "price_date": pd.Timestamp(close.index[-1]),
    }


def screen_price_history(
    history: pd.DataFrame, metadata: pd.DataFrame, direction: str
) -> pd.DataFrame:
    rows = []
    for record in metadata.to_dict("records"):
        metrics = latest_price_metrics(ticker_close_series(history, record["ticker"]))
        if metrics is None or metrics["price"] <= metrics["weekly_ema50"]:
            continue
        change = metrics["daily_change_pct"]
        if direction == "Up" and change <= 0:
            continue
        if direction == "Down" and change >= 0:
            continue
        rows.append({**record, **metrics})
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("Market Cap", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data(ttl="1hr", show_spinner=False)
def get_option_expiries(ticker: str) -> tuple[str, ...]:
    return tuple(yf.Ticker(ticker).options)


@st.cache_data(ttl="30min", show_spinner=False)
def get_option_chain(ticker: str, expiry: str) -> pd.DataFrame:
    chain = yf.Ticker(ticker).option_chain(expiry)
    frames = []
    for option_type, source in (("call", chain.calls), ("put", chain.puts)):
        if source is None or source.empty:
            continue
        frame = source.copy()
        frame["option_type"] = option_type
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def days_to_expiry(expiry: str, as_of: date | None = None) -> int:
    as_of = as_of or datetime.now(MARKET_TIMEZONE).date()
    return (date.fromisoformat(expiry) - as_of).days


def expiries_within_horizon(
    expiries: tuple[str, ...] | list[str], horizon_dte: int, as_of: date | None = None
) -> list[str]:
    selected = []
    for expiry in expiries:
        dte = days_to_expiry(expiry, as_of)
        if 0 < dte <= int(horizon_dte):
            selected.append(expiry)
    return sorted(selected)


def prepare_options(
    options: pd.DataFrame, expiry: str, as_of: date | None = None
) -> pd.DataFrame:
    if options.empty:
        return pd.DataFrame()
    frame = options.copy()
    frame["expiry"] = pd.Timestamp(expiry)
    frame["dte"] = days_to_expiry(expiry, as_of)
    for column in ("strike", "impliedVolatility", "openInterest", "volume"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["openInterest"] = frame["openInterest"].fillna(0.0)
    frame["volume"] = frame["volume"].fillna(0.0)
    return frame[
        frame["strike"].gt(0)
        & frame["impliedVolatility"].between(0.0001, 5.0)
        & frame["option_type"].isin(["call", "put"])
    ].reset_index(drop=True)


def black_scholes_delta_gamma(
    spot: float,
    strike: float,
    dte: int,
    volatility: float,
    option_type: str,
    risk_free_rate: float = 0.045,
) -> tuple[float, float]:
    if spot <= 0 or strike <= 0 or dte <= 0 or volatility <= 0:
        return np.nan, np.nan
    time_years = float(dte) / 365.0
    root_time = math.sqrt(time_years)
    d1 = (
        math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_years
    ) / (volatility * root_time)
    density = math.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    call_delta = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    delta = call_delta if option_type == "call" else call_delta - 1.0
    gamma = density / (spot * volatility * root_time)
    return delta, gamma


def add_model_greeks(
    options: pd.DataFrame, spot: float, risk_free_rate: float
) -> pd.DataFrame:
    frame = options.copy()
    values = frame.apply(
        lambda row: black_scholes_delta_gamma(
            spot,
            float(row["strike"]),
            int(row["dte"]),
            float(row["impliedVolatility"]),
            str(row["option_type"]),
            risk_free_rate,
        ),
        axis=1,
        result_type="expand",
    )
    values.columns = ["delta", "gamma"]
    frame[["delta", "gamma"]] = values
    return frame


def calculate_net_gex(options: pd.DataFrame, spot: float) -> float:
    if options.empty:
        return np.nan
    signs = np.where(options["option_type"].eq("call"), 1.0, -1.0)
    exposure = (
        options["gamma"] * options["openInterest"] * 100.0 * spot**2 * 0.01 * signs
    )
    return float(exposure.replace([np.inf, -np.inf], np.nan).sum(min_count=1))


def interpolated_atm_iv(
    options: pd.DataFrame, spot: float, target_dte: int = 30
) -> dict | None:
    rows = []
    for (expiry, dte), expiry_frame in options.groupby(["expiry", "dte"]):
        if int(dte) <= 0:
            continue
        type_ivs = []
        for option_type in ("call", "put"):
            candidates = expiry_frame[expiry_frame["option_type"].eq(option_type)]
            candidates = candidates[candidates["impliedVolatility"].notna()]
            if candidates.empty:
                continue
            nearest = (candidates["strike"] - spot).abs().idxmin()
            type_ivs.append(float(candidates.loc[nearest, "impliedVolatility"]))
        if type_ivs:
            rows.append(
                {"expiry": expiry, "dte": int(dte), "iv": float(np.mean(type_ivs))}
            )
    if not rows:
        return None
    term = pd.DataFrame(rows).sort_values("dte")
    exact = term[term["dte"].eq(target_dte)]
    if not exact.empty:
        return {
            "iv": float(exact.iloc[0]["iv"]),
            "lower_dte": target_dte,
            "upper_dte": target_dte,
        }
    lower = term[term["dte"].lt(target_dte)]
    upper = term[term["dte"].gt(target_dte)]
    if lower.empty or upper.empty:
        return None
    low = lower.iloc[-1]
    high = upper.iloc[0]
    low_time = float(low["dte"]) / 365.0
    high_time = float(high["dte"]) / 365.0
    target_time = float(target_dte) / 365.0
    weight = (target_time - low_time) / (high_time - low_time)
    target_variance = float(low["iv"]) ** 2 * low_time
    target_variance += weight * (float(high["iv"]) ** 2 * high_time - target_variance)
    if target_variance < 0:
        return None
    return {
        "iv": math.sqrt(target_variance / target_time),
        "lower_dte": int(low["dte"]),
        "upper_dte": int(high["dte"]),
    }


def risk_reversal_25_delta(options: pd.DataFrame, target_dte: int = 30) -> dict | None:
    if options.empty:
        return None
    available = options[["expiry", "dte"]].drop_duplicates()
    nearest_index = (available["dte"] - target_dte).abs().idxmin()
    chosen = available.loc[nearest_index]
    expiry_frame = options[options["expiry"].eq(chosen["expiry"])]
    call_candidates = expiry_frame[expiry_frame["option_type"].eq("call")].dropna(
        subset=["delta", "impliedVolatility"]
    )
    put_candidates = expiry_frame[expiry_frame["option_type"].eq("put")].dropna(
        subset=["delta", "impliedVolatility"]
    )
    if call_candidates.empty or put_candidates.empty:
        return None
    call_index = (call_candidates["delta"] - 0.25).abs().idxmin()
    put_index = (put_candidates["delta"] + 0.25).abs().idxmin()
    call_row = call_candidates.loc[call_index]
    put_row = put_candidates.loc[put_index]
    return {
        "risk_reversal": float(
            call_row["impliedVolatility"] - put_row["impliedVolatility"]
        ),
        "dte": int(chosen["dte"]),
        "call_strike": float(call_row["strike"]),
        "put_strike": float(put_row["strike"]),
        "call_delta": float(call_row["delta"]),
        "put_delta": float(put_row["delta"]),
    }


def analyze_option_frame(
    options: pd.DataFrame,
    spot: float,
    market_cap: float,
    risk_free_rate: float,
) -> dict:
    modeled = add_model_greeks(options, spot, risk_free_rate)
    net_gex = calculate_net_gex(modeled, spot)
    atm_iv = interpolated_atm_iv(modeled, spot, 30)
    reversal = risk_reversal_25_delta(modeled, 30)
    return {
        "net_gex": net_gex,
        "normalized_gex_bps": net_gex / market_cap * 10_000.0,
        "iv_30_pct": atm_iv["iv"] * 100.0 if atm_iv else np.nan,
        "iv_lower_dte": atm_iv["lower_dte"] if atm_iv else np.nan,
        "iv_upper_dte": atm_iv["upper_dte"] if atm_iv else np.nan,
        "risk_reversal_25d_pct": reversal["risk_reversal"] * 100.0
        if reversal
        else np.nan,
        "risk_reversal_dte": reversal["dte"] if reversal else np.nan,
        "rr_call_strike": reversal["call_strike"] if reversal else np.nan,
        "rr_put_strike": reversal["put_strike"] if reversal else np.nan,
    }


def classify_gex_regimes(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["gex_regime"] = "Unavailable"
    valid = frame["normalized_gex_bps"].notna()
    positive = frame.loc[
        valid & frame["normalized_gex_bps"].ge(0), "normalized_gex_bps"
    ]
    negative = frame.loc[
        valid & frame["normalized_gex_bps"].lt(0), "normalized_gex_bps"
    ]
    positive_cutoff = positive.quantile(0.75) if not positive.empty else np.nan
    negative_cutoff = negative.quantile(0.25) if not negative.empty else np.nan
    frame.loc[valid & frame["normalized_gex_bps"].ge(0), "gex_regime"] = "Positive"
    frame.loc[valid & frame["normalized_gex_bps"].lt(0), "gex_regime"] = "Negative"
    if pd.notna(positive_cutoff):
        frame.loc[
            valid & frame["normalized_gex_bps"].ge(positive_cutoff), "gex_regime"
        ] = "Highly positive"
    if pd.notna(negative_cutoff):
        frame.loc[
            valid & frame["normalized_gex_bps"].le(negative_cutoff), "gex_regime"
        ] = "Highly negative"
    return frame


def analyze_options_candidates(
    candidates: pd.DataFrame,
    horizon_dte: int,
    risk_free_rate: float,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    status_rows = []
    total = len(candidates)
    consecutive_provider_failures = 0
    for position, record in enumerate(candidates.to_dict("records"), start=1):
        ticker = record["ticker"]
        if progress_callback:
            progress_callback(position, total, ticker)
        try:
            expiries = get_option_expiries(ticker)
        except Exception as exc:  # noqa: BLE001 - record per-symbol provider failures
            status_rows.append(
                {"ticker": ticker, "status": "Expiry lookup failed", "detail": str(exc)}
            )
            consecutive_provider_failures += 1
            if consecutive_provider_failures >= MAX_CONSECUTIVE_PROVIDER_FAILURES:
                status_rows[-1]["detail"] += (
                    "; scan stopped after repeated provider failures"
                )
                break
            continue
        selected = expiries_within_horizon(expiries, horizon_dte)
        if not selected:
            status_rows.append(
                {"ticker": ticker, "status": "No expiry in horizon", "detail": ""}
            )
            continue
        frames = []
        failed_expiries = []
        for expiry in selected:
            try:
                chain = prepare_options(get_option_chain(ticker, expiry), expiry)
            except Exception:  # noqa: BLE001 - preserve partial results by expiry
                chain = pd.DataFrame()
            if chain.empty:
                failed_expiries.append(expiry)
            else:
                frames.append(chain)
            time.sleep(REQUEST_PAUSE_SECONDS)
        if not frames:
            status_rows.append(
                {"ticker": ticker, "status": "No usable option chain", "detail": ""}
            )
            consecutive_provider_failures += 1
            if consecutive_provider_failures >= MAX_CONSECUTIVE_PROVIDER_FAILURES:
                status_rows[-1]["detail"] = (
                    "Scan stopped after repeated provider failures; rerun to reuse "
                    "cached successes"
                )
                break
            continue
        consecutive_provider_failures = 0
        options = pd.concat(frames, ignore_index=True)
        analytics = analyze_option_frame(
            options,
            float(record["price"]),
            float(record["Market Cap"]),
            risk_free_rate,
        )
        result_rows.append(
            {
                **record,
                **analytics,
                "expiry_count": len(frames),
                "failed_expiry_count": len(failed_expiries),
                "option_snapshot": pd.Timestamp.now(tz="America/New_York"),
            }
        )
        status_rows.append(
            {
                "ticker": ticker,
                "status": "Analyzed",
                "detail": f"{len(frames)} expiries; {len(failed_expiries)} failed",
            }
        )
    results = pd.DataFrame(result_rows)
    statuses = pd.DataFrame(status_rows)
    if results.empty:
        return results, statuses
    results["market_cap_b"] = results["Market Cap"] / 1_000_000_000.0
    results["iv_hv_spread_pct"] = results["iv_30_pct"] - results["hv_21_pct"]
    results = classify_gex_regimes(results)
    return results.sort_values("Market Cap", ascending=False).reset_index(
        drop=True
    ), statuses


def volatility_scatter(results: pd.DataFrame) -> go.Figure | None:
    plot_data = results.dropna(
        subset=["hv_21_pct", "iv_30_pct", "normalized_gex_bps"]
    ).copy()
    if plot_data.empty:
        return None
    figure = px.scatter(
        plot_data,
        x="hv_21_pct",
        y="iv_30_pct",
        size="market_cap_b",
        color="normalized_gex_bps",
        color_continuous_scale=GEX_DIVERGING_COLOR_SCALE,
        color_continuous_midpoint=0,
        hover_name="ticker",
        hover_data={
            "Name": True,
            "market_cap_b": ":.1f",
            "daily_change_pct": ":+.2f",
            "iv_hv_spread_pct": ":+.2f",
            "risk_reversal_25d_pct": ":+.2f",
            "net_gex": ":,.0f",
            "normalized_gex_bps": ":+.3f",
            "gex_regime": True,
            "hv_21_pct": ":.2f",
            "iv_30_pct": ":.2f",
        },
        labels={
            "hv_21_pct": "21-session historical volatility (%)",
            "iv_30_pct": "30-day implied volatility (%)",
            "normalized_gex_bps": "GEX / market cap (bp)",
            "market_cap_b": "Market cap ($B)",
        },
        title="30-day implied volatility vs. 21-session historical volatility",
    )
    lower = min(plot_data["hv_21_pct"].min(), plot_data["iv_30_pct"].min())
    upper = max(plot_data["hv_21_pct"].max(), plot_data["iv_30_pct"].max())
    padding = max((upper - lower) * 0.08, 1.0)
    figure.add_shape(
        type="line",
        x0=max(0.0, lower - padding),
        y0=max(0.0, lower - padding),
        x1=upper + padding,
        y1=upper + padding,
        line={"color": "gray", "dash": "dash"},
    )
    figure.update_xaxes(range=[max(0.0, lower - padding), upper + padding])
    figure.update_yaxes(range=[max(0.0, lower - padding), upper + padding])
    figure.update_layout(height=325, margin={"l": 20, "r": 20, "t": 50, "b": 20})
    return figure


def tradingview_html(ticker: str) -> str:
    return f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {{
        "autosize": true,
        "height": "290",
        "symbol": "{ticker}",
        "interval": "W",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "hide_top_toolbar": true,
        "allow_symbol_change": false,
        "save_image": false,
        "calendar": false,
        "studies": ["STD;MA%Ribbon"],
        "support_host": "https://www.tradingview.com"
      }}
      </script>
    </div>
    <!-- TradingView Widget END -->
    """


def display_results_table(results: pd.DataFrame) -> None:
    available = [column for column in DISPLAY_COLUMNS if column in results]
    display = results[available].rename(columns=DISPLAY_COLUMNS)
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=325,
        column_config={
            "Market Cap ($B)": st.column_config.NumberColumn(format="$%.1fB"),
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Day %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Weekly EMA50": st.column_config.NumberColumn(format="$%.2f"),
            "Net GEX ($)": st.column_config.NumberColumn(format="$%.0f"),
            "GEX / Market Cap (bp)": st.column_config.NumberColumn(format="%+.3f"),
            "30D IV (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "21D HV (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "IV-HV (pp)": st.column_config.NumberColumn(format="%+.2f"),
            "25D RR (%)": st.column_config.NumberColumn(format="%+.2f%%"),
        },
    )


def render_chart_grid(results: pd.DataFrame, charts_per_page: int) -> tuple[int, int]:
    tickers = results.sort_values("Market Cap", ascending=False)["ticker"].tolist()
    total_pages = max(1, math.ceil(len(tickers) / charts_per_page))
    current = min(
        max(int(st.session_state.get("market_gex_chart_page", 1)), 1), total_pages
    )
    st.session_state["market_gex_chart_page"] = current
    start = (current - 1) * charts_per_page
    visible = tickers[start : start + charts_per_page]
    st.caption(
        f"Showing charts {start + 1}-{min(start + charts_per_page, len(tickers))} of {len(tickers)}"
    )
    columns = st.columns(3)
    for index, ticker in enumerate(visible):
        with columns[index % 3]:
            st.markdown(
                f"{ticker} - [Finviz](https://finviz.com/quote.ashx?t={ticker}&p=d) "
                f"[Profitviz](https://profitviz.com/{ticker})"
            )
            components.html(tradingview_html(ticker), height=300)
    return current, total_pages


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")
    metadata = load_metadata()
    sectors = sorted(metadata["Sector Display"].unique().tolist())
    industries = sorted(metadata["Industry"].unique().tolist())

    with st.sidebar:
        st.header("Screen configuration")
        minimum_market_cap_b = st.number_input(
            "Minimum market cap ($B)", min_value=1.0, value=10.0, step=1.0
        )
        included_sectors = st.multiselect(
            "Included sectors",
            sectors,
            default=sectors,
            help="Clear sectors to exclude them.",
        )
        excluded_industries = st.multiselect(
            "Excluded industries",
            industries,
            default=[],
            placeholder="For example, search Biotechnology",
        )
        direction = st.radio("Direction", ["All", "Up", "Down"], horizontal=True)
        horizon_dte = st.number_input("Expiration horizon (DTE)", 30, 180, 45, 5)
        maximum_symbols = st.number_input("Maximum options candidates", 1, 500, 50, 10)
        risk_free_rate_pct = st.number_input(
            "Risk-free rate (%)",
            0.0,
            20.0,
            4.5,
            0.1,
            help="User-supplied to avoid another market-data request.",
        )
        charts_per_page = st.number_input("Charts per page", 3, 30, 20, 1)
        run_screen = st.button("Run combined screen", type="primary", width="stretch")

    equity_universe, _security_exclusions = filter_metadata(
        metadata,
        minimum_market_cap_b * 1_000_000_000.0,
        sectors,
        [],
    )
    top_500_universe = equity_universe.head(500).copy()
    eligible = top_500_universe[
        top_500_universe["Sector Display"].isin(included_sectors)
        & ~top_500_universe["Industry"].isin(excluded_industries)
    ].copy()
    current_run_config = {
        "minimum_market_cap_b": minimum_market_cap_b,
        "included_sectors": included_sectors,
        "excluded_industries": excluded_industries,
        "horizon_dte": horizon_dte,
        "maximum_symbols": maximum_symbols,
        "risk_free_rate_pct": risk_free_rate_pct,
    }

    if run_screen:
        st.session_state.pop("market_gex_results", None)
        st.session_state.pop("market_gex_statuses", None)
        if eligible.empty:
            st.warning("No symbols remain after the metadata filters.")
        else:
            progress = st.progress(
                0,
                text=(
                    f"Downloading adjusted prices for {len(eligible)} members of the "
                    "top-500 universe…"
                ),
            )
            history, failures = download_price_history(eligible["ticker"].tolist())
            progress.progress(
                15, text="Calculating daily direction and the live 50-week EMA…"
            )
            candidates = screen_price_history(history, eligible, "All")
            analysis_input = candidates.head(int(maximum_symbols)).copy()
            st.session_state["market_gex_price_failures"] = failures

            def update_progress(position: int, total: int, ticker: str) -> None:
                percent = 20 + int((position - 1) / max(total, 1) * 79)
                progress.progress(
                    percent, text=f"Analyzing options {position} of {total}: {ticker}"
                )

            results, statuses = analyze_options_candidates(
                analysis_input,
                int(horizon_dte),
                float(risk_free_rate_pct) / 100.0,
                update_progress,
            )
            st.session_state["market_gex_results"] = results
            st.session_state["market_gex_statuses"] = statuses
            st.session_state["market_gex_candidate_count"] = len(candidates)
            st.session_state["market_gex_run_config"] = current_run_config
            progress.progress(
                100, text=f"Combined screen complete: {len(results)} usable symbols"
            )

    results = st.session_state.get("market_gex_results", pd.DataFrame())
    statuses = st.session_state.get("market_gex_statuses", pd.DataFrame())
    stored_run_config = st.session_state.get("market_gex_run_config")
    if not results.empty and stored_run_config != current_run_config:
        st.warning(
            "The scan configuration changed. Results below are from the previous run; "
            "press Run combined screen to refresh them."
        )

    if results.empty:
        st.info("Run the combined screen to build the volatility and GEX results.")
        if not statuses.empty:
            with st.expander("Options analysis statuses"):
                st.dataframe(statuses, hide_index=True, width="stretch")
        return

    if direction == "Up":
        display_results = results[results["daily_change_pct"].gt(0)].copy()
    elif direction == "Down":
        display_results = results[results["daily_change_pct"].lt(0)].copy()
    else:
        display_results = results.copy()
    display_results = display_results.sort_values("Market Cap", ascending=False)
    if display_results.empty:
        st.info(f"No completed results match the {direction} direction filter.")
        return

    plot_column, table_column = st.columns([1.05, 1.0])
    with plot_column:
        figure = volatility_scatter(display_results)
        if figure is None:
            st.warning(
                "No symbols have complete implied- and historical-volatility data."
            )
        else:
            st.plotly_chart(figure, width="stretch")
    with table_column:
        st.markdown("#### Sortable results")
        display_results_table(display_results)

    _current_page, total_pages = render_chart_grid(
        display_results, int(charts_per_page)
    )

    download_frame = display_results.rename(columns=DISPLAY_COLUMNS)
    page_column, spacer_column, download_column = st.columns(
        [1, 4, 1.35], vertical_alignment="bottom"
    )
    with page_column:
        st.number_input(
            "Chart page",
            min_value=1,
            max_value=total_pages,
            key="market_gex_chart_page",
        )
    with spacer_column:
        st.empty()
    with download_column:
        st.download_button(
            "Download results CSV",
            download_frame.to_csv(index=False).encode("utf-8"),
            file_name=(
                f"market_gex_volatility_"
                f"{datetime.now(MARKET_TIMEZONE).date():%Y%m%d}.csv"
            ),
            mime="text/csv",
            width="stretch",
        )
    with st.expander("Data quality and methodology"):
        if not statuses.empty:
            st.dataframe(statuses, hide_index=True, width="stretch")
        st.markdown(
            "GEX is call gamma exposure minus put gamma exposure, weighted by open "
            "interest and scaled to a 1% underlying move. Normalized GEX divides that "
            "estimate by market cap and reports basis points. This is a positioning "
            "convention, not observed dealer inventory. Thirty-day IV interpolates "
            "ATM total variance; historical volatility uses 21 adjusted close-to-close "
            "log returns annualized by √252. Yahoo data may be delayed or incomplete."
        )


if __name__ == "__main__":
    main()
