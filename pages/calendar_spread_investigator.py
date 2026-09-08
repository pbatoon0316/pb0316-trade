from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import re
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# Analytics


@dataclass(frozen=True)
class MetricDefinition:
    label: str
    column: str
    decimals: int
    diverging: bool


METRICS = {
    "signed_iv_gap": MetricDefinition(
        "Front IV − Back IV (vol pts)", "signed_iv_gap", 2, True
    ),
    "iv_gap_per_day": MetricDefinition(
        "Front IV − Back IV per day", "iv_gap_per_day", 3, True
    ),
    "forward_iv": MetricDefinition("Implied forward IV (%)", "forward_iv", 2, False),
}


def available_strikes(options: pd.DataFrame, option_type: str) -> list[float]:
    values = options.loc[options["option_type"] == option_type, "strike"].dropna().unique()
    return sorted(float(value) for value in values)


def directional_atm_strike(
    strikes: list[float], spot: float, option_type: str
) -> float:
    if not strikes:
        raise ValueError("No strikes are available for this option type.")
    ordered = sorted(float(strike) for strike in strikes)
    if option_type == "call":
        call_target = float(math.ceil(spot))
        at_or_above = [strike for strike in ordered if strike >= call_target]
        return at_or_above[0] if at_or_above else ordered[-1]
    put_target = float(math.floor(spot))
    at_or_below = [strike for strike in ordered if strike <= put_target]
    return at_or_below[-1] if at_or_below else ordered[0]


def term_structure_strikes(
    strikes: list[float],
    spot: float,
    selected_strike: float,
    option_type: str,
    pct_band: float = 0.01,
    maximum_lines: int = 5,
) -> list[float]:
    """Choose a compact one-sided strike band in the option's OTM direction."""
    ordered = sorted(float(strike) for strike in strikes)
    if not ordered:
        return []

    selected_index = min(
        range(len(ordered)), key=lambda index: abs(ordered[index] - selected_strike)
    )
    selected = ordered[selected_index]
    if option_type == "call":
        directional = ordered[selected_index:]
        in_band = [strike for strike in directional if strike <= spot * (1 + pct_band)]
    else:
        directional = list(reversed(ordered[: selected_index + 1]))
        in_band = [strike for strike in directional if strike >= spot * (1 - pct_band)]

    candidates = list(in_band[:maximum_lines])
    for strike in directional:
        if len(candidates) >= maximum_lines:
            break
        if strike not in candidates:
            candidates.append(strike)
    if selected not in candidates:
        candidates.insert(0, selected)
    return sorted(candidates[:maximum_lines])


def strike_term_structure(
    options: pd.DataFrame, option_type: str, strike: float
) -> pd.DataFrame:
    matching = options[
        (options["option_type"] == option_type)
        & np.isclose(options["strike"].astype(float), float(strike))
    ].copy()
    if matching.empty:
        return pd.DataFrame(columns=["expiry", "dte", "iv"])

    matching["iv"] = matching["impliedVolatility"].astype(float)
    term = (
        matching.groupby(["expiry", "dte"], as_index=False)["iv"]
        .median()
        .sort_values(["dte", "expiry"])
    )
    return term.reset_index(drop=True)


def build_pair_metrics(term: pd.DataFrame) -> pd.DataFrame:
    rows = []
    records = term.sort_values("dte").to_dict("records")
    for front_index, front in enumerate(records):
        for back in records[front_index + 1 :]:
            if int(front["dte"]) >= int(back["dte"]):
                continue

            front_iv = float(front["iv"])
            back_iv = float(back["iv"])
            day_gap = int(back["dte"]) - int(front["dte"])
            signed_gap = (front_iv - back_iv) * 100.0
            front_variance = front_iv**2 * float(front["dte"]) / 365.0
            back_variance = back_iv**2 * float(back["dte"]) / 365.0
            forward_variance = (back_variance - front_variance) / (day_gap / 365.0)

            rows.append(
                {
                    "front_expiry": pd.Timestamp(front["expiry"]),
                    "back_expiry": pd.Timestamp(back["expiry"]),
                    "front_dte": int(front["dte"]),
                    "back_dte": int(back["dte"]),
                    "front_iv": front_iv * 100.0,
                    "back_iv": back_iv * 100.0,
                    "signed_iv_gap": signed_gap,
                    "iv_gap_per_day": signed_gap / day_gap,
                    "forward_iv": (
                        np.sqrt(forward_variance) * 100.0
                        if forward_variance >= 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def rank_pairs(pairs: pd.DataFrame, metric_key: str, limit: int = 10) -> pd.DataFrame:
    metric = METRICS[metric_key]
    if pairs.empty:
        return pairs.copy()
    ranked = (
        pairs[pairs["signed_iv_gap"] > 0]
        .dropna(subset=[metric.column])
        .sort_values(metric.column, ascending=False)
        .head(limit)
        .copy()
    )
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def black_scholes_delta(
    spot: float,
    strike: float,
    dte: int,
    iv_percent: float,
    option_type: str,
    risk_free_rate: float = 0.0,
) -> float:
    """Estimate a contract delta from the option-chain IV already in memory."""
    time_to_expiry = max(float(dte), 1.0) / 365.0
    volatility = float(iv_percent) / 100.0
    if spot <= 0 or strike <= 0 or volatility <= 0:
        return np.nan
    d1 = (
        math.log(float(spot) / float(strike))
        + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * math.sqrt(time_to_expiry))
    call_delta = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    return call_delta if option_type == "call" else call_delta - 1.0


def add_pair_deltas(
    ranked: pd.DataFrame,
    spot: float,
    strike: float,
    option_type: str,
) -> pd.DataFrame:
    """Add front/back estimated deltas only after the display pairs are ranked."""
    result = ranked.copy()
    if result.empty:
        result["front_delta"] = pd.Series(dtype=float)
        result["back_delta"] = pd.Series(dtype=float)
        return result
    result["front_delta"] = result.apply(
        lambda row: black_scholes_delta(
            spot, strike, row["front_dte"], row["front_iv"], option_type
        ),
        axis=1,
    )
    result["back_delta"] = result.apply(
        lambda row: black_scholes_delta(
            spot, strike, row["back_dte"], row["back_iv"], option_type
        ),
        axis=1,
    )
    return result


def optionstrat_calendar_url(
    ticker: str,
    option_type: str,
    strike: float,
    front_expiry,
    back_expiry,
    options: pd.DataFrame | None = None,
) -> str:
    """Build an OptionStrat long-calendar URL for a ranked expiration pair."""

    def leg_symbol(expiry) -> str:
        expiry = pd.Timestamp(expiry)
        if options is not None and "contractSymbol" in options.columns:
            matches = options[
                (options["option_type"] == option_type)
                & (options["expiry"] == expiry)
                & np.isclose(options["strike"].astype(float), float(strike))
            ]
            if not matches.empty:
                contract_symbol = str(matches.iloc[0]["contractSymbol"])
                occ_match = re.fullmatch(r"(.+?)(\d{6})([CP])(\d{8})", contract_symbol)
                if occ_match:
                    root, expiry_code, call_put, strike_code = occ_match.groups()
                    contract_strike = int(strike_code) / 1000.0
                    return f".{root}{expiry_code}{call_put}{contract_strike:g}"

        root = ticker.upper().lstrip("^").replace("-", "")
        call_put = "C" if option_type == "call" else "P"
        return f".{root}{expiry:%y%m%d}{call_put}{float(strike):g}"

    underlying = ticker.upper().lstrip("^")
    strategy = "calendar-call-spread" if option_type == "call" else "calendar-put-spread"
    front_leg = leg_symbol(front_expiry)
    back_leg = leg_symbol(back_expiry)
    return (
        f"https://optionstrat.com/build/{strategy}/{underlying}/"
        f"-{front_leg},{back_leg}"
    )


def compute_historical_volatility(price_history: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(price_history["Close"], errors="coerce").dropna()
    log_returns = np.log(close / close.shift(1))
    result = pd.DataFrame(index=close.index)
    for window in (5, 10, 20, 30, 50, 100):
        result[f"HV{window}"] = (
            log_returns.rolling(window).std(ddof=1) * np.sqrt(252.0) * 100.0
        )
    return result.dropna(how="all")


def interpolated_atm_iv(
    options: pd.DataFrame, spot: float, target_dte: int = 30
) -> dict | None:
    """Estimate constant-maturity ATM IV by interpolating total variance."""
    rows = []
    for (expiry, dte), expiry_frame in options.groupby(["expiry", "dte"]):
        if int(dte) <= 0:
            continue
        type_ivs = []
        for option_type in ("call", "put"):
            candidates = expiry_frame[expiry_frame["option_type"] == option_type].copy()
            candidates = candidates[candidates["impliedVolatility"].notna()]
            if candidates.empty:
                continue
            nearest_index = (candidates["strike"].astype(float) - spot).abs().idxmin()
            type_ivs.append(float(candidates.loc[nearest_index, "impliedVolatility"]))
        if type_ivs:
            rows.append(
                {
                    "expiry": pd.Timestamp(expiry),
                    "dte": int(dte),
                    "iv": float(np.mean(type_ivs)),
                }
            )

    term = pd.DataFrame(rows).sort_values("dte") if rows else pd.DataFrame()
    if term.empty:
        return None

    exact = term[term["dte"] == target_dte]
    if not exact.empty:
        row = exact.iloc[0]
        return {
            "iv": float(row["iv"]),
            "lower_dte": target_dte,
            "upper_dte": target_dte,
        }

    lower = term[term["dte"] < target_dte]
    upper = term[term["dte"] > target_dte]
    if lower.empty or upper.empty:
        return None

    lower_row = lower.iloc[-1]
    upper_row = upper.iloc[0]
    lower_time = float(lower_row["dte"]) / 365.0
    upper_time = float(upper_row["dte"]) / 365.0
    target_time = float(target_dte) / 365.0
    lower_variance = float(lower_row["iv"]) ** 2 * lower_time
    upper_variance = float(upper_row["iv"]) ** 2 * upper_time
    weight = (target_time - lower_time) / (upper_time - lower_time)
    target_variance = lower_variance + weight * (upper_variance - lower_variance)
    if target_variance < 0:
        return None

    return {
        "iv": float(np.sqrt(target_variance / target_time)),
        "lower_dte": int(lower_row["dte"]),
        "upper_dte": int(upper_row["dte"]),
    }


# Market data


MARKET_TIMEZONE = ZoneInfo("America/New_York")
YFINANCE_CACHE = Path(tempfile.gettempdir()) / "pb0316_yfinance_cache"
YFINANCE_CACHE.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE))


def market_date():
    return datetime.now(MARKET_TIMEZONE).date()


@st.cache_data(ttl=900, show_spinner=False)
def _spot_price(ticker: str) -> float:
    stock = yf.Ticker(ticker)
    try:
        last_price = stock.fast_info["last_price"]
        if pd.notna(last_price) and float(last_price) > 0:
            return float(last_price)
    except Exception:
        pass

    history = stock.history(period="5d", auto_adjust=True)
    if history.empty or history["Close"].dropna().empty:
        raise ValueError(f"No current price was returned for {ticker}.")
    return float(history["Close"].dropna().iloc[-1])


@st.cache_data(ttl=900, show_spinner=False)
def _listed_expiries(ticker: str) -> list[str]:
    return list(yf.Ticker(ticker).options)


@st.cache_data(ttl=900, show_spinner=False)
def _chain_for_expiry(ticker: str, expiry: str) -> pd.DataFrame:
    chain = yf.Ticker(ticker).option_chain(expiry)
    frames = []
    for option_type, frame in (("call", chain.calls), ("put", chain.puts)):
        if frame.empty:
            continue
        option_frame = frame.copy()
        option_frame["option_type"] = option_type
        option_frame["expiry"] = pd.Timestamp(expiry)
        frames.append(option_frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_calendar_snapshot(ticker: str, max_dte: int = 60):
    """Load the spot and every listed option chain through max_dte."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Enter a ticker symbol.")

    as_of = market_date()
    spot = _spot_price(ticker)
    listed_expiries = _listed_expiries(ticker)
    expiries = []
    for expiry in listed_expiries:
        expiry_date = pd.Timestamp(expiry).date()
        dte = (expiry_date - as_of).days
        if 0 <= dte <= max_dte:
            expiries.append(expiry)

    if len(expiries) < 2:
        raise ValueError(
            f"Fewer than two option expirations were found within {max_dte} DTE."
        )

    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for expiry in expiries:
        try:
            frame = _chain_for_expiry(ticker, expiry)
            if frame.empty:
                failures.append(expiry)
            else:
                frames.append(frame)
        except Exception:
            failures.append(expiry)

    if not frames:
        raise ValueError("No option-chain data was returned.")

    options = pd.concat(frames, ignore_index=True)
    options["strike"] = pd.to_numeric(options["strike"], errors="coerce")
    options["impliedVolatility"] = pd.to_numeric(
        options["impliedVolatility"], errors="coerce"
    )
    options["dte"] = options["expiry"].map(
        lambda value: (value.date() - as_of).days
    )
    options = options[
        options["strike"].notna()
        & options["impliedVolatility"].between(0.0001, 10.0)
    ].copy()

    return spot, options, sorted(failures), datetime.now(MARKET_TIMEZONE)


@st.cache_data(ttl=3600, show_spinner=False)
def load_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    history = yf.Ticker(ticker.strip().upper()).history(period=period, auto_adjust=True)
    if history.empty or "Close" not in history:
        raise ValueError(f"No price history was returned for {ticker}.")
    return history[["Close"]].dropna().copy()


def earnings_session(timestamp: pd.Timestamp) -> str:
    """Classify Yahoo's earnings timestamp into a market session."""
    hour = timestamp.hour
    if hour < 12:
        return "Before open"
    if hour >= 16:
        return "After close"
    return "Time not provided"


@st.cache_data(ttl=900, show_spinner=False)
def load_next_earnings(ticker: str) -> dict | None:
    """Return the next Yahoo earnings timestamp and its reported market session."""
    ticker = ticker.strip().upper()
    now = pd.Timestamp.now(tz=str(MARKET_TIMEZONE))
    try:
        earnings = yf.Ticker(ticker).get_earnings_dates(limit=12)
    except Exception:
        earnings = None

    if earnings is not None and not earnings.empty:
        dates = pd.DatetimeIndex(earnings.index)
        if dates.tz is None:
            dates = dates.tz_localize(MARKET_TIMEZONE)
        else:
            dates = dates.tz_convert(MARKET_TIMEZONE)
        future_dates = dates[dates >= now]
        if len(future_dates):
            timestamp = pd.Timestamp(future_dates.min())
            return {
                "timestamp": timestamp,
                "date": timestamp.date(),
                "session": earnings_session(timestamp),
                "has_time": True,
            }

    try:
        calendar = yf.Ticker(ticker).calendar or {}
        calendar_dates = calendar.get("Earnings Date", [])
        if calendar_dates:
            next_date = min(
                pd.Timestamp(value).date()
                for value in calendar_dates
                if pd.Timestamp(value).date() >= now.date()
            )
            return {
                "timestamp": None,
                "date": next_date,
                "session": "Time not provided",
                "has_time": False,
            }
    except (Exception, ValueError):
        pass
    return None


# Charts


def _expiry_label(expiry, dte: int) -> str:
    return f"{pd.Timestamp(expiry):%b %d} ({int(dte)}d)"


def term_structure_figure(
    terms_by_strike: dict[float, pd.DataFrame],
    option_type: str,
    selected_strike: float,
    spot: float,
    earnings_dte: int | None = None,
    earnings_label: str | None = None,
    max_dte: int = 60,
) -> go.Figure:
    figure = go.Figure()
    palette = ("#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a", "#19d3f3")
    ordered_strikes = sorted(
        terms_by_strike,
        key=lambda value: (np.isclose(value, selected_strike), value),
    )
    for index, strike in enumerate(ordered_strikes):
        term = terms_by_strike[strike]
        is_selected = bool(np.isclose(strike, selected_strike))
        labels = [_expiry_label(row.expiry, row.dte) for row in term.itertuples()]
        figure.add_trace(
            go.Scatter(
                x=term["dte"],
                y=term["iv"] * 100.0,
                mode="lines+markers",
                name=f"Strike {strike:g}" + (" · selected" if is_selected else ""),
                opacity=1.0 if is_selected else 0.5,
                line={
                    "color": "#2563eb" if is_selected else palette[index % len(palette)],
                    "width": 3 if is_selected else 1.5,
                },
                marker={"size": 8 if is_selected else 5},
                customdata=np.column_stack(
                    (
                        labels,
                        term["expiry"].dt.strftime("%Y-%m-%d"),
                        np.full(len(term), strike),
                    )
                ),
                hovertemplate=(
                    "Strike: %{customdata[2]}<br>%{customdata[0]}"
                    "<br>Expiry: %{customdata[1]}<br>IV: %{y:.2f}%<extra></extra>"
                ),
            )
        )

    if earnings_dte is not None and 0 <= earnings_dte <= max_dte:
        figure.add_vline(
            x=earnings_dte,
            line={"color": "#dc2626", "width": 2, "dash": "dot"},
        )
        figure.add_annotation(
            x=earnings_dte,
            y=0,
            xref="x",
            yref="paper",
            text=earnings_label or "Earnings",
            showarrow=False,
            align="center",
            xanchor="left",
            xshift=8,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.75)",
            borderpad=4,
            font={"color": "#dc2626"},
        )

    figure.update_layout(
        title=f"{option_type.title()} IV term structures",
        xaxis={
            "title": "Days to expiration",
            "unifiedhovertitle": {"text": "DTE: %{x}"},
        },
        yaxis_title="Implied volatility (%)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 50, "r": 25, "t": 85, "b": 50},
        height=520,
    )
    return figure


def pair_heatmap_figure(pairs: pd.DataFrame, metric_key: str) -> go.Figure:
    metric = METRICS[metric_key]
    front_order = (
        pairs[["front_expiry", "front_dte"]]
        .drop_duplicates()
        .sort_values("front_dte")
    )
    back_order = (
        pairs[["back_expiry", "back_dte"]]
        .drop_duplicates()
        .sort_values("back_dte")
    )
    front_labels = [
        _expiry_label(row.front_expiry, row.front_dte) for row in front_order.itertuples()
    ]
    back_labels = [
        _expiry_label(row.back_expiry, row.back_dte) for row in back_order.itertuples()
    ]
    front_lookup = {
        row.front_expiry: _expiry_label(row.front_expiry, row.front_dte)
        for row in front_order.itertuples()
    }
    back_lookup = {
        row.back_expiry: _expiry_label(row.back_expiry, row.back_dte)
        for row in back_order.itertuples()
    }

    working = pairs.copy()
    working["front_label"] = working["front_expiry"].map(front_lookup)
    working["back_label"] = working["back_expiry"].map(back_lookup)
    matrix = working.pivot(index="front_label", columns="back_label", values=metric.column)
    matrix = matrix.reindex(index=front_labels, columns=back_labels)

    finite_values = matrix.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    positive_values = finite_values[finite_values > 0]
    bound = float(np.max(positive_values)) if positive_values.size else 1.0
    if bound == 0:
        bound = 1.0

    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            texttemplate=f"%{{z:.{metric.decimals}f}}",
            textfont={"size": 11},
            hovertemplate="%{y} → %{x}<br>%{z}<extra></extra>",
            colorbar={"title": metric.label},
            hoverongaps=False,
            colorscale=[
                [0.0, "#ffffff"],
                [1.0, "#00b050"],
            ],
            autocolorscale=False,
            reversescale=False,
            zmin=0,
            zmax=bound,
        )
    )
    figure.update_layout(
        title=metric.label,
        xaxis_title="Back expiration",
        yaxis_title="Front expiration",
        xaxis={"side": "top", "tickangle": -45},
        yaxis={"autorange": "reversed"},
        margin={"l": 110, "r": 60, "t": 120, "b": 40},
        height=520,
    )
    return figure


def historical_volatility_figure(
    hv: pd.DataFrame, current_30d_iv: float | None = None
) -> go.Figure:
    figure = go.Figure()
    series_colors = (
        ("HV5", "#dc2626"),
        ("HV10", "#2563eb"),
        ("HV20", "#f59e0b"),
        ("HV30", "#16a34a"),
        ("HV50", "#9333ea"),
        ("HV100", "#0891b2"),
    )
    for column, color in series_colors:
        initially_visible = column in {"HV10", "HV20"}
        figure.add_trace(
            go.Scatter(
                x=hv.index,
                y=hv[column],
                mode="lines",
                name=column,
                line={"color": color, "width": 2},
                visible=True if initially_visible else "legendonly",
                hovertemplate=f"{column}: %{{y:.2f}}%<extra></extra>",
            )
        )
    if current_30d_iv is not None and not hv.empty:
        figure.add_trace(
            go.Scatter(
                x=[hv.index.min(), hv.index.max()],
                y=[current_30d_iv, current_30d_iv],
                mode="lines",
                name="Current 30d IV",
                line={"color": "#000000", "width": 2, "dash": "dash"},
                showlegend=False,
                hovertemplate="Current 30d IV: %{y:.2f}%<extra></extra>",
            )
        )
        midpoint = hv.index.min() + (hv.index.max() - hv.index.min()) / 2
        figure.add_annotation(
            x=midpoint,
            y=current_30d_iv,
            text=f"Current 30d IV: {current_30d_iv:.2f}%",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.75)",
            borderpad=4,
            font={"color": "#000000"},
        )
    figure.update_layout(
        title="Historical realized volatility",
        xaxis_title=None,
        yaxis_title="Annualized volatility (%)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 50, "r": 25, "t": 85, "b": 40},
        height=520,
    )
    return figure


# Streamlit page


DEFAULT_MAX_DTE = 60
DTE_OPTIONS = [60, 90, 120, 180]
STATE = "calendar_spread_"


def state_key(name: str) -> str:
    return f"{STATE}{name}"


def load_ticker(ticker: str, max_dte: int) -> None:
    ticker = ticker.strip().upper()
    with st.spinner(f"Loading {ticker} option chains through {max_dte} DTE..."):
        spot, options, failures, fetched_at = load_calendar_snapshot(ticker, max_dte)
        history = None
        history_error = None
        try:
            history = load_price_history(ticker)
        except Exception as exc:
            history_error = str(exc)
        try:
            earnings = load_next_earnings(ticker)
        except Exception:
            earnings = None

    st.session_state[state_key("ticker")] = ticker
    st.session_state[state_key("spot")] = spot
    st.session_state[state_key("options")] = options
    st.session_state[state_key("failures")] = failures
    st.session_state[state_key("fetched_at")] = fetched_at
    st.session_state[state_key("history")] = history
    st.session_state[state_key("history_error")] = history_error
    st.session_state[state_key("earnings")] = earnings
    st.session_state[state_key("max_dte")] = max_dte


def pair_table(
    ranked: pd.DataFrame, metric_key: str, optionstrat_urls: list[str]
) -> pd.DataFrame:
    display = ranked[
        [
            "rank",
            "front_expiry",
            "back_expiry",
            "front_dte",
            "back_dte",
            "front_iv",
            "back_iv",
            "front_delta",
            "back_delta",
            "signed_iv_gap",
            "iv_gap_per_day",
            "forward_iv",
        ]
    ].copy()
    display["front_expiry"] = display["front_expiry"].dt.strftime("%Y-%m-%d")
    display["back_expiry"] = display["back_expiry"].dt.strftime("%Y-%m-%d")
    display["OptionStrat"] = optionstrat_urls
    display = display.rename(
        columns={
            "rank": "Rank",
            "front_expiry": "Front Expiry",
            "back_expiry": "Back Expiry",
            "front_dte": "Front DTE",
            "back_dte": "Back DTE",
            "front_iv": "Front IV %",
            "back_iv": "Back IV %",
            "front_delta": "Front Delta",
            "back_delta": "Back Delta",
            "signed_iv_gap": "Signed Gap",
            "iv_gap_per_day": "Gap / Day",
            "forward_iv": "Forward IV %",
        }
    )
    numeric_columns = display.select_dtypes(include="number").columns.difference(
        ["Rank", "Front DTE", "Back DTE"]
    )
    display[numeric_columns] = display[numeric_columns].round(3)

    ordered = [
        "Rank",
        "Front Expiry",
        "Back Expiry",
        "Front DTE",
        "Back DTE",
        "Front IV %",
        "Back IV %",
        "Signed Gap",
    ]
    ordered += [
        column
        for column in display.columns
        if column not in ordered and column != "OptionStrat"
    ]
    ordered.append("OptionStrat")
    return display[ordered]


def main() -> None:
    st.set_page_config(page_title="Calendar Spread Investigator", layout="wide")

    with st.sidebar:
        ticker_input = st.text_input("Ticker", value="SPY", key=state_key("ticker_input"))
        normalized_input = ticker_input.strip().upper()
        dte_context_key = state_key("dte_context")
        requested_dte_key = state_key("requested_max_dte")
        ticker_needs_initial_load = (
            st.session_state.get(state_key("ticker")) != normalized_input
        )
        if st.session_state.get(dte_context_key) != normalized_input:
            st.session_state[requested_dte_key] = DEFAULT_MAX_DTE
            st.session_state[dte_context_key] = normalized_input
        requested_max_dte = st.selectbox(
            "Maximum DTE",
            options=DTE_OPTIONS,
            index=DTE_OPTIONS.index(DEFAULT_MAX_DTE),
            format_func=lambda value: f"{value} days",
            key=requested_dte_key,
            disabled=ticker_needs_initial_load,
        )
        if st.button("Load / Refresh", type="primary", use_container_width=True):
            try:
                load_ticker(ticker_input, requested_max_dte)
            except Exception as exc:
                st.error(str(exc))

    options = st.session_state.get(state_key("options"))
    if options is None or options.empty:
        st.info("Enter a ticker and select **Load / Refresh** to begin.")
        return

    ticker = st.session_state[state_key("ticker")]
    spot = float(st.session_state[state_key("spot")])
    loaded_max_dte = int(
        st.session_state.get(state_key("max_dte"), DEFAULT_MAX_DTE)
    )

    with st.sidebar:
        st.divider()
        if requested_max_dte != loaded_max_dte:
            st.info(
                f"Select Load / Refresh to apply the {requested_max_dte}-day range."
            )
        option_type = st.radio(
            "Option type",
            options=["call", "put"],
            format_func=lambda value: value.title(),
            horizontal=True,
            key=state_key("option_type"),
        )

        strikes = available_strikes(options, option_type)
        if not strikes:
            st.error(f"No {option_type} strikes are available.")
            return
        strike_widget_key = state_key("strike")
        strike_context_key = state_key("strike_context")
        strike_context = f"{ticker}:{option_type}"
        context_changed = st.session_state.get(strike_context_key) != strike_context
        if context_changed or st.session_state.get(strike_widget_key) not in strikes:
            st.session_state[strike_widget_key] = directional_atm_strike(
                strikes, spot, option_type
            )
            st.session_state[strike_context_key] = strike_context
        strike = st.selectbox(
            "Strike",
            options=strikes,
            format_func=lambda value: f"{value:g}",
            key=strike_widget_key,
        )

        term = strike_term_structure(options, option_type, strike)
        if len(term) < 2:
            st.warning(
                "This strike is not listed for at least two expirations within "
                f"{loaded_max_dte} DTE."
            )
            return

        metric_labels = {key: definition.label for key, definition in METRICS.items()}
        if st.session_state.get(state_key("metric")) not in METRICS:
            st.session_state[state_key("metric")] = "signed_iv_gap"
        metric_key = st.selectbox(
            "Gap metric",
            options=list(METRICS),
            format_func=lambda value: metric_labels[value],
            key=state_key("metric"),
        )
        fetched_at = st.session_state.get(state_key("fetched_at"))
        if fetched_at is not None:
            st.caption(f"Option snapshot loaded {fetched_at:%Y-%m-%d %H:%M:%S %Z}.")

    pairs = build_pair_metrics(term)
    atm_30 = interpolated_atm_iv(options, spot, target_dte=30)

    summary_columns = st.columns(3)
    summary_columns[0].metric("Underlying", f"{ticker} · ${spot:,.2f}")
    summary_columns[1].metric(
        "Selected contract", f"{option_type.title()} {strike:g}"
    )
    if atm_30 is not None:
        summary_columns[2].metric("Current 30-day ATM IV", f"{atm_30['iv'] * 100:.2f}%")
    else:
        summary_columns[2].metric("Current 30-day ATM IV", "N/A")

    failures = st.session_state.get(state_key("failures"), [])
    if failures:
        st.warning(
            f"{len(failures)} expiration(s) could not be loaded: {', '.join(failures)}"
        )

    nearby_strikes = term_structure_strikes(strikes, spot, strike, option_type)
    nearby_terms = {
        nearby_strike: nearby_term
        for nearby_strike in sorted(nearby_strikes)
        if not (
            nearby_term := strike_term_structure(options, option_type, nearby_strike)
        ).empty
    }
    earnings = st.session_state.get(state_key("earnings"))
    fetched_at = st.session_state.get(state_key("fetched_at"))
    earnings_dte = None
    earnings_label = None
    if earnings is not None and fetched_at is not None:
        earnings_dte = (earnings["date"] - fetched_at.date()).days
        earnings_label = (
            f"Next earnings: {earnings['date']:%b %d, %Y}<br>"
            f"{earnings['session']}"
        )

    term_column, matrix_column = st.columns(2)

    with term_column:
        st.plotly_chart(
            term_structure_figure(
                nearby_terms,
                option_type,
                strike,
                spot,
                earnings_dte=earnings_dte,
                earnings_label=earnings_label,
                max_dte=loaded_max_dte,
            ),
            width="stretch",
            key=state_key("term_chart_configurable_dte_v1"),
        )

    with matrix_column:
        st.plotly_chart(
            pair_heatmap_figure(pairs, metric_key),
            width="stretch",
            key=state_key("matrix_chart_white_green_v1"),
        )
    hv_column, _ = st.columns(2)
    history = st.session_state.get(state_key("history"))
    with hv_column:
        if history is None or history.empty:
            st.warning(
                st.session_state.get(state_key("history_error"))
                or "Historical price data is unavailable."
            )
        else:
            hv = compute_historical_volatility(history).tail(252)
            st.plotly_chart(
                historical_volatility_figure(
                    hv, atm_30["iv"] * 100.0 if atm_30 is not None else None
                ),
                width="stretch",
                key=state_key("hv_chart_with_iv_reference_v2"),
            )

    st.subheader("Top 5 expiration pairs")
    ranked = rank_pairs(pairs, metric_key, limit=5)
    ranked = add_pair_deltas(ranked, spot, strike, option_type)
    optionstrat_urls = [
        optionstrat_calendar_url(
            ticker,
            option_type,
            strike,
            row.front_expiry,
            row.back_expiry,
            options,
        )
        for row in ranked.itertuples()
    ]
    st.dataframe(
        pair_table(ranked, metric_key, optionstrat_urls),
        hide_index=True,
        width="stretch",
        column_config={
            "OptionStrat": st.column_config.LinkColumn(
                "OptionStrat", display_text="Open diagram"
            )
        },
    )


if __name__ == "__main__":
    main()
