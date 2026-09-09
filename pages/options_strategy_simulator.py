"""Self-contained options strategy P&L simulator Streamlit page."""

# ============================================================
# IMPORTS AND PAGE CONFIGURATION
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import brentq
from scipy.stats import norm
import streamlit as st
import yfinance as yf


# yfinance 1.x uses a small SQLite cookie/time-zone cache. Its default user
# cache directory can be read-only in hosted or sandboxed Streamlit sessions.
YFINANCE_CACHE_DIR = Path(tempfile.gettempdir()) / "pb0316_options_simulator_yfinance"
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))


st.set_page_config(
    page_title="Options Strategy Simulator",
    page_icon="📐",
    layout="wide",
)


# ============================================================
# CONSTANTS AND DATA MODELS
# ============================================================

CONTRACT_MULTIPLIER = 100
CALENDAR_DAYS = 365.0
MIN_VOLATILITY = 0.0001
MAX_VOLATILITY = 5.0
DEFAULT_RATE = 0.043
DEFAULT_FRONT_DTE = 21
DEFAULT_BACK_DTE = 45
STANDARD_CRUSH_POINTS = 5.0
WIDE_SPREAD_THRESHOLD = 0.25
LOW_OPEN_INTEREST = 20
LOW_VOLUME = 5

SINGLE_EXPIRY_STRATEGIES = [
    "Long Call",
    "Short Call",
    "Long Put",
    "Short Put",
    "Bull Call Debit Spread",
    "Bear Call Credit Spread",
    "Bear Put Debit Spread",
    "Bull Put Credit Spread",
    "Short Straddle",
    "Short Strangle",
    "Iron Condor",
]
TIME_SPREAD_STRATEGIES = [
    "Call Calendar / Diagonal",
    "Put Calendar / Diagonal",
    "Double Calendar / Diagonal",
]
ALL_STRATEGIES = SINGLE_EXPIRY_STRATEGIES + TIME_SPREAD_STRATEGIES


@dataclass(frozen=True)
class OptionLeg:
    option_type: str
    side: int
    quantity: int
    expiration: date
    strike: float
    bid: float
    ask: float
    last: float
    market_mid: float
    yahoo_iv: float
    model_iv: float
    volume: float
    open_interest: float
    calculated_iv: float = 0.30
    iv_source: str = "Yahoo"
    quote_source: str = "midpoint"
    multiplier: int = CONTRACT_MULTIPLIER

    @property
    def side_label(self) -> str:
        return "Long" if self.side > 0 else "Short"

    @property
    def label(self) -> str:
        return f"{self.side_label} {self.option_type.title()}"


@dataclass(frozen=True)
class MarketContext:
    symbol: str
    spot: float
    expirations: tuple[date, ...]
    dividend_yield: float
    retrieved_at: datetime


@dataclass(frozen=True)
class ExecutionEstimate:
    natural: float
    midpoint: float
    favorable: float


@dataclass(frozen=True)
class AdvancedMetrics:
    """Position-scaled entry exposures, efficiency ratios and scenarios."""

    front_vega: float
    back_vega: float | None
    capital_at_risk: float | None
    vega_to_risk_percent: float | None
    theta_to_risk_percent: float | None
    theta_to_gamma_risk: float | None
    back_to_front_vega: float | None
    crush_capture_ratio: float | None
    back_iv_crush_breakeven: str
    vanna: float
    charm: float
    vomma: float
    center_pnl_after_shock: float
    pnl_through_expected_move: float
    raw_iv_gap_points: float | None


# ============================================================
# GENERIC UTILITIES
# ============================================================

def finite_float(value: object, default: float = 0.0) -> float:
    """Return a finite float or a controlled default."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def years_between(start: date, end: date) -> float:
    return max((end - start).days, 0) / CALENDAR_DAYS


def nearest_expiration_index(
    expirations: Sequence[date],
    valuation_date: date,
    target_dte: int,
) -> int:
    """Return the index of the listed expiration closest to a target DTE."""
    if not expirations:
        raise ValueError("At least one expiration is required.")
    return min(
        range(len(expirations)),
        key=lambda index: abs((expirations[index] - valuation_date).days - target_dte),
    )


def money(value: float) -> str:
    if not math.isfinite(value):
        return "—"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def price_text(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def execution_range_text(execution: ExecutionEstimate) -> str:
    """Format natural, midpoint and favorable prices as one compact range."""
    return (
        f"${abs(execution.natural):,.2f} · "
        f"{abs(execution.midpoint):,.2f} · "
        f"{abs(execution.favorable):,.2f}"
    )


def signed_iv(value: float) -> str:
    return f"{value * 100:.1f}%"


def iv_text(value: float) -> str:
    return signed_iv(value) if math.isfinite(value) else "—"


def query_text(name: str, default: str = "") -> str:
    """Read one persisted page value without exposing query API quirks."""
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        value = value[-1] if value else default
    return str(value)


def query_float(name: str, default: float) -> float:
    try:
        return float(query_text(name, str(default)))
    except (TypeError, ValueError):
        return default


def persist_query_value(name: str, value: object) -> None:
    """Persist interactive setup state in the URL for browser refreshes."""
    text_value = str(value)
    if query_text(name) != text_value:
        st.query_params[name] = text_value


def reset_position_defaults(context_key: str) -> None:
    """Clear current-position values while preserving view preferences."""
    state_prefixes = (
        f"strike_value_{context_key}_",
        f"strike_list_{context_key}_",
        f"strike_slider_{context_key}_",
        f"strike_mode_{context_key}_",
        f"chart_range_{context_key}_",
        f"front_iv_value_{context_key}_",
        f"back_iv_value_{context_key}_",
    )
    exact_state_keys = {
        f"front_iv_value_{context_key}",
        f"back_iv_value_{context_key}",
        f"simulation_date_{context_key}",
    }
    for key in list(st.session_state):
        if key in exact_state_keys or key.startswith(state_prefixes):
            del st.session_state[key]

    for key in list(st.query_params):
        if key.startswith("strike_") and key != "strike_view":
            del st.query_params[key]
        elif key.startswith(("front_iv", "back_iv")):
            del st.query_params[key]


# ============================================================
# YAHOO MARKET DATA
# ============================================================

@st.cache_data(ttl=120, show_spinner=False)
def _load_market_context_payload(
    symbol: str,
) -> tuple[str, float, tuple[str, ...], float, str]:
    """Load a cache-safe underlying snapshot using only primitive values."""
    ticker = yf.Ticker(symbol)
    expiration_strings = tuple(str(item) for item in ticker.options)
    if not expiration_strings:
        raise ValueError(f"Yahoo Finance returned no listed options for {symbol}.")

    spot = finite_float(ticker.fast_info.get("last_price"), -1.0)
    if spot <= 0:
        history = ticker.history(period="5d", auto_adjust=False)
        if history.empty:
            raise ValueError(f"A current underlying price was not available for {symbol}.")
        spot = finite_float(history["Close"].dropna().iloc[-1], -1.0)
    if spot <= 0:
        raise ValueError(f"A valid underlying price was not available for {symbol}.")

    dividend_yield = 0.0
    try:
        dividend_yield = finite_float(ticker.info.get("dividendYield"), 0.0)
        if dividend_yield > 1.0:
            dividend_yield /= 100.0
        dividend_yield = max(0.0, min(dividend_yield, 0.25))
    except Exception:
        dividend_yield = 0.0

    return (
        symbol,
        spot,
        expiration_strings,
        dividend_yield,
        datetime.now().astimezone().isoformat(),
    )


def load_market_context(symbol: str) -> MarketContext:
    """Rebuild the page model outside Streamlit's serialization boundary."""
    (
        cached_symbol,
        spot,
        expiration_strings,
        dividend_yield,
        retrieved_at,
    ) = _load_market_context_payload(symbol)
    return MarketContext(
        symbol=cached_symbol,
        spot=spot,
        expirations=tuple(
            datetime.strptime(item, "%Y-%m-%d").date()
            for item in expiration_strings
        ),
        dividend_yield=dividend_yield,
        retrieved_at=datetime.fromisoformat(retrieved_at),
    )


@st.cache_data(ttl=900, show_spinner=False)
def load_option_chain(symbol: str, expiration_iso: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load exactly one option chain and retain the useful raw quote fields."""
    chain = yf.Ticker(symbol).option_chain(expiration_iso)
    columns = [
        "contractSymbol", "strike", "bid", "ask", "lastPrice",
        "impliedVolatility", "volume", "openInterest",
    ]

    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        available = [column for column in columns if column in frame.columns]
        result = frame.loc[:, available].copy()
        for column in columns:
            if column not in result.columns:
                result[column] = np.nan
        for column in columns[1:]:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        result = result.dropna(subset=["strike"]).drop_duplicates("strike")
        return result.sort_values("strike").reset_index(drop=True)

    return clean(chain.calls), clean(chain.puts)


# ============================================================
# BLACK-SCHOLES-MERTON PRICING AND GREEKS
# ============================================================

def intrinsic_value(spot: np.ndarray | float, strike: float, option_type: str) -> np.ndarray | float:
    values = np.asarray(spot, dtype=float)
    payoff = np.maximum(values - strike, 0.0) if option_type == "call" else np.maximum(strike - values, 0.0)
    return float(payoff) if values.ndim == 0 else payoff


def _d1_d2(
    spot: np.ndarray | float,
    strike: float,
    time_years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> tuple[np.ndarray, np.ndarray]:
    safe_spot = np.maximum(np.asarray(spot, dtype=float), 1e-12)
    safe_volatility = max(finite_float(volatility, MIN_VOLATILITY), MIN_VOLATILITY)
    root_time = math.sqrt(max(time_years, 1e-12))
    d1 = (
        np.log(safe_spot / strike)
        + (rate - dividend_yield + 0.5 * safe_volatility**2) * time_years
    ) / (safe_volatility * root_time)
    return d1, d1 - safe_volatility * root_time


def bsm_price(
    spot: np.ndarray | float,
    strike: float,
    time_years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
) -> np.ndarray | float:
    """Return the European option value per share with safe expiry behavior."""
    values = np.asarray(spot, dtype=float)
    if time_years <= 0 or volatility <= MIN_VOLATILITY:
        result = np.asarray(intrinsic_value(values, strike, option_type), dtype=float)
    else:
        d1, d2 = _d1_d2(values, strike, time_years, rate, dividend_yield, volatility)
        stock_pv = values * math.exp(-dividend_yield * time_years)
        strike_pv = strike * math.exp(-rate * time_years)
        if option_type == "call":
            result = stock_pv * norm.cdf(d1) - strike_pv * norm.cdf(d2)
        else:
            result = strike_pv * norm.cdf(-d2) - stock_pv * norm.cdf(-d1)
        result = np.maximum(result, 0.0)
    return float(result) if values.ndim == 0 else result


def bsm_greeks(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
) -> dict[str, float]:
    """Return per-share delta/gamma plus theta/day and vega/one IV point."""
    if time_years <= 0 or spot <= 0 or strike <= 0 or volatility <= MIN_VOLATILITY:
        if option_type == "call":
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    d1, d2 = _d1_d2(spot, strike, time_years, rate, dividend_yield, volatility)
    d1f, d2f = float(d1), float(d2)
    discount_q = math.exp(-dividend_yield * time_years)
    discount_r = math.exp(-rate * time_years)
    density = norm.pdf(d1f)
    common_theta = -(spot * discount_q * density * volatility) / (2.0 * math.sqrt(time_years))
    if option_type == "call":
        delta = discount_q * norm.cdf(d1f)
        theta_annual = common_theta - rate * strike * discount_r * norm.cdf(d2f) + dividend_yield * spot * discount_q * norm.cdf(d1f)
    else:
        delta = discount_q * (norm.cdf(d1f) - 1.0)
        theta_annual = common_theta + rate * strike * discount_r * norm.cdf(-d2f) - dividend_yield * spot * discount_q * norm.cdf(-d1f)
    gamma = discount_q * density / (spot * volatility * math.sqrt(time_years))
    vega_per_point = spot * discount_q * density * math.sqrt(time_years) / 100.0
    return {
        "delta": finite_float(delta),
        "gamma": finite_float(gamma),
        "theta": finite_float(theta_annual / CALENDAR_DAYS),
        "vega": finite_float(vega_per_point),
    }


# ============================================================
# IMPLIED VOLATILITY AND CHAIN NORMALIZATION
# ============================================================

def solve_implied_volatility(
    observed_price: float,
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    dividend_yield: float,
    option_type: str,
) -> float | None:
    """Invert BSM using a bounded Brent solver; return None when no valid root exists."""
    if observed_price <= 0 or time_years <= 0:
        return None
    lower_bound = float(bsm_price(spot, strike, time_years, rate, dividend_yield, MIN_VOLATILITY, option_type))
    upper_bound = float(bsm_price(spot, strike, time_years, rate, dividend_yield, MAX_VOLATILITY, option_type))
    if observed_price < lower_bound - 1e-6 or observed_price > upper_bound + 1e-6:
        return None
    try:
        return float(
            brentq(
                lambda sigma: float(bsm_price(spot, strike, time_years, rate, dividend_yield, sigma, option_type)) - observed_price,
                MIN_VOLATILITY,
                MAX_VOLATILITY,
                xtol=1e-8,
            )
        )
    except (ValueError, RuntimeError, OverflowError):
        return None


def normalize_chain(
    frame: pd.DataFrame,
    option_type: str,
    expiration: date,
    spot: float,
    rate: float,
    dividend_yield: float,
    valuation_date: date,
) -> pd.DataFrame:
    """Cache Yahoo and locally calculated IV candidates plus their deltas."""
    result = frame.copy()
    time_years = years_between(valuation_date, expiration)
    records: list[dict[str, object]] = []
    for row in result.to_dict("records"):
        bid = finite_float(row.get("bid"), 0.0)
        ask = finite_float(row.get("ask"), 0.0)
        last = finite_float(row.get("lastPrice"), 0.0)
        yahoo_iv = finite_float(row.get("impliedVolatility"), 0.0)
        valid_market = bid >= 0 and ask > 0 and ask >= bid
        if valid_market:
            observed = (bid + ask) / 2.0
            source = "midpoint"
        elif last > 0:
            observed = last
            source = "last fallback"
        else:
            fallback_iv = yahoo_iv if yahoo_iv > MIN_VOLATILITY else 0.30
            observed = float(bsm_price(spot, float(row["strike"]), time_years, rate, dividend_yield, fallback_iv, option_type))
            source = "model fallback"
        solved_iv = solve_implied_volatility(
            observed, spot, float(row["strike"]), time_years, rate, dividend_yield, option_type
        )
        yahoo_effective_iv = (
            yahoo_iv
            if yahoo_iv > MIN_VOLATILITY
            else (solved_iv if solved_iv is not None else 0.30)
        )
        calculated_effective_iv = (
            solved_iv if solved_iv is not None else yahoo_effective_iv
        )
        yahoo_greek = bsm_greeks(
            spot,
            float(row["strike"]),
            time_years,
            rate,
            dividend_yield,
            yahoo_effective_iv,
            option_type,
        )
        calculated_greek = bsm_greeks(
            spot,
            float(row["strike"]),
            time_years,
            rate,
            dividend_yield,
            calculated_effective_iv,
            option_type,
        )
        midpoint = (bid + ask) / 2.0 if valid_market else observed
        spread_ratio = (ask - bid) / midpoint if valid_market and midpoint > 0 else math.inf
        warnings: list[str] = []
        if not valid_market:
            warnings.append("incomplete market quote")
        if bid <= 0:
            warnings.append("zero bid")
        if spread_ratio > WIDE_SPREAD_THRESHOLD:
            warnings.append("wide spread")
        volume = finite_float(row.get("volume"), 0.0)
        open_interest = finite_float(row.get("openInterest"), 0.0)
        if volume < LOW_VOLUME and open_interest < LOW_OPEN_INTEREST:
            warnings.append("low activity")
        iv_discrepancy = (
            solved_iv is not None
            and yahoo_iv > MIN_VOLATILITY
            and abs(solved_iv - yahoo_iv) > 0.10
        )
        records.append(
            {
                **row,
                "bid": bid,
                "ask": ask,
                "lastPrice": last,
                "marketMid": midpoint,
                "yahooIV": yahoo_iv,
                "yahooEffectiveIV": yahoo_effective_iv,
                "calculatedIV": solved_iv if solved_iv is not None else np.nan,
                "calculatedEffectiveIV": calculated_effective_iv,
                "yahooDelta": yahoo_greek["delta"],
                "calculatedDelta": calculated_greek["delta"],
                "modelIV": yahoo_effective_iv,
                "delta": yahoo_greek["delta"],
                "volume": volume,
                "openInterest": open_interest,
                "quoteSource": source,
                "baseWarning": ", ".join(warnings),
                "ivDiscrepancy": iv_discrepancy,
                "warning": ", ".join(warnings),
                "ivSource": "Yahoo",
            }
        )
    return pd.DataFrame.from_records(records).sort_values("strike").reset_index(drop=True)


def apply_iv_source(frame: pd.DataFrame, iv_source: str) -> pd.DataFrame:
    """Activate one cached IV candidate without repeating local IV solves."""
    result = frame.copy()
    use_calculated = iv_source == "Calculated"
    result["modelIV"] = result[
        "calculatedEffectiveIV" if use_calculated else "yahooEffectiveIV"
    ]
    result["delta"] = result[
        "calculatedDelta" if use_calculated else "yahooDelta"
    ]
    result["ivSource"] = iv_source

    warnings: list[str] = []
    for row in result.to_dict("records"):
        parts = [part for part in str(row.get("baseWarning", "")).split(", ") if part]
        if use_calculated and bool(row.get("ivDiscrepancy", False)):
            parts.append("IV discrepancy")
        if use_calculated and not math.isfinite(
            finite_float(row.get("calculatedIV"), math.nan)
        ):
            parts.append("calculated IV unavailable")
        if not use_calculated and finite_float(row.get("yahooIV"), 0.0) <= MIN_VOLATILITY:
            parts.append("Yahoo IV unavailable")
        warnings.append(", ".join(parts))
    result["warning"] = warnings
    return result


@st.cache_data(ttl=900, show_spinner=False)
def load_prepared_option_chain(
    symbol: str,
    expiration_iso: str,
    spot: float,
    rate: float,
    dividend_yield: float,
    valuation_date_iso: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize one chain, including expensive local IV solves.

    This function is the boundary between market preparation and interactive
    simulation. Its arguments contain every input that can change normalized
    quotes or solved IVs. Date, IV-shock, entry-price, and chart-range controls
    are deliberately absent, so those widgets reuse this prepared result.
    """
    expiration = date.fromisoformat(expiration_iso)
    valuation_date = date.fromisoformat(valuation_date_iso)
    raw_calls, raw_puts = load_option_chain(symbol, expiration_iso)
    calls = normalize_chain(
        raw_calls,
        "call",
        expiration,
        spot,
        rate,
        dividend_yield,
        valuation_date,
    )
    puts = normalize_chain(
        raw_puts,
        "put",
        expiration,
        spot,
        rate,
        dividend_yield,
        valuation_date,
    )
    return calls, puts


def chain_for_type(chains: dict[date, dict[str, pd.DataFrame]], expiration: date, option_type: str) -> pd.DataFrame:
    return chains[expiration][option_type]


def row_for_strike(frame: pd.DataFrame, strike: float) -> pd.Series:
    index = (frame["strike"] - strike).abs().idxmin()
    return frame.loc[index]


def leg_from_chain(
    chains: dict[date, dict[str, pd.DataFrame]],
    option_type: str,
    side: int,
    expiration: date,
    strike: float,
) -> OptionLeg:
    row = row_for_strike(chain_for_type(chains, expiration, option_type), strike)
    return OptionLeg(
        option_type=option_type,
        side=side,
        quantity=1,
        expiration=expiration,
        strike=float(row["strike"]),
        bid=finite_float(row["bid"]),
        ask=finite_float(row["ask"]),
        last=finite_float(row["lastPrice"]),
        market_mid=finite_float(row["marketMid"]),
        yahoo_iv=finite_float(row["yahooIV"]),
        model_iv=max(finite_float(row["modelIV"], 0.30), MIN_VOLATILITY),
        volume=finite_float(row["volume"]),
        open_interest=finite_float(row["openInterest"]),
        calculated_iv=finite_float(row["calculatedIV"], math.nan),
        iv_source=str(row["ivSource"]),
        quote_source=str(row["quoteSource"]),
    )


# ============================================================
# STRATEGY TEMPLATES AND DEFAULT STRIKES
# ============================================================

def nearest_strike(frame: pd.DataFrame, target: float) -> float:
    """Return the listed strike closest to a requested price."""
    return float(frame.loc[(frame["strike"] - target).abs().idxmin(), "strike"])


def nearest_atm(frame: pd.DataFrame, spot: float) -> float:
    return nearest_strike(frame, spot)


def nearest_delta(frame: pd.DataFrame, target_absolute_delta: float) -> float:
    valid = frame[np.isfinite(frame["delta"])].copy()
    if valid.empty:
        return float(frame.iloc[len(frame) // 2]["strike"])
    index = (valid["delta"].abs() - target_absolute_delta).abs().idxmin()
    return float(valid.loc[index, "strike"])


def adjacent_strike(frame: pd.DataFrame, anchor: float, direction: int) -> float:
    strikes = np.asarray(sorted(frame["strike"].astype(float).unique()))
    anchor_index = int(np.argmin(np.abs(strikes - anchor)))
    target_index = min(max(anchor_index + direction, 0), len(strikes) - 1)
    return float(strikes[target_index])


def default_leg_specs(
    strategy: str,
    chains: dict[date, dict[str, pd.DataFrame]],
    front_expiration: date,
    back_expiration: date | None,
    spot: float,
) -> list[tuple[str, int, date, float]]:
    calls = chain_for_type(chains, front_expiration, "call")
    puts = chain_for_type(chains, front_expiration, "put")
    atm_call = nearest_atm(calls, spot)
    atm_put = nearest_atm(puts, spot)

    if strategy == "Long Call":
        return [("call", 1, front_expiration, atm_call)]
    if strategy == "Short Call":
        return [("call", -1, front_expiration, atm_call)]
    if strategy == "Long Put":
        return [("put", 1, front_expiration, atm_put)]
    if strategy == "Short Put":
        return [("put", -1, front_expiration, atm_put)]
    if strategy == "Bull Call Debit Spread":
        return [("call", 1, front_expiration, atm_call), ("call", -1, front_expiration, adjacent_strike(calls, atm_call, 1))]
    if strategy == "Bear Call Credit Spread":
        short_strike = nearest_delta(calls, 0.20)
        return [("call", -1, front_expiration, short_strike), ("call", 1, front_expiration, adjacent_strike(calls, short_strike, 1))]
    if strategy == "Bear Put Debit Spread":
        return [("put", 1, front_expiration, atm_put), ("put", -1, front_expiration, adjacent_strike(puts, atm_put, -1))]
    if strategy == "Bull Put Credit Spread":
        short_strike = nearest_delta(puts, 0.20)
        return [("put", -1, front_expiration, short_strike), ("put", 1, front_expiration, adjacent_strike(puts, short_strike, -1))]
    if strategy == "Short Straddle":
        shared = nearest_atm(calls, spot)
        return [("put", -1, front_expiration, shared), ("call", -1, front_expiration, shared)]
    if strategy == "Short Strangle":
        return [("put", -1, front_expiration, nearest_delta(puts, 0.20)), ("call", -1, front_expiration, nearest_delta(calls, 0.20))]
    if strategy == "Iron Condor":
        short_put = nearest_delta(puts, 0.20)
        short_call = nearest_delta(calls, 0.20)
        return [
            ("put", 1, front_expiration, adjacent_strike(puts, short_put, -1)),
            ("put", -1, front_expiration, short_put),
            ("call", -1, front_expiration, short_call),
            ("call", 1, front_expiration, adjacent_strike(calls, short_call, 1)),
        ]

    if back_expiration is None:
        raise ValueError("A back expiration is required for a calendar or diagonal.")
    back_calls = chain_for_type(chains, back_expiration, "call")
    back_puts = chain_for_type(chains, back_expiration, "put")
    if strategy == "Call Calendar / Diagonal":
        strike = nearest_delta(calls, 0.20)
        return [
            ("call", -1, front_expiration, strike),
            ("call", 1, back_expiration, nearest_strike(back_calls, strike)),
        ]
    if strategy == "Put Calendar / Diagonal":
        strike = nearest_delta(puts, 0.20)
        return [
            ("put", -1, front_expiration, strike),
            ("put", 1, back_expiration, nearest_strike(back_puts, strike)),
        ]
    if strategy == "Double Calendar / Diagonal":
        # Define the structure from the front-expiry 20-delta wings, then
        # vertically align the back legs by strike whenever the listing allows.
        put_strike = nearest_delta(puts, 0.20)
        call_strike = nearest_delta(calls, 0.20)
        return [
            ("put", -1, front_expiration, put_strike),
            ("put", 1, back_expiration, nearest_strike(back_puts, put_strike)),
            ("call", -1, front_expiration, call_strike),
            ("call", 1, back_expiration, nearest_strike(back_calls, call_strike)),
        ]
    raise ValueError(f"Unsupported strategy: {strategy}")


# ============================================================
# POSITION VALUATION, EXECUTION AND SIMULATION
# ============================================================

def execution_estimate(legs: Sequence[OptionLeg]) -> ExecutionEstimate:
    """Aggregate signed debit (+) / credit (-) prices per share."""
    natural = midpoint = favorable = 0.0
    for leg in legs:
        valid_quote = leg.ask > 0 and leg.ask >= leg.bid >= 0
        long_natural = leg.ask if valid_quote else leg.market_mid
        long_favorable = leg.bid if valid_quote else leg.market_mid
        if leg.side > 0:
            natural += leg.quantity * long_natural
            midpoint += leg.quantity * leg.market_mid
            favorable += leg.quantity * long_favorable
        else:
            natural -= leg.quantity * long_favorable
            midpoint -= leg.quantity * leg.market_mid
            favorable -= leg.quantity * long_natural
    return ExecutionEstimate(natural=natural, midpoint=midpoint, favorable=favorable)


def adjusted_volatility(leg: OptionLeg, front_expiration: date, front_points: float, back_points: float) -> float:
    adjustment = front_points if leg.expiration == front_expiration else back_points
    return min(max(leg.model_iv + adjustment / 100.0, MIN_VOLATILITY), MAX_VOLATILITY)


def average_starting_iv(
    legs: Sequence[OptionLeg], expiration: date
) -> float:
    """Return the average unshocked model IV for legs in one expiration."""
    values = [leg.model_iv for leg in legs if leg.expiration == expiration]
    return float(np.mean(values)) if values else 0.0


def position_value(
    legs: Sequence[OptionLeg],
    underlying_prices: np.ndarray | float,
    valuation_date: date,
    rate: float,
    dividend_yield: float,
    front_expiration: date,
    front_iv_points: float,
    back_iv_points: float,
) -> np.ndarray | float:
    values = np.asarray(underlying_prices, dtype=float)
    total = np.zeros_like(values)
    for leg in legs:
        leg_price = bsm_price(
            values,
            leg.strike,
            years_between(valuation_date, leg.expiration),
            rate,
            dividend_yield,
            adjusted_volatility(leg, front_expiration, front_iv_points, back_iv_points),
            leg.option_type,
        )
        total += leg.side * leg.quantity * leg.multiplier * np.asarray(leg_price)
    return float(total) if values.ndim == 0 else total


def position_greeks(
    legs: Sequence[OptionLeg],
    spot: float,
    valuation_date: date,
    rate: float,
    dividend_yield: float,
    front_expiration: date,
    front_iv_points: float,
    back_iv_points: float,
) -> dict[str, float]:
    totals = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    for leg in legs:
        greek = bsm_greeks(
            spot,
            leg.strike,
            years_between(valuation_date, leg.expiration),
            rate,
            dividend_yield,
            adjusted_volatility(leg, front_expiration, front_iv_points, back_iv_points),
            leg.option_type,
        )
        scale = leg.side * leg.quantity * leg.multiplier
        for name in totals:
            totals[name] += scale * greek[name]
    return totals


def position_secondary_greeks(
    legs: Sequence[OptionLeg],
    spot: float,
    valuation_date: date,
    rate: float,
    dividend_yield: float,
) -> dict[str, float]:
    """Return entry Vanna, Charm and Vomma in trader-friendly units.

    Vanna is delta-equivalent shares per +1 IV point, Charm is the next
    calendar day's delta change, and Vomma is dollars per IV-point squared.
    """
    totals = {"vanna": 0.0, "charm": 0.0, "vomma": 0.0}
    next_date = valuation_date + timedelta(days=1)
    for leg in legs:
        time_years = years_between(valuation_date, leg.expiration)
        if time_years <= 0.0:
            continue
        volatility = max(leg.model_iv, MIN_VOLATILITY)
        d1, d2 = _d1_d2(
            spot,
            leg.strike,
            time_years,
            rate,
            dividend_yield,
            volatility,
        )
        d1_value, d2_value = float(d1), float(d2)
        discount_q = math.exp(-dividend_yield * time_years)
        density = norm.pdf(d1_value)
        raw_vega = spot * discount_q * density * math.sqrt(time_years)
        vanna_per_point = -discount_q * density * d2_value / volatility / 100.0
        vomma_per_point_squared = (
            raw_vega * d1_value * d2_value / volatility / 10_000.0
        )

        delta_now = bsm_greeks(
            spot,
            leg.strike,
            time_years,
            rate,
            dividend_yield,
            volatility,
            leg.option_type,
        )["delta"]
        delta_next_day = bsm_greeks(
            spot,
            leg.strike,
            years_between(next_date, leg.expiration),
            rate,
            dividend_yield,
            volatility,
            leg.option_type,
        )["delta"]

        scale = leg.side * leg.quantity * leg.multiplier
        totals["vanna"] += scale * vanna_per_point
        totals["charm"] += scale * (delta_next_day - delta_now)
        totals["vomma"] += scale * vomma_per_point_squared
    return totals


def vega_by_expiration(
    legs: Sequence[OptionLeg],
    spot: float,
    valuation_date: date,
    rate: float,
    dividend_yield: float,
    front_expiration: date,
) -> tuple[float, float | None]:
    """Return signed position Vega for front and back expiration groups."""
    front_vega = 0.0
    back_vega = 0.0
    has_back_legs = False
    for leg in legs:
        leg_vega = bsm_greeks(
            spot,
            leg.strike,
            years_between(valuation_date, leg.expiration),
            rate,
            dividend_yield,
            leg.model_iv,
            leg.option_type,
        )["vega"] * leg.side * leg.quantity * leg.multiplier
        if leg.expiration == front_expiration:
            front_vega += leg_vega
        else:
            back_vega += leg_vega
            has_back_legs = True
    return front_vega, back_vega if has_back_legs else None


def capital_at_risk(
    strategy: str,
    legs: Sequence[OptionLeg],
    entry_per_share: float,
    rate: float,
    dividend_yield: float,
    front_expiration: date,
) -> float | None:
    """Return fixed or modeled loss capital; None denotes unlimited risk."""
    if strategy in {"Short Call", "Short Straddle", "Short Strangle"}:
        return None
    high_price = max(leg.strike for leg in legs) * 3.0
    grid = np.unique(
        np.concatenate(
            (np.linspace(0.0, high_price, 6001), np.asarray([leg.strike for leg in legs]))
        )
    )
    pnl = np.asarray(
        position_value(
            legs,
            grid,
            front_expiration,
            rate,
            dividend_yield,
            front_expiration,
            0.0,
            0.0,
        )
    ) - entry_per_share * CONTRACT_MULTIPLIER
    return max(-float(np.min(pnl)), 0.0)


def safe_ratio(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def calculate_advanced_metrics(
    strategy: str,
    legs: Sequence[OptionLeg],
    spot: float,
    entry_per_share: float,
    entry_date: date,
    selected_date: date,
    rate: float,
    dividend_yield: float,
    front_expiration: date,
    front_iv_points: float,
    back_iv_points: float,
    entry_greeks: dict[str, float],
) -> AdvancedMetrics:
    """Calculate transparent entry ratios and user-controlled P&L scenarios."""
    secondary = position_secondary_greeks(
        legs, spot, entry_date, rate, dividend_yield
    )
    front_vega, back_vega = vega_by_expiration(
        legs, spot, entry_date, rate, dividend_yield, front_expiration
    )
    risk_capital = capital_at_risk(
        strategy, legs, entry_per_share, rate, dividend_yield, front_expiration
    )
    vega_to_risk = safe_ratio(entry_greeks["vega"] * 100.0, risk_capital)
    theta_to_risk = safe_ratio(entry_greeks["theta"] * 100.0, risk_capital)
    gamma_move_risk = 0.5 * abs(entry_greeks["gamma"]) * (spot * 0.01) ** 2
    theta_to_gamma = safe_ratio(entry_greeks["theta"], gamma_move_risk)
    back_front_ratio = (
        safe_ratio(abs(back_vega), abs(front_vega))
        if back_vega is not None
        else None
    )

    entry_dollars = entry_per_share * CONTRACT_MULTIPLIER
    center_price = float(np.mean([leg.strike for leg in legs]))

    def pnl_at_center(
        valuation_date: date,
        front_points: float,
        back_points: float,
    ) -> float:
        return float(
            position_value(
                legs,
                center_price,
                valuation_date,
                rate,
                dividend_yield,
                front_expiration,
                front_points,
                back_points,
            )
        ) - entry_dollars

    center_pnl = pnl_at_center(selected_date, front_iv_points, back_iv_points)

    front_ivs = [leg.model_iv for leg in legs if leg.expiration == front_expiration]
    back_ivs = [leg.model_iv for leg in legs if leg.expiration != front_expiration]
    front_iv = float(np.mean(front_ivs)) if front_ivs else 0.0
    expected_move = spot * front_iv * math.sqrt(
        years_between(entry_date, front_expiration)
    )
    move_grid = np.linspace(
        max(0.01, spot - expected_move),
        spot + expected_move,
        301,
    )
    move_pnl = np.asarray(
        position_value(
            legs,
            move_grid,
            front_expiration,
            rate,
            dividend_yield,
            front_expiration,
            front_iv_points,
            back_iv_points,
        )
    ) - entry_dollars
    pnl_through_move = float(np.min(move_pnl))

    crush_capture: float | None = None
    back_iv_breakeven = "—"
    raw_iv_gap: float | None = None
    if back_vega is not None and back_ivs:
        base_center = pnl_at_center(entry_date, 0.0, 0.0)
        front_crush_center = pnl_at_center(
            entry_date, -STANDARD_CRUSH_POINTS, 0.0
        )
        back_crush_center = pnl_at_center(
            entry_date, 0.0, -STANDARD_CRUSH_POINTS
        )
        front_crush_benefit = front_crush_center - base_center
        back_crush_damage = base_center - back_crush_center
        crush_capture = safe_ratio(front_crush_benefit, back_crush_damage)
        raw_iv_gap = (front_iv - float(np.mean(back_ivs))) * 100.0

        baseline_front_expiry = pnl_at_center(front_expiration, 0.0, 0.0)
        maximum_drop = max(min(back_ivs) * 100.0 - 0.01, 0.0)
        if baseline_front_expiry <= 0.0 or maximum_drop <= 0.0:
            back_iv_breakeven = "0.0 pt"
        else:
            pnl_after_maximum_drop = pnl_at_center(
                front_expiration, 0.0, -maximum_drop
            )
            if pnl_after_maximum_drop > 0.0:
                back_iv_breakeven = f">{maximum_drop:.1f} pt"
            else:
                solved_drop = brentq(
                    lambda drop: pnl_at_center(front_expiration, 0.0, -drop),
                    0.0,
                    maximum_drop,
                )
                back_iv_breakeven = f"{solved_drop:.1f} pt"

    return AdvancedMetrics(
        front_vega=front_vega,
        back_vega=back_vega,
        capital_at_risk=risk_capital,
        vega_to_risk_percent=vega_to_risk,
        theta_to_risk_percent=theta_to_risk,
        theta_to_gamma_risk=theta_to_gamma,
        back_to_front_vega=back_front_ratio,
        crush_capture_ratio=crush_capture,
        back_iv_crush_breakeven=back_iv_breakeven,
        vanna=secondary["vanna"],
        charm=secondary["charm"],
        vomma=secondary["vomma"],
        center_pnl_after_shock=center_pnl,
        pnl_through_expected_move=pnl_through_move,
        raw_iv_gap_points=raw_iv_gap,
    )


def breakevens(prices: np.ndarray, pnl: np.ndarray) -> list[float]:
    roots: list[float] = []
    for index in range(len(prices) - 1):
        left, right = pnl[index], pnl[index + 1]
        if left == 0:
            roots.append(float(prices[index]))
        elif left * right < 0:
            weight = abs(left) / (abs(left) + abs(right))
            roots.append(float(prices[index] + weight * (prices[index + 1] - prices[index])))
    return roots


def expiration_breakevens(
    legs: Sequence[OptionLeg],
    spot: float,
    entry_per_share: float,
    rate: float,
    dividend_yield: float,
    front_expiration: date,
) -> list[float]:
    """Find expiry breakevens on a broad grid independent of chart range."""
    highest_reference = max(spot, *(leg.strike for leg in legs))
    scan_prices = np.linspace(0.01, highest_reference * 3.0, 8001)
    expiry_pnl = np.asarray(
        position_value(
            legs,
            scan_prices,
            front_expiration,
            rate,
            dividend_yield,
            front_expiration,
            0.0,
            0.0,
        )
    ) - entry_per_share * CONTRACT_MULTIPLIER
    return breakevens(scan_prices, expiry_pnl)


def default_chart_range_percent(
    center_price: float,
    break_even_points: Sequence[float],
    outer_padding_percent: float = 3.0,
) -> float:
    """Return a symmetric range that clears the outer breakevens by ~3%."""
    if not break_even_points:
        return outer_padding_percent
    padded_low = min(break_even_points) * (1.0 - outer_padding_percent / 100.0)
    padded_high = max(break_even_points) * (1.0 + outer_padding_percent / 100.0)
    required_percent = max(
        (center_price - padded_low) / center_price * 100.0,
        (padded_high - center_price) / center_price * 100.0,
        0.25,
    )
    return math.ceil(required_percent * 4.0) / 4.0


def risk_summary(
    strategy: str,
    legs: Sequence[OptionLeg],
    entry_per_share: float,
    rate: float,
    dividend_yield: float,
    front_expiration: date,
) -> tuple[str, str, str]:
    """Return max profit, max loss and the scope label."""
    entry_dollars = entry_per_share * CONTRACT_MULTIPLIER
    if strategy in TIME_SPREAD_STRATEGIES:
        strikes = [leg.strike for leg in legs]
        high = max(strikes) * 2.5
        grid = np.linspace(0.01, high, 2401)
        pnl = np.asarray(position_value(legs, grid, front_expiration, rate, dividend_yield, front_expiration, 0.0, 0.0)) - entry_dollars
        return money(float(np.max(pnl))), money(abs(float(np.min(pnl)))), "Modeled at front expiration"
    if strategy in {"Long Call"}:
        return "Unlimited", money(max(entry_dollars, 0.0)), "Theoretical"
    if strategy in {"Short Call", "Short Straddle", "Short Strangle"}:
        return money(max(-entry_dollars, 0.0)), "Unlimited", "Theoretical"

    highest_strike = max(leg.strike for leg in legs)
    grid = np.unique(np.concatenate((np.linspace(0.0, highest_strike * 3.0, 6001), np.asarray([leg.strike for leg in legs]))))
    expiry_value = np.asarray(position_value(legs, grid, front_expiration, rate, dividend_yield, front_expiration, 0.0, 0.0))
    pnl = expiry_value - entry_dollars
    return money(float(np.max(pnl))), money(abs(float(np.min(pnl)))), "Theoretical"


# ============================================================
# PLOTLY CHART
# ============================================================

def loss_fill_polygons(
    prices: np.ndarray,
    theoretical_pnl: np.ndarray,
    expiry_pnl: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build independent red polygons where theoretical is below expiration.

    Plotly's ``tonexty`` fill can join disconnected regions across NaN gaps,
    producing large diagonal wedges. Returning one closed polygon per
    contiguous region keeps every red shape bounded by the two P&L curves.
    """
    negative_expiry_boundary = np.minimum(expiry_pnl, 0.0)
    band_thickness = negative_expiry_boundary - theoretical_pnl
    active = band_thickness > 1e-9
    polygons: list[tuple[np.ndarray, np.ndarray]] = []

    index = 0
    while index < len(prices):
        if not active[index]:
            index += 1
            continue

        start = index
        while index + 1 < len(prices) and active[index + 1]:
            index += 1
        end = index

        segment_x = prices[start : end + 1].astype(float).tolist()
        lower_y = theoretical_pnl[start : end + 1].astype(float).tolist()
        upper_y = negative_expiry_boundary[start : end + 1].astype(float).tolist()

        # Extend each segment to the exact zero-width intersection when the
        # band begins or ends between adjacent price-grid points.
        if start > 0 and band_thickness[start - 1] <= 0.0:
            left_thickness = band_thickness[start - 1]
            right_thickness = band_thickness[start]
            fraction = -left_thickness / (right_thickness - left_thickness)
            crossing_x = prices[start - 1] + fraction * (prices[start] - prices[start - 1])
            lower_crossing = theoretical_pnl[start - 1] + fraction * (
                theoretical_pnl[start] - theoretical_pnl[start - 1]
            )
            upper_crossing = negative_expiry_boundary[start - 1] + fraction * (
                negative_expiry_boundary[start] - negative_expiry_boundary[start - 1]
            )
            crossing_y = (lower_crossing + upper_crossing) / 2.0
            segment_x.insert(0, float(crossing_x))
            lower_y.insert(0, float(crossing_y))
            upper_y.insert(0, float(crossing_y))

        if end + 1 < len(prices) and band_thickness[end + 1] <= 0.0:
            left_thickness = band_thickness[end]
            right_thickness = band_thickness[end + 1]
            fraction = left_thickness / (left_thickness - right_thickness)
            crossing_x = prices[end] + fraction * (prices[end + 1] - prices[end])
            lower_crossing = theoretical_pnl[end] + fraction * (
                theoretical_pnl[end + 1] - theoretical_pnl[end]
            )
            upper_crossing = negative_expiry_boundary[end] + fraction * (
                negative_expiry_boundary[end + 1] - negative_expiry_boundary[end]
            )
            crossing_y = (lower_crossing + upper_crossing) / 2.0
            segment_x.append(float(crossing_x))
            lower_y.append(float(crossing_y))
            upper_y.append(float(crossing_y))

        polygon_x = np.asarray(segment_x + segment_x[::-1], dtype=float)
        polygon_y = np.asarray(lower_y + upper_y[::-1], dtype=float)
        polygons.append((polygon_x, polygon_y))
        index += 1

    return polygons


def shared_price_ticks(
    lower: float,
    upper: float,
    target_intervals: int = 8,
) -> np.ndarray:
    """Return readable major ticks shared by the price and percent axes."""
    span = upper - lower
    if not np.isfinite(span) or span <= 1e-12:
        return np.asarray([lower], dtype=float)

    raw_step = span / max(target_intervals, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    normalized_step = raw_step / magnitude
    nice_fraction = min(
        (1.0, 2.0, 2.5, 5.0, 10.0),
        key=lambda candidate: abs(candidate - normalized_step),
    )
    step = nice_fraction * magnitude
    first = math.ceil((lower - step * 1e-9) / step) * step
    last = math.floor((upper + step * 1e-9) / step) * step
    if first > last:
        return np.asarray([(lower + upper) / 2.0], dtype=float)

    count = int(round((last - first) / step)) + 1
    return first + np.arange(count, dtype=float) * step


def spot_percentage_ticks(
    lower_price: float,
    upper_price: float,
    spot: float,
    target_intervals: int = 8,
) -> tuple[np.ndarray, list[str]]:
    """Return percentage ticks whose zero point is exactly the spot price."""
    lower_percent = ((lower_price / spot) - 1.0) * 100.0
    upper_percent = ((upper_price / spot) - 1.0) * 100.0
    percent_span = upper_percent - lower_percent
    raw_step = percent_span / max(target_intervals, 1)
    magnitude = 10.0 ** math.floor(math.log10(max(raw_step, 1e-9)))
    normalized_step = raw_step / magnitude
    nice_fraction = min(
        (1.0, 2.0, 2.5, 5.0, 10.0),
        key=lambda candidate: abs(candidate - normalized_step),
    )
    percent_step = nice_fraction * magnitude

    first_multiple = math.ceil(
        (lower_percent - percent_step * 1e-9) / percent_step
    )
    last_multiple = math.floor(
        (upper_percent + percent_step * 1e-9) / percent_step
    )
    multiples = np.arange(first_multiple, last_multiple + 1, dtype=float)
    percent_values = multiples * percent_step
    price_values = spot * (1.0 + percent_values / 100.0)

    decimals = 2 if percent_step < 0.1 else 1
    labels = [
        "0%" if abs(value) < 1e-9 else f"{value:+.{decimals}f}%"
        for value in percent_values
    ]
    return price_values, labels


def build_pnl_figure(
    prices: np.ndarray,
    selected_pnl: np.ndarray,
    expiry_pnl: np.ndarray,
    spot: float,
    selected_date: date,
    curve_expiration: date,
    break_even_points: Sequence[float],
    is_time_spread: bool,
    legs: Sequence[OptionLeg] = (),
    pnl_percentage_basis: float | None = None,
) -> go.Figure:
    figure = go.Figure()

    # Shade from zero toward the selected-date theoretical curve, but cap the
    # fill at the expiration curve. If the curves are on opposite sides of
    # zero, there is no shared profit/loss region to shade at that price.
    profit_fill = np.minimum(
        np.maximum(selected_pnl, 0.0),
        np.maximum(expiry_pnl, 0.0),
    )
    figure.add_trace(
        go.Scatter(
            x=prices,
            y=profit_fill,
            mode="lines",
            line={"width": 0},
            fill="tozeroy",
            fillcolor="rgba(34, 197, 94, 0.5)",
            hoverinfo="skip",
            showlegend=False,
            name="Capped theoretical profit area",
        )
    )
    for polygon_x, polygon_y in loss_fill_polygons(prices, selected_pnl, expiry_pnl):
        figure.add_trace(
            go.Scatter(
                x=polygon_x,
                y=polygon_y,
                mode="lines",
                line={"width": 0},
                fill="toself",
                fillcolor="rgba(239, 68, 68, 0.5)",
                hoverinfo="skip",
                showlegend=False,
                name="Loss between curves",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=prices,
            y=expiry_pnl,
            mode="lines",
            name=("Front-expiry P&L" if is_time_spread else "Expiration P&L"),
            line={"color": "#2563eb", "width": 3},
            hovertemplate="Underlying $%{x:,.2f}<br>P&L $%{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=prices,
            y=selected_pnl,
            mode="lines",
            name=f"{selected_date:%b %d} theoretical",
            line={"color": "#000000", "width": 2.5, "dash": "dash"},
            hovertemplate="Underlying $%{x:,.2f}<br>P&L $%{y:,.0f}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_width=1, line_color="#94a3b8")
    figure.add_vline(
        x=spot,
        line_width=1.5,
        line_dash="dot",
        line_color="#64748b",
        annotation_text=f"Spot ${spot:,.2f}",
        annotation_position="top right",
        annotation_bgcolor="rgba(100, 116, 139, 0.5)",
        annotation_font_color="#ffffff",
        annotation_borderpad=4,
    )
    for point in break_even_points:
        percent_from_spot = ((point / spot) - 1.0) * 100.0
        figure.add_vline(
            x=point,
            line_width=1,
            line_dash="dot",
            line_color="#16a34a",
            annotation_text=f"BE ${point:,.2f} ({percent_from_spot:+.1f}%)",
            annotation_position="top left",
            annotation_bgcolor="rgba(22, 163, 74, 0.5)",
            annotation_font_color="#ffffff",
            annotation_borderpad=4,
        )

    x_min, x_max = float(np.min(prices)), float(np.max(prices))
    price_tick_values = shared_price_ticks(x_min, x_max)
    # Anchor the percentage scale at spot so that its intervals radiate from a
    # clearly defined 0%. Each tick position remains an exact price mapping.
    percentage_tick_prices, x_percent_labels = spot_percentage_ticks(
        x_min, x_max, spot
    )

    plotted_pnl = np.concatenate((selected_pnl, expiry_pnl, np.asarray([0.0])))
    y_min, y_max = float(np.min(plotted_pnl)), float(np.max(plotted_pnl))
    y_span = y_max - y_min
    if y_span <= 1e-9:
        y_span = max(abs(y_min), 1.0)
    y_padding = y_span * 0.08
    y_range = [y_min - y_padding, y_max + y_padding]

    # Directional triangles point toward the zero line: short legs sit just
    # above it and long legs just below it. One trace per side keeps the legend
    # compact even for four-leg structures.
    marker_offset = y_span * 0.025
    for side, name, y_value, symbol, color in (
        (-1, "Short strikes", marker_offset, "triangle-down", "#dc2626"),
        (1, "Long strikes", -marker_offset, "triangle-up", "#16a34a"),
    ):
        side_legs = [leg for leg in legs if leg.side == side]
        if not side_legs:
            continue
        figure.add_trace(
            go.Scatter(
                x=[leg.strike for leg in side_legs],
                y=[y_value] * len(side_legs),
                mode="markers",
                name=name,
                marker={
                    "symbol": symbol,
                    "size": 18,
                    "color": color,
                    "line": {"color": "#ffffff", "width": 1},
                },
                customdata=[
                    f"{leg.label} · {leg.expiration:%b %d, %Y}"
                    for leg in side_legs
                ],
                hovertemplate=(
                    "%{customdata}<br>Strike $%{x:,.2f}<extra></extra>"
                ),
            )
        )

    percentage_axis: dict[str, object] = {"visible": False}
    if pnl_percentage_basis is not None and abs(pnl_percentage_basis) > 1e-9:
        percentage_basis = abs(pnl_percentage_basis)
        percentage_range = [
            y_range[0] / percentage_basis * 100.0,
            y_range[1] / percentage_basis * 100.0,
        ]
        percentage_tick_values = np.linspace(
            percentage_range[0], percentage_range[1], 9
        )
        percentage_axis = {
            "anchor": "x",
            "overlaying": "y",
            "side": "right",
            "range": percentage_range,
            "tickmode": "array",
            "tickvals": percentage_tick_values,
            "ticktext": [f"{value:+.0f}%" for value in percentage_tick_values],
            "title": {"text": "% P/L", "standoff": 8},
            "showgrid": False,
            "showline": True,
            "linecolor": "#64748b",
            "ticks": "outside",
            "ticklen": 6,
            "tickwidth": 1.2,
            "tickcolor": "#64748b",
        }

    figure.update_layout(
        height=540,
        margin={"l": 20, "r": 75, "t": 85, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.22, "x": 0},
        xaxis_title="Underlying price",
        yaxis_title="Position P&L ($)",
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "gridcolor": "#eef2f7",
            "showline": True,
            "linecolor": "#64748b",
            "ticks": "outside",
            "ticklen": 6,
            "tickwidth": 1.2,
            "tickcolor": "#64748b",
            "tickmode": "array",
            "tickvals": price_tick_values,
            "range": [x_min, x_max],
        },
        xaxis2={
            "anchor": "y",
            "overlaying": "x",
            "matches": "x",
            "side": "top",
            "range": [x_min, x_max],
            "tickmode": "array",
            "tickvals": percentage_tick_prices,
            "ticktext": x_percent_labels,
            "title": {"text": "% from spot", "standoff": 8},
            "showgrid": False,
            "showline": True,
            "linecolor": "#64748b",
            "ticks": "outside",
            "ticklen": 6,
            "tickwidth": 1.2,
            "tickcolor": "#64748b",
        },
        yaxis={
            "gridcolor": "#eef2f7",
            "zeroline": False,
            "showline": True,
            "linecolor": "#64748b",
            "ticks": "outside",
            "ticklen": 6,
            "tickwidth": 1.2,
            "tickcolor": "#64748b",
            "nticks": 9,
            "range": y_range,
        },
        yaxis2=percentage_axis,
    )
    # Plotly only guarantees an overlaid axis is drawn when a trace references
    # it. Transparent endpoints keep both percentage axes reliably present
    # without changing autorange, hover behavior, or the visible curves.
    if pnl_percentage_basis is not None and abs(pnl_percentage_basis) > 1e-9:
        figure.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[percentage_range[0], percentage_range[1]],
                xaxis="x2",
                yaxis="y2",
                mode="markers",
                marker={"opacity": 0},
                hoverinfo="skip",
                showlegend=False,
                name="Percentage-axis scale",
            )
        )
    return figure


# ============================================================
# STRIKE SELECTOR AND DISPLAY HELPERS
# ============================================================

def synchronize_anchored_strike(
    source_widget_key: str,
    source_canonical_key: str,
    source_query_key: str,
    target_widget_key: str,
    target_canonical_key: str,
    target_query_key: str,
    target_strikes: tuple[float, ...],
    anchor_state_key: str,
) -> None:
    """Synchronize a paired strike before Streamlit begins its rerun."""
    source_strike = float(st.session_state[source_widget_key])
    st.session_state[source_canonical_key] = source_strike
    persist_query_value(source_query_key, f"{source_strike:g}")
    if not st.session_state.get(anchor_state_key, False):
        return

    target_strike = min(
        target_strikes, key=lambda strike: abs(strike - source_strike)
    )
    st.session_state[target_canonical_key] = target_strike
    st.session_state[target_widget_key] = target_strike
    persist_query_value(target_query_key, f"{target_strike:g}")


def render_strike_selector(
    leg: OptionLeg,
    leg_index: int,
    frame: pd.DataFrame,
    context_key: str,
    view_mode: str,
    initial_strike: float,
    anchor_target_index: int | None = None,
    anchor_target_strikes: tuple[float, ...] = (),
    anchor_state_key: str = "",
) -> float:
    all_strikes = sorted(frame["strike"].astype(float).unique())
    option_labels: dict[float, str] = {}
    for strike in all_strikes:
        option_row = row_for_strike(frame, strike)
        option_labels[strike] = (
            f"${strike:,.2f}  ·  Δ {finite_float(option_row['delta']):+.2f}"
            f"  ·  IV {signed_iv(finite_float(option_row['modelIV']))}"
        )
    canonical_key = f"strike_value_{context_key}_{leg_index}"
    mode_key = f"strike_mode_{context_key}_{leg_index}"
    if canonical_key not in st.session_state:
        st.session_state[canonical_key] = min(
            all_strikes, key=lambda item: abs(item - initial_strike)
        )
    elif float(st.session_state[canonical_key]) not in all_strikes:
        st.session_state[canonical_key] = min(
            all_strikes,
            key=lambda item: abs(item - float(st.session_state[canonical_key])),
        )

    widget_kind = "list" if view_mode == "List view" else "slider"
    widget_key = f"strike_{widget_kind}_{context_key}_{leg_index}"
    rebuild_widget = (
        widget_key not in st.session_state
        or st.session_state.get(mode_key) != view_mode
        or (
            widget_key in st.session_state
            and float(st.session_state[widget_key]) not in all_strikes
        )
    )
    if rebuild_widget and widget_key in st.session_state:
        # Let the widget receive an explicit default below. Preloading its key
        # and omitting index/value can race with Streamlit's widget restoration
        # and silently select the first (lowest) strike.
        del st.session_state[widget_key]
    st.session_state[mode_key] = view_mode
    canonical_strike = float(st.session_state[canonical_key])
    change_callback: dict[str, object] = {}
    if anchor_target_index is not None and anchor_target_strikes:
        target_widget_key = (
            f"strike_{widget_kind}_{context_key}_{anchor_target_index}"
        )
        change_callback = {
            "on_change": synchronize_anchored_strike,
            "args": (
                widget_key,
                canonical_key,
                f"strike_{leg_index}",
                target_widget_key,
                f"strike_value_{context_key}_{anchor_target_index}",
                f"strike_{anchor_target_index}",
                anchor_target_strikes,
                anchor_state_key,
            ),
        }

    # Render the heading in its final position before creating the widget.
    # A prior st.empty() placeholder was filled only after the selector was
    # created, causing the control row to shift visibly on every app rerun.
    displayed_strike = (
        canonical_strike
        if rebuild_widget
        else float(st.session_state.get(widget_key, canonical_strike))
    )
    displayed_row = row_for_strike(frame, displayed_strike)
    displayed_warning_value = displayed_row.get("warning", "")
    displayed_warning = (
        str(displayed_warning_value).strip()
        if pd.notna(displayed_warning_value)
        else ""
    )
    displayed_icon = " ⚠" if displayed_warning else ""
    st.markdown(
        f"**{leg.label} · {leg.expiration:%b %d, %Y}{displayed_icon}**"
    )

    if view_mode == "List view":
        list_defaults = (
            {"index": all_strikes.index(canonical_strike)}
            if rebuild_widget
            else {}
        )
        selected = st.selectbox(
            "Strike",
            options=all_strikes,
            format_func=lambda item: option_labels[item],
            key=widget_key,
            label_visibility="collapsed",
            **list_defaults,
            **change_callback,
        )
    else:
        slider_defaults = (
            {"value": canonical_strike} if rebuild_widget else {}
        )
        selected = st.select_slider(
            "Strike",
            options=all_strikes,
            format_func=lambda item: option_labels[item],
            key=widget_key,
            label_visibility="collapsed",
            **slider_defaults,
            **change_callback,
        )
    selected = float(selected)
    st.session_state[canonical_key] = selected
    row = row_for_strike(frame, selected)
    warning_value = row.get("warning", "")
    warning = str(warning_value).strip() if pd.notna(warning_value) else ""
    bid_text = price_text(finite_float(row["bid"])).replace("$", r"\$")
    ask_text = price_text(finite_float(row["ask"])).replace("$", r"\$")
    detail = (
        f"Bid / ask  {bid_text} / {ask_text}"
    )
    if warning:
        detail += f"  ·  ⚠ {warning}"
    st.caption(detail)
    return selected


def quote_warnings(leg: OptionLeg) -> list[str]:
    warnings: list[str] = []
    if leg.bid <= 0:
        warnings.append("zero bid")
    if leg.ask <= 0 or leg.ask < leg.bid:
        warnings.append("incomplete quote")
    midpoint = (leg.bid + leg.ask) / 2.0
    if midpoint > 0 and (leg.ask - leg.bid) / midpoint > WIDE_SPREAD_THRESHOLD:
        warnings.append("wide spread")
    if leg.volume < LOW_VOLUME and leg.open_interest < LOW_OPEN_INTEREST:
        warnings.append("low activity")
    if leg.iv_source == "Calculated":
        if not math.isfinite(leg.calculated_iv):
            warnings.append("calculated IV unavailable")
        elif (
            leg.yahoo_iv > MIN_VOLATILITY
            and abs(leg.calculated_iv - leg.yahoo_iv) > 0.10
        ):
            warnings.append("IV discrepancy")
    elif leg.yahoo_iv <= MIN_VOLATILITY:
        warnings.append("Yahoo IV unavailable")
    if leg.quote_source != "midpoint":
        warnings.append(leg.quote_source)
    return warnings


def optional_metric(value: float | None, template: str) -> str:
    return "—" if value is None or not math.isfinite(value) else template.format(value)


def metric_rows_html(rows: Sequence[tuple[str, str]]) -> str:
    return "".join(
        f'<div class="metric-readout-row"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in rows
    )


def advanced_metric_groups(
    advanced: AdvancedMetrics,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Format advanced metrics once for both available presentation layouts."""
    secondary_greeks = [
        ("Vanna / IV pt", f"{advanced.vanna:+,.3f}"),
        ("Charm / day", f"{advanced.charm:+,.3f}"),
        ("Vomma / IV pt²", f"${advanced.vomma:+,.4f}"),
    ]
    ratios = [
        ("Capital at risk", optional_metric(advanced.capital_at_risk, "${:,.0f}")),
        ("Front Vega / pt", f"${advanced.front_vega:+,.2f}"),
        ("Back Vega / pt", optional_metric(advanced.back_vega, "${:+,.2f}")),
        ("Vega / risk", optional_metric(advanced.vega_to_risk_percent, "{:+.3f}%")),
        ("Theta / risk", optional_metric(advanced.theta_to_risk_percent, "{:+.3f}%/day")),
        ("Theta / 1% Γ risk", optional_metric(advanced.theta_to_gamma_risk, "{:+.2f}×")),
        ("Back / front Vega", optional_metric(advanced.back_to_front_vega, "{:.2f}×")),
        (f"{STANDARD_CRUSH_POINTS:g}-pt crush capture", optional_metric(advanced.crush_capture_ratio, "{:.2f}×")),
        ("Back-IV crush BE", advanced.back_iv_crush_breakeven),
    ]
    scenarios = [
        ("Center P&L shocked", money(advanced.center_pnl_after_shock)),
        ("Min P&L ± exp move", money(advanced.pnl_through_expected_move)),
        ("Front − back IV", optional_metric(advanced.raw_iv_gap_points, "{:+.1f} pt")),
    ]
    return [
        ("", secondary_greeks),
        ("EFFICIENCY & VEGA RATIOS", ratios),
        ("SCENARIOS", scenarios),
    ]


def render_advanced_metric_sections(advanced: AdvancedMetrics) -> None:
    sections = "".join(
        (
            f'<div class="metric-readout-heading metric-readout-greeks">{heading}</div>'
            if heading
            else ""
        )
        + metric_rows_html(rows)
        for heading, rows in advanced_metric_groups(advanced)
    )
    st.markdown(
        f'<div class="metric-readout">{sections}</div>',
        unsafe_allow_html=True,
    )


def render_right_metrics_panel(
    entry_per_share: float,
    execution: ExecutionEstimate,
    max_profit: str,
    max_loss: str,
    greeks: dict[str, float],
    advanced: AdvancedMetrics,
) -> None:
    """Render a compact vertical readout without Streamlit metric cards."""
    debit_credit = "Debit" if entry_per_share >= 0 else "Credit"
    risk_and_execution = [
        (f"Net {debit_credit}", execution_range_text(execution)),
        ("Max profit", max_profit),
        ("Max loss", max_loss),
    ]
    greek_values = [
        ("Delta", f"{greeks['delta']:+,.1f}"),
        ("Gamma", f"{greeks['gamma']:+,.3f}"),
        ("Theta / day", f"${greeks['theta']:+,.2f}"),
        ("Vega / IV pt", f"${greeks['vega']:+,.2f}"),
    ]

    st.markdown(
        f"""
        <div class="metric-readout">
            <div class="metric-readout-heading">RISK &amp; EXECUTION</div>
            {metric_rows_html(risk_and_execution)}
            <div class="metric-readout-heading metric-readout-greeks">ENTRY GREEKS</div>
            {metric_rows_html(greek_values)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_advanced_metric_sections(advanced)


def render_legacy_metric_grid(
    entry_per_share: float,
    execution: ExecutionEstimate,
    max_profit: str,
    max_loss: str,
    greeks: dict[str, float],
    advanced: AdvancedMetrics,
) -> None:
    """Render the original metric-card layout as the instant fallback view."""
    debit_credit = "Debit" if entry_per_share >= 0 else "Credit"
    metric_columns = st.columns(3)
    metric_columns[0].metric(
        f"Net {debit_credit}", execution_range_text(execution)
    )
    metric_columns[1].metric("Max profit", max_profit)
    metric_columns[2].metric("Max loss", max_loss)

    greek_columns = st.columns(4)
    greek_columns[0].metric("Delta", f"{greeks['delta']:+,.1f}")
    greek_columns[1].metric("Gamma", f"{greeks['gamma']:+,.3f}")
    greek_columns[2].metric("Theta / day", f"${greeks['theta']:+,.2f}")
    greek_columns[3].metric("Vega / IV pt", f"${greeks['vega']:+,.2f}")
    render_advanced_metric_sections(advanced)


# ============================================================
# APPLICATION UI
# ============================================================

def app_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: none;
            padding-left: clamp(1.25rem, 2.5vw, 3rem);
            padding-right: clamp(1.25rem, 2.5vw, 3rem);
            padding-top: 2rem;
            width: 100%;
        }
        [data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: .7rem .85rem;}
        [data-testid="stMetricLabel"] {font-size: .78rem; color: #64748b;}
        [data-testid="stMetricValue"] {font-size: 1.25rem;}
        .sim-subtle {color: #64748b; font-size: .9rem;}
        .metric-readout {margin-top: .25rem;}
        .metric-readout-heading {
            color: #64748b;
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .08em;
            margin: .4rem 0 .2rem;
        }
        .metric-readout-greeks {margin-top: 1.2rem;}
        .metric-readout-row {
            align-items: baseline;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            font-size: .86rem;
            gap: .5rem;
            justify-content: space-between;
            padding: .42rem 0;
        }
        .metric-readout-row span {color: #64748b;}
        .metric-readout-row strong {font-size: .92rem; text-align: right;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    app_styles()

    persisted_symbol = query_text("ticker", "SPY").strip().upper() or "SPY"
    with st.sidebar:
        st.header("Position setup")
        with st.form("symbol_form", border=False):
            requested_symbol = st.text_input(
                "Ticker",
                value=st.session_state.get("loaded_symbol", persisted_symbol),
            ).strip().upper()
            load_clicked = st.form_submit_button("Load ticker", type="primary", use_container_width=True)
        if load_clicked or "loaded_symbol" not in st.session_state:
            st.session_state["loaded_symbol"] = requested_symbol or persisted_symbol
        symbol = str(st.session_state["loaded_symbol"])
        persisted_strategy = query_text("strategy", "Short Strangle")
        strategy_index = (
            ALL_STRATEGIES.index(persisted_strategy)
            if persisted_strategy in ALL_STRATEGIES
            else ALL_STRATEGIES.index("Short Strangle")
        )
        strategy = st.selectbox(
            "Strategy",
            ALL_STRATEGIES,
            index=strategy_index,
            key="strategy_selection",
        )
        with st.expander("Advanced settings"):
            saved_iv_source = query_text("iv_source", "Yahoo")
            iv_source = st.selectbox(
                "Implied volatility source",
                ["Yahoo", "Calculated"],
                index=1 if saved_iv_source == "Calculated" else 0,
                help=(
                    "Yahoo is the default. Calculated IV is locally solved "
                    "from the option midpoint when available."
                ),
                key="iv_source_selection",
            )
            use_right_metrics_panel = st.toggle(
                "Right-side metrics panel",
                value=True,
                help="Turn this off to restore the previous full-width metric-card layout.",
            )
            rate_percent = st.number_input(
                "Risk-free rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=DEFAULT_RATE * 100,
                step=0.05,
                help="Manual SOFR/SR3-style approximation used for theoretical pricing.",
            )
            dividend_override = st.checkbox("Override dividend yield")
            dividend_percent = st.number_input(
                "Dividend yield (%)",
                min_value=0.0,
                max_value=25.0,
                value=0.0,
                step=0.05,
                disabled=not dividend_override,
            )
    persist_query_value("ticker", symbol)
    persist_query_value("strategy", strategy)
    persist_query_value("iv_source", iv_source)
    try:
        market_ready_key = f"market_ready_{symbol}"
        if st.session_state.get(market_ready_key):
            market = load_market_context(symbol)
        else:
            with st.spinner(f"Loading {symbol} market information…"):
                market = load_market_context(symbol)
            st.session_state[market_ready_key] = True
    except Exception as exc:
        st.error(f"Could not load options for {symbol}. Check the ticker or try again shortly.")
        with st.expander("Technical detail"):
            st.code(str(exc))
        return

    today = date.today()
    valid_expirations = [expiration for expiration in market.expirations if expiration >= today]
    if not valid_expirations:
        st.warning(f"No current expirations were returned for {symbol}.")
        return

    rate = rate_percent / 100.0
    dividend_yield = dividend_percent / 100.0 if dividend_override else market.dividend_yield
    is_time_spread = strategy in TIME_SPREAD_STRATEGIES

    with st.sidebar:
        saved_front_iso = query_text("front_expiration")
        saved_front = next(
            (
                expiration
                for expiration in valid_expirations
                if expiration.isoformat() == saved_front_iso
            ),
            None,
        )
        front_index = (
            valid_expirations.index(saved_front)
            if saved_front is not None
            else nearest_expiration_index(
                valid_expirations,
                today,
                DEFAULT_FRONT_DTE,
            )
        )
        front_expiration = st.selectbox(
            "Expiration" if not is_time_spread else "Front expiration",
            valid_expirations,
            index=front_index,
            format_func=lambda item: f"{item:%b %d, %Y} · {(item - today).days} DTE",
        )
        back_expiration: date | None = None
        if is_time_spread:
            back_choices = [expiration for expiration in valid_expirations if expiration > front_expiration]
            if not back_choices:
                st.warning("Choose an earlier front expiration to create a time spread.")
                return
            saved_back_iso = query_text("back_expiration")
            saved_back = next(
                (
                    expiration
                    for expiration in back_choices
                    if expiration.isoformat() == saved_back_iso
                ),
                None,
            )
            back_index = (
                back_choices.index(saved_back)
                if saved_back is not None
                else nearest_expiration_index(
                    back_choices,
                    today,
                    DEFAULT_BACK_DTE,
                )
            )
            back_expiration = st.selectbox(
                "Back expiration",
                back_choices,
                index=back_index,
                format_func=lambda item: f"{item:%b %d, %Y} · {(item - today).days} DTE",
            )
    persist_query_value("front_expiration", front_expiration.isoformat())
    if back_expiration is not None:
        persist_query_value("back_expiration", back_expiration.isoformat())

    required_expirations = [front_expiration] + ([back_expiration] if back_expiration else [])

    def prepared_chains() -> dict[date, dict[str, pd.DataFrame]]:
        result: dict[date, dict[str, pd.DataFrame]] = {}
        for expiration in required_expirations:
            calls, puts = load_prepared_option_chain(
                symbol=symbol,
                expiration_iso=expiration.isoformat(),
                spot=market.spot,
                rate=rate,
                dividend_yield=dividend_yield,
                valuation_date_iso=today.isoformat(),
            )
            calls = apply_iv_source(calls, iv_source)
            puts = apply_iv_source(puts, iv_source)
            if calls.empty or puts.empty:
                raise ValueError(
                    f"The {expiration:%Y-%m-%d} chain was incomplete."
                )
            result[expiration] = {"call": calls, "put": puts}
        return result

    try:
        chain_ready_key = (
            f"chain_ready_{symbol}_{front_expiration}_{back_expiration}_"
            f"{rate:.6f}_{dividend_yield:.6f}"
        )
        if st.session_state.get(chain_ready_key):
            chains = prepared_chains()
        else:
            with st.spinner("Loading selected option chain…"):
                chains = prepared_chains()
            st.session_state[chain_ready_key] = True
    except Exception as exc:
        st.error("The selected option chain could not be loaded. Yahoo may be temporarily unavailable.")
        with st.expander("Technical detail"):
            st.code(str(exc))
        return

    # Version the persisted setup because strike slots are positional and the
    # double-diagonal display order changed in this release.
    context_key = f"v3_{symbol}_{strategy}_{front_expiration}_{back_expiration}"
    try:
        specifications = default_leg_specs(strategy, chains, front_expiration, back_expiration, market.spot)
        default_legs = [leg_from_chain(chains, option_type, side, expiration, strike) for option_type, side, expiration, strike in specifications]
    except Exception as exc:
        st.warning("There are not enough usable strikes to initialize this strategy.")
        with st.expander("Technical detail"):
            st.code(str(exc))
        return

    # The new 4:1 layout is reversible from Advanced settings. Both layouts
    # share all calculations; only their presentation differs.
    if use_right_metrics_panel:
        main_view, metrics_view = st.columns([4, 1], gap="large")
    else:
        main_view = st.container()
        metrics_view = None

    anchor_widget_key = f"anchor_pairs_{context_key}"
    with main_view:
        st.markdown(f"### {symbol} · {price_text(market.spot)}")
        st.markdown(f"#### {strategy}")
        strike_settings = st.columns([1, 1, 0.55])
        with strike_settings[0]:
            saved_view = query_text("strike_view", "List view")
            strike_view = st.radio(
                "Strike view",
                ["List view", "Slider view"],
                index=0 if saved_view != "Slider view" else 1,
                horizontal=True,
                key="strike_view_selection",
                label_visibility="collapsed",
            )
        with strike_settings[1]:
            if is_time_spread:
                anchor_pairs = st.checkbox(
                    "Anchor Short/Long pairs",
                    value=(
                        query_text("anchor_pairs", "false").lower() == "true"
                        if query_text("position_context") == context_key
                        else False
                    ),
                    key=anchor_widget_key,
                    help=(
                        "Keep each short and long pair at the same strike, or "
                        "the closest strike available in the other expiration."
                    ),
                )
            else:
                anchor_pairs = False
        with strike_settings[2]:
            st.button(
                "Reset defaults",
                key=f"reset_defaults_{context_key}",
                on_click=reset_position_defaults,
                args=(context_key,),
                help=(
                    "Restore strategy strikes, starting IV, chart range, and "
                    "the default simulation date."
                ),
                use_container_width=True,
            )
        columns = st.columns(min(len(default_legs), 4))

    persist_query_value("strike_view", strike_view)
    persist_query_value("anchor_pairs", str(anchor_pairs).lower())
    same_persisted_context = query_text("position_context") == context_key
    selected_legs: list[OptionLeg] = []
    for index, leg in enumerate(default_legs):
        with columns[index % len(columns)]:
            frame = chain_for_type(chains, leg.expiration, leg.option_type)
            target_index = next(
                (
                    candidate_index
                    for candidate_index, candidate in enumerate(default_legs)
                    if candidate_index != index
                    and candidate.option_type == leg.option_type
                    and candidate.side == -leg.side
                ),
                None,
            )
            target_strikes: tuple[float, ...] = ()
            if target_index is not None:
                target_leg = default_legs[target_index]
                target_frame = chain_for_type(
                    chains, target_leg.expiration, target_leg.option_type
                )
                target_strikes = tuple(
                    sorted(target_frame["strike"].astype(float).unique())
                )
            initial_strike = (
                query_float(f"strike_{index}", leg.strike)
                if same_persisted_context
                else leg.strike
            )
            selected_strike = render_strike_selector(
                leg,
                index,
                frame,
                context_key,
                strike_view,
                initial_strike,
                target_index,
                target_strikes,
                anchor_widget_key,
            )
            selected_legs.append(leg_from_chain(chains, leg.option_type, leg.side, leg.expiration, selected_strike))

    persist_query_value("position_context", context_key)
    for index, leg in enumerate(selected_legs):
        persist_query_value(f"strike_{index}", f"{leg.strike:g}")

    execution = execution_estimate(selected_legs)
    # The live strategy midpoint is the entry basis and updates automatically
    # whenever a strike or expiration changes.
    entry_per_share = float(execution.midpoint)

    # Discover expiration breakevens independently of the visible chart, then
    # size the initial strike-centered window to clear the outer roots by 3%.
    average_strike = float(np.mean([leg.strike for leg in selected_legs]))
    all_break_even_points = expiration_breakevens(
        selected_legs,
        market.spot,
        entry_per_share,
        rate,
        dividend_yield,
        front_expiration,
    )
    automatic_range_percent = default_chart_range_percent(
        average_strike, all_break_even_points
    )
    strike_key = "_".join(f"{leg.strike:g}" for leg in selected_legs)
    front_starting_iv = average_starting_iv(selected_legs, front_expiration)
    front_iv_label = (
        "Front Implied Volatility (IV)"
        if is_time_spread
        else "Implied Volatility (IV)"
    )
    iv_source_key = iv_source.lower()
    front_iv_query_key = f"front_iv_{iv_source_key}"
    back_iv_query_key = f"back_iv_{iv_source_key}"

    with main_view:
        st.divider()
        if is_time_spread:
            range_column, control_mid, control_right = st.columns([0.8, 1, 1])
        else:
            range_column, control_mid = st.columns([0.8, 1])
            control_right = None
    with range_column:
        range_percent = st.number_input(
            "Chart price range (±%)",
            min_value=0.25,
            value=float(automatic_range_percent),
            step=0.25,
            format="%.2f",
            key=f"chart_range_{context_key}_{strike_key}",
            help="Defaults to about 3% beyond the outer expiration breakevens.",
        )
    with control_mid:
        saved_front_iv = (
            query_float(front_iv_query_key, front_starting_iv * 100.0)
            if same_persisted_context
            else front_starting_iv * 100.0
        )
        front_iv_percent = st.number_input(
            front_iv_label,
            min_value=0.1,
            max_value=500.0,
            value=min(max(float(saved_front_iv), 0.1), 500.0),
            step=0.1,
            format="%.1f",
            help="Enter IV as a percentage, such as 14.6 for 14.6%.",
            key=f"front_iv_value_{context_key}_{iv_source_key}",
        )
        front_iv_points = front_iv_percent - front_starting_iv * 100.0
    if control_right is not None:
        with control_right:
            back_starting_iv = average_starting_iv(
                selected_legs, back_expiration
            )
            saved_back_iv = (
                query_float(back_iv_query_key, back_starting_iv * 100.0)
                if same_persisted_context
                else back_starting_iv * 100.0
            )
            back_iv_percent = st.number_input(
                "Back Implied Volatility (IV)",
                min_value=0.1,
                max_value=500.0,
                value=min(max(float(saved_back_iv), 0.1), 500.0),
                step=0.1,
                format="%.1f",
                help="Enter IV as a percentage, such as 14.6 for 14.6%.",
                key=f"back_iv_value_{context_key}_{iv_source_key}",
            )
            back_iv_points = back_iv_percent - back_starting_iv * 100.0
    else:
        back_iv_points = front_iv_points

    persist_query_value(front_iv_query_key, f"{front_iv_percent:g}")
    if control_right is not None:
        persist_query_value(back_iv_query_key, f"{back_iv_percent:g}")

    with main_view:
        if front_expiration > today:
            front_dte = (front_expiration - today).days
            default_simulation_date = today + timedelta(days=front_dte // 2)
            selected_date = st.slider(
                "Simulation date",
                min_value=today,
                max_value=front_expiration,
                value=default_simulation_date,
                format="MMM D, YYYY",
                help="Defaults to halfway between today and front expiration. At front expiry, front legs become intrinsic while back legs retain time value.",
                key=f"simulation_date_{context_key}",
            )
        else:
            selected_date = today
            st.markdown("**Simulation date**")
            st.caption("This expiration is today; the curve is at expiration.")

    # Center the chart on the position itself rather than the current spot.
    # An arithmetic mean keeps the behavior predictable for any leg count.
    lower_price = max(0.01, average_strike * (1.0 - range_percent / 100.0))
    upper_price = average_strike * (1.0 + range_percent / 100.0)
    price_grid = np.linspace(lower_price, upper_price, 401)
    entry_dollars = entry_per_share * CONTRACT_MULTIPLIER
    selected_value = np.asarray(
        position_value(selected_legs, price_grid, selected_date, rate, dividend_yield, front_expiration, front_iv_points, back_iv_points)
    )
    expiry_value = np.asarray(
        position_value(selected_legs, price_grid, front_expiration, rate, dividend_yield, front_expiration, front_iv_points, back_iv_points)
    )
    selected_pnl = selected_value - entry_dollars
    expiry_pnl = expiry_value - entry_dollars
    break_even_points = [
        point
        for point in all_break_even_points
        if lower_price <= point <= upper_price
    ]
    figure = build_pnl_figure(
        price_grid,
        selected_pnl,
        expiry_pnl,
        market.spot,
        selected_date,
        front_expiration,
        break_even_points,
        is_time_spread,
        legs=selected_legs,
        pnl_percentage_basis=abs(entry_dollars),
    )
    with main_view:
        st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})

    # Exposure metrics are anchored to the entry date and unshocked model IV,
    # independent of the chart's simulation-date and IV scenario controls.
    entry_greeks = position_greeks(
        selected_legs,
        market.spot,
        today,
        rate,
        dividend_yield,
        front_expiration,
        0.0,
        0.0,
    )
    advanced_metrics = calculate_advanced_metrics(
        strategy=strategy,
        legs=selected_legs,
        spot=market.spot,
        entry_per_share=entry_per_share,
        entry_date=today,
        selected_date=selected_date,
        rate=rate,
        dividend_yield=dividend_yield,
        front_expiration=front_expiration,
        front_iv_points=front_iv_points,
        back_iv_points=back_iv_points,
        entry_greeks=entry_greeks,
    )
    max_profit, max_loss, _ = risk_summary(
        strategy, selected_legs, entry_per_share, rate, dividend_yield, front_expiration
    )
    if metrics_view is not None:
        with metrics_view:
            render_right_metrics_panel(
                entry_per_share,
                execution,
                max_profit,
                max_loss,
                entry_greeks,
                advanced_metrics,
            )
    else:
        with main_view:
            render_legacy_metric_grid(
                entry_per_share,
                execution,
                max_profit,
                max_loss,
                entry_greeks,
                advanced_metrics,
            )

    with main_view:
        with st.expander("Quote and model diagnostics"):
            diagnostics: list[dict[str, object]] = []
            for leg in selected_legs:
                warnings = quote_warnings(leg)
                diagnostics.append(
                    {
                        "Leg": leg.label,
                        "Expiration": leg.expiration.isoformat(),
                        "Strike": leg.strike,
                        "Bid": leg.bid,
                        "Ask": leg.ask,
                        "Last": leg.last,
                        "Mid": leg.market_mid,
                        "Yahoo IV": iv_text(leg.yahoo_iv),
                        "Calculated IV": iv_text(leg.calculated_iv),
                        "Active IV": iv_text(leg.model_iv),
                        "IV source": leg.iv_source,
                        "Volume": int(leg.volume),
                        "Open interest": int(leg.open_interest),
                        "Quote source": leg.quote_source,
                        "Warning": ", ".join(warnings) if warnings else "—",
                    }
                )
            st.dataframe(pd.DataFrame(diagnostics), hide_index=True, use_container_width=True)
            st.caption(
                "Yahoo IV is the default pricing source. Calculated IV is "
                "locally solved from the midpoint and falls back when a valid "
                "solution is unavailable."
            )

    with main_view:
        st.caption(f"Quotes received {market.retrieved_at:%b %d, %I:%M %p %Z}.")

if __name__ == "__main__":
    main()
