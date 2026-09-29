import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


METRIC_COLUMNS = {
    "Gamma": "net_gex",
    "Vanna": "net_vanna_exposure",
    "Charm": "net_charm_exposure",
}


def reset_data_state():
    """Clear downloaded and computed data for a fresh snapshot."""
    keys_to_clear = [
        "ticker",
        "spot",
        "sr3_price",
        "risk_free_rate",
        "available_expiries",
        "selected_expiries",
        "loaded_expiries",
        "raw_options_df",
        "gex_df",
        "greeks_df",
        "heatmap_df",
        "snapshot_time",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)


@st.cache_data(ttl=900)
def get_available_expiries(ticker):
    stock = yf.Ticker(ticker)
    return list(stock.options)


@st.cache_data(ttl=300)
def get_spot_price(ticker):
    stock = yf.Ticker(ticker)

    try:
        fast_info = getattr(stock, "fast_info", {})
        last_price = fast_info.get("last_price") if fast_info else None
    except Exception:
        last_price = None

    if last_price and last_price > 0:
        return float(last_price)

    history = stock.history(period="5d")
    if history.empty or "Close" not in history:
        raise ValueError("Could not fetch a valid spot price.")

    close_values = history["Close"].dropna()
    if close_values.empty:
        raise ValueError("Could not fetch a valid spot price.")

    return float(close_values.iloc[-1])


@st.cache_data(ttl=900)
def get_sr3_risk_free_rate():
    sr3 = yf.Ticker("SR3=F")
    sr3_price = None

    try:
        fast_info = getattr(sr3, "fast_info", {})
        if fast_info:
            sr3_price = fast_info.get("last_price")
    except Exception:
        sr3_price = None

    if not sr3_price or sr3_price <= 0:
        history = sr3.history(period="10d")
        if not history.empty and "Close" in history:
            close_values = history["Close"].dropna()
            if not close_values.empty:
                sr3_price = float(close_values.iloc[-1])

    if not sr3_price or sr3_price <= 0:
        return None, 0.05, False

    risk_free_rate = (100.0 - float(sr3_price)) / 100.0
    return float(sr3_price), float(risk_free_rate), True


@st.cache_data(ttl=900)
def get_option_chain_for_expiry(ticker, expiry):
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiry)

    frames = []
    for option_type, frame in [("call", chain.calls), ("put", chain.puts)]:
        if frame is None or frame.empty:
            continue

        option_df = frame.copy()
        option_df["option_type"] = option_type
        frames.append(option_df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_option_chains(ticker, expiries, spot, snapshot_time):
    frames = []
    failed_expiries = []

    for expiry in expiries:
        try:
            chain_df = get_option_chain_for_expiry(ticker, expiry)
            if chain_df.empty:
                failed_expiries.append(expiry)
            else:
                chain_df = chain_df.copy()
                chain_df["ticker"] = ticker
                chain_df["expiry"] = expiry
                chain_df["snapshot_time"] = snapshot_time
                chain_df["spot"] = spot
                frames.append(chain_df)
        except Exception:
            failed_expiries.append(expiry)

    if frames:
        raw_options_df = pd.concat(frames, ignore_index=True)
    else:
        raw_options_df = pd.DataFrame()

    first_columns = ["ticker", "snapshot_time", "spot", "expiry", "option_type"]
    existing_first_columns = [column for column in first_columns if column in raw_options_df.columns]
    other_columns = [column for column in raw_options_df.columns if column not in existing_first_columns]
    raw_options_df = raw_options_df[existing_first_columns + other_columns]

    return raw_options_df, failed_expiries


def load_missing_expiries(expiries, loading_message):
    loaded_expiries = st.session_state.get("loaded_expiries", [])
    missing_expiries = [expiry for expiry in expiries if expiry not in loaded_expiries]

    if not missing_expiries:
        st.info("All requested expiries are already loaded.")
        return

    with st.spinner(loading_message):
        new_df, failed_expiries = load_option_chains(
            st.session_state["ticker"],
            missing_expiries,
            st.session_state["spot"],
            st.session_state["snapshot_time"],
        )

        if not new_df.empty:
            current_df = st.session_state["raw_options_df"]
            st.session_state["raw_options_df"] = pd.concat(
                [current_df, new_df],
                ignore_index=True,
            )

        successful_expiries = [
            expiry for expiry in missing_expiries if expiry not in failed_expiries
        ]
        st.session_state["loaded_expiries"] = loaded_expiries + successful_expiries

        if failed_expiries:
            st.warning(f"Some expiries returned no data: {', '.join(failed_expiries)}")
        if successful_expiries:
            st.success(f"Loaded: {', '.join(successful_expiries)}")


def get_this_year_expiries(expiries):
    current_year = date.today().year
    return [
        expiry
        for expiry in expiries
        if datetime.strptime(expiry, "%Y-%m-%d").date().year == current_year
    ]


def calculate_dte(expiry):
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    return (expiry_date - date.today()).days


def calculate_black_scholes_second_order_greeks(S, K, T, sigma, r):
    if pd.isna(S) or pd.isna(K) or pd.isna(T) or pd.isna(sigma) or pd.isna(r):
        return np.nan, np.nan, np.nan
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan, np.nan, np.nan
    if sigma > 5:
        return np.nan, np.nan, np.nan

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    normal_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    gamma = normal_pdf / (S * sigma * sqrt_T)

    # Vanna is the change in delta per 1.00 change in volatility. Exposure below
    # scales it to a one-percentage-point (0.01) volatility move.
    vanna = -normal_pdf * d2 / sigma

    # Calendar-time charm: d(delta)/d(calendar time) = -d(delta)/d(T).
    # With no dividend-yield input, call and put charm have the same value.
    charm = -normal_pdf * (2 * r * T - d2 * sigma * sqrt_T) / (
        2 * T * sigma * sqrt_T
    )
    return gamma, vanna, charm


def compute_greek_exposures(raw_options_df, spot, risk_free_rate):
    if raw_options_df.empty:
        return pd.DataFrame()

    df = raw_options_df.copy()

    needed_columns = [
        "ticker",
        "expiry",
        "option_type",
        "strike",
        "impliedVolatility",
        "openInterest",
        "volume",
    ]
    for column in needed_columns:
        if column not in df.columns:
            df[column] = np.nan

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
    df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["dte"] = df["expiry"].apply(calculate_dte)
    df["T"] = df["dte"].apply(lambda dte: max(dte, 1) / 365.0)

    greek_values = df.apply(
        lambda row: calculate_black_scholes_second_order_greeks(
            spot, row["strike"], row["T"], row["impliedVolatility"], risk_free_rate
        ),
        axis=1,
        result_type="expand",
    )
    greek_values.columns = ["gamma", "vanna", "charm"]
    df[["gamma", "vanna", "charm"]] = greek_values

    for greek in ["gamma", "vanna", "charm"]:
        df[f"call_{greek}_value"] = np.where(
            df["option_type"] == "call", df[greek], np.nan
        )
        df[f"put_{greek}_value"] = np.where(
            df["option_type"] == "put", df[greek], np.nan
        )
    df["call_oi_value"] = np.where(df["option_type"] == "call", df["openInterest"], 0)
    df["put_oi_value"] = np.where(df["option_type"] == "put", df["openInterest"], 0)
    df["call_volume_value"] = np.where(df["option_type"] == "call", df["volume"], 0)
    df["put_volume_value"] = np.where(df["option_type"] == "put", df["volume"], 0)

    def sum_greek(values):
        valid_values = values.dropna()
        if valid_values.empty:
            return np.nan
        return valid_values.sum()

    grouped = (
        df.groupby(["ticker", "expiry", "dte", "strike"], dropna=False)
        .agg(
            call_gamma=("call_gamma_value", sum_greek),
            put_gamma=("put_gamma_value", sum_greek),
            call_vanna=("call_vanna_value", sum_greek),
            put_vanna=("put_vanna_value", sum_greek),
            call_charm=("call_charm_value", sum_greek),
            put_charm=("put_charm_value", sum_greek),
            call_oi=("call_oi_value", "sum"),
            put_oi=("put_oi_value", "sum"),
            call_volume=("call_volume_value", "sum"),
            put_volume=("put_volume_value", "sum"),
        )
        .reset_index()
    )

    for greek in ["gamma", "vanna", "charm"]:
        grouped[f"call_{greek}"] = np.where(
            grouped["call_oi"] == 0, 0, grouped[f"call_{greek}"]
        )
        grouped[f"put_{greek}"] = np.where(
            grouped["put_oi"] == 0, 0, grouped[f"put_{greek}"]
        )

    # Gamma exposure is approximate dollar gamma per 1% move in the underlying.
    gex_multiplier = 100 * spot**2 * 0.01
    grouped["call_gex"] = grouped["call_gamma"] * grouped["call_oi"] * gex_multiplier
    grouped["put_gex"] = -1 * grouped["put_gamma"] * grouped["put_oi"] * gex_multiplier
    grouped["net_gex"] = grouped["call_gex"] + grouped["put_gex"]

    contract_multiplier = 100
    one_volatility_point = 0.01
    one_calendar_day = 1 / 365.0
    grouped["call_vanna_exposure"] = (
        grouped["call_vanna"]
        * grouped["call_oi"]
        * contract_multiplier
        * one_volatility_point
    )
    grouped["put_vanna_exposure"] = (
        -1
        * grouped["put_vanna"]
        * grouped["put_oi"]
        * contract_multiplier
        * one_volatility_point
    )
    grouped["net_vanna_exposure"] = (
        grouped["call_vanna_exposure"] + grouped["put_vanna_exposure"]
    )
    grouped["call_charm_exposure"] = (
        grouped["call_charm"]
        * grouped["call_oi"]
        * contract_multiplier
        * one_calendar_day
    )
    grouped["put_charm_exposure"] = (
        -1
        * grouped["put_charm"]
        * grouped["put_oi"]
        * contract_multiplier
        * one_calendar_day
    )
    grouped["net_charm_exposure"] = (
        grouped["call_charm_exposure"] + grouped["put_charm_exposure"]
    )
    grouped["total_oi"] = grouped["call_oi"] + grouped["put_oi"]
    grouped["total_volume"] = grouped["call_volume"] + grouped["put_volume"]

    return grouped.sort_values(["expiry", "strike"]).reset_index(drop=True)


def build_listed_strikes_heatmap_matrix(
    greeks_df, selected_expiries, metric, lower_strike, upper_strike
):
    if greeks_df.empty:
        return pd.DataFrame()

    metric_column = METRIC_COLUMNS[metric]

    filtered_df = greeks_df[
        (greeks_df["expiry"].isin(selected_expiries))
        & (greeks_df["strike"] >= lower_strike)
        & (greeks_df["strike"] <= upper_strike)
    ].copy()

    if filtered_df.empty:
        return pd.DataFrame()

    def pivot_column(column):
        matrix_df = filtered_df.pivot_table(
            index="strike",
            columns="expiry",
            values=column,
            aggfunc=lambda values: values.sum(min_count=1),
        )
        matrix_df = matrix_df.reindex(columns=selected_expiries)
        return matrix_df.sort_index()

    heatmap_df = pivot_column(metric_column)
    heatmap_df.attrs["volume_df"] = pivot_column("total_volume")
    heatmap_df.attrs["open_interest_df"] = pivot_column("total_oi")
    return heatmap_df


def format_exposure_value(value):
    if pd.isna(value):
        return ""

    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"


def format_count_value(value):
    if pd.isna(value):
        return ""
    return f"{value:,.0f}"


def make_heatmap_fig(heatmap_df, metric, color_scale_mode, spot, ticker, snapshot_time):
    values = heatmap_df.values.astype(float)

    if heatmap_df.empty or np.all(np.isnan(values)):
        return None, "All heatmap values are missing."

    abs_values = np.abs(values)
    if color_scale_mode == "90th percentile":
        zmax = np.nanpercentile(abs_values, 90)
    elif color_scale_mode == "95th Percentile":
        zmax = np.nanpercentile(abs_values, 95)
    elif color_scale_mode == "99th Percentile":
        zmax = np.nanpercentile(abs_values, 99)
    else:
        zmax = np.nanmax(abs_values)

    if pd.isna(zmax) or zmax == 0:
        return None, "All heatmap values are zero or missing."

    hover_text = []
    volume_df = heatmap_df.attrs.get("volume_df")
    open_interest_df = heatmap_df.attrs.get("open_interest_df")

    for strike in heatmap_df.index:
        row_text = []
        for expiry in heatmap_df.columns:
            value = heatmap_df.loc[strike, expiry]
            volume = volume_df.loc[strike, expiry] if volume_df is not None else np.nan
            open_interest = (
                open_interest_df.loc[strike, expiry] if open_interest_df is not None else np.nan
            )
            row_text.append(
                f"Expiry: {expiry}"
                f"<br>Strike: {strike:g}"
                f"<br>{metric}: {format_exposure_value(value)}"
                f"<br>Volume: {format_count_value(volume)}"
                f"<br>Open Interest: {format_count_value(open_interest)}"
            )
        hover_text.append(row_text)

    title = f"{ticker} {metric} Heatmap"
    if snapshot_time:
        title += f" - {snapshot_time}"

    piyg_with_white_center = [
        [0.0, "#8e0152"],
        [0.25, "#de77ae"],
        [0.5, "#ffffff"],
        [0.75, "#7fbc41"],
        [1.0, "#276419"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=list(heatmap_df.columns),
            y=list(heatmap_df.index),
            colorscale=piyg_with_white_center,
            zmin=-zmax,
            zmax=zmax,
            zmid=0,
            hoverinfo="text",
            text=hover_text,
            colorbar=dict(
                title=dict(text=metric, side="top"),
                x=1.01, xanchor="left", thickness=18,
                tickformat="~s", xpad=8,
            ),
        )
    )

    # White strike labels without a background.
    # Annotations do not replace the heatmap's exposure/volume/OI tooltip.
    strike_labels = []
    for row_index, strike in enumerate(heatmap_df.index):
        for column_index, expiry in enumerate(heatmap_df.columns):
            if np.isfinite(values[row_index, column_index]):
                strike_labels.append(dict(
                    x=expiry, y=float(strike), text=f"{strike:g}",
                    showarrow=False, font=dict(color="white", size=10),
                    bgcolor="rgba(0,0,0,0)", borderpad=0,
                    captureevents=False,
                ))
    fig.update_layout(annotations=strike_labels)

    fig.update_xaxes(
        type="category", categoryorder="array",
        categoryarray=list(heatmap_df.columns),
        tickmode="array", tickvals=list(heatmap_df.columns),
        ticktext=[datetime.strptime(expiry, "%Y-%m-%d").strftime("%b %d %Y")
                  for expiry in heatmap_df.columns],
        tickangle=0,
    )

    fig.add_hline(
        y=spot,
        line_color="#222222",
        line_width=2,
        line_dash="dash",
        annotation_text=f"Spot {spot:.2f}",
        annotation_position="top left",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Expiration Date",
        yaxis_title="Strike Price",
        height=720,
        margin=dict(l=40, r=40, t=70, b=40),
    )

    return fig, None


def make_total_exposure_by_strike_fig(
    greeks_df, selected_expiries, metric, lower_strike, upper_strike, spot
):
    filtered_df = greeks_df[
        (greeks_df["expiry"].isin(selected_expiries))
        & (greeks_df["strike"] >= lower_strike)
        & (greeks_df["strike"] <= upper_strike)
    ].copy()

    if filtered_df.empty:
        return None

    metric_column = METRIC_COLUMNS[metric]
    total_exposure_by_strike = filtered_df.groupby("strike")[metric_column].sum(min_count=1)
    total_exposure_by_strike = total_exposure_by_strike.dropna()

    if total_exposure_by_strike.empty:
        return None

    colors = np.where(total_exposure_by_strike >= 0, "#276419", "#8e0152")

    fig = go.Figure(
        data=go.Bar(
            x=total_exposure_by_strike.values,
            y=total_exposure_by_strike.index,
            orientation="h",
            marker_color=colors,
            hovertemplate=(
                "Strike: %{y:g}<br>"
                f"Total {metric}: %{{x:,.2f}}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0, line_color="#555555", line_width=1)
    fig.add_hline(
        y=spot,
        line_color="#222222",
        line_width=2,
        line_dash="dash",
    )

    fig.update_layout(
        title=f"Total {metric} by Strike",
        xaxis_title=metric,
        yaxis_title="",
        height=720,
        margin=dict(l=20, r=20, t=70, b=40),
        showlegend=False,
    )

    return fig


@st.cache_data(show_spinner=False)
def calculate_gamma_profile(raw_options_df, selected_expiries, spot_prices,
                            risk_free_rate, as_of):
    """Reprice every usable contract at each spot, with fixed IV and OI.

    Match the existing heatmap's one-day minimum time to expiry and its
    call-positive / put-negative dollar gamma per 1% spot-move convention.
    Strike display limits must never filter the option universe here.
    """
    profile = np.full(len(spot_prices), np.nan, dtype=float)
    needed = {"expiry", "strike", "impliedVolatility", "openInterest", "option_type"}
    if raw_options_df is None or not needed.issubset(raw_options_df.columns):
        return profile
    options = raw_options_df[
        raw_options_df["expiry"].isin(selected_expiries)
    ].copy()
    for column in ("strike", "impliedVolatility", "openInterest"):
        options[column] = pd.to_numeric(options[column], errors="coerce")
    expiry = pd.to_datetime(options["expiry"], errors="coerce")
    dte = (expiry - pd.Timestamp(as_of)).dt.days
    valid = (
        np.isfinite(options[["strike", "impliedVolatility", "openInterest"]]).all(axis=1)
        & options["strike"].gt(0)
        & options["impliedVolatility"].gt(0)
        & options["impliedVolatility"].le(5)
        & options["openInterest"].gt(0)
        & options["option_type"].isin(["call", "put"])
        & dte.ge(0)
    )
    options = options.loc[valid]
    if options.empty or not np.isfinite(risk_free_rate):
        return profile
    strike = options["strike"].to_numpy(dtype=float)
    sigma = options["impliedVolatility"].to_numpy(dtype=float)
    time_years = dte.loc[valid].clip(lower=1).to_numpy(dtype=float) / 365.0
    signed_oi = options["openInterest"].to_numpy(dtype=float) * np.where(
        options["option_type"].eq("call"), 1.0, -1.0
    )
    vol_time = sigma * np.sqrt(time_years)
    # Bound temporary array sizes for large multi-expiry chains.
    for start in range(0, len(spot_prices), 32):
        spots = np.asarray(spot_prices[start:start + 32], dtype=float)[:, None]
        d1 = (np.log(spots / strike) +
              (risk_free_rate + 0.5 * sigma**2) * time_years) / vol_time
        gamma = np.exp(-0.5 * d1**2) / (np.sqrt(2 * np.pi) * spots * vol_time)
        profile[start:start + len(spots)] = (
            gamma * signed_oi * 100 * spots**2 * 0.01
        ).sum(axis=1)
    return profile


def add_gamma_profile(fig, raw_options_df, selected_expiries, risk_free_rate, spot):
    """Overlay on the existing axes without changing vertical range or margins."""
    if fig is None:
        return
    lower, upper = fig.layout.yaxis.range
    prices = np.unique(np.append(np.linspace(lower, upper, 801), spot))
    prices = prices[prices > 0]
    exposure = calculate_gamma_profile(
        raw_options_df, tuple(selected_expiries), prices,
        risk_free_rate, date.today(),
    )
    if not np.isfinite(exposure).any():
        return
    fig.add_trace(go.Scatter(
        x=exposure, y=prices, mode="lines", xaxis="x2",
        line=dict(color="black", width=2.5),
        name="Net Gamma vs. Spot", showlegend=False,
        connectgaps=False,
        hovertemplate=(
            "Hypothetical spot: %{y:,.2f}<br>"
            "Net Gamma: $%{x:,.0f} per 1% move<extra></extra>"
        ),
    ))
    # Independent horizontal scale: the aggregate profile must not compress
    # the per-strike bars. Both traces continue to share the same Y axis.
    fig.update_layout(
        title="Gamma by Strike & Net Gamma vs. Spot",
        xaxis2=dict(
            overlaying="x", anchor="y", side="top",
            title=dict(text="Net Gamma vs. Spot ($ / 1% move)", standoff=4),
            tickformat="~s", tickfont=dict(size=10),
            showgrid=False, zeroline=False, zerolinecolor="black",
            zerolinewidth=1, automargin=False, autorange=True,
        ),
    )


def align_strike_plots(heatmap_fig, exposure_fig, heatmap_df, spot):
    """Use the same pixel height and numeric strike range in both plots."""
    strikes = np.sort(heatmap_df.index.to_numpy(dtype=float))
    lower_padding = (strikes[1] - strikes[0]) / 2 if len(strikes) > 1 else 2.5
    upper_padding = (strikes[-1] - strikes[-2]) / 2 if len(strikes) > 1 else 2.5
    strike_range = [min(strikes[0] - lower_padding, spot),
                    max(strikes[-1] + upper_padding, spot)]
    # Only vertical margins must match. Reserve room for the heatmap colorbar
    # explicitly because automatic margin expansion would break Y alignment.
    for fig in (heatmap_fig, exposure_fig):
        if fig is None:
            continue
        fig.update_layout(
            height=720,
            margin=dict(l=60, r=120 if fig is heatmap_fig else 40,
                        t=70, b=90, autoexpand=False),
        )
        fig.update_yaxes(range=strike_range, autorange=False,
                         domain=[0, 1], automargin=False)
        fig.update_xaxes(automargin=False)


def dataframe_to_csv(df):
    return df.to_csv(index=True).encode("utf-8")


def show_snapshot_metrics():
    ticker = st.session_state.get("ticker", "")
    spot = st.session_state.get("spot", 0)
    available_count = len(st.session_state.get("available_expiries", []))
    loaded_count = len(st.session_state.get("loaded_expiries", []))
    raw_options_df = st.session_state.get("raw_options_df")
    row_count = len(raw_options_df) if raw_options_df is not None else 0

    st.caption(
        f"{ticker} | Spot {spot:.2f} | "
        f"{loaded_count} loaded expiries / {available_count} available | "
        f"{row_count:,} option rows"
    )


def render_data_download_buttons(cols):
    raw_options_df = st.session_state.get("raw_options_df")
    greeks_df = st.session_state.get("greeks_df")
    heatmap_df = st.session_state.get("heatmap_df")

    if raw_options_df is not None and not raw_options_df.empty:
        cols[0].download_button(
            "Download raw options CSV",
            dataframe_to_csv(raw_options_df),
            file_name="raw_options.csv",
            mime="text/csv",
        )

    if greeks_df is not None and not greeks_df.empty:
        cols[1].download_button(
            "Download computed Greeks CSV",
            dataframe_to_csv(greeks_df),
            file_name="computed_greek_exposures.csv",
            mime="text/csv",
        )

    if heatmap_df is not None and not heatmap_df.empty:
        cols[2].download_button(
            "Download heatmap CSV",
            dataframe_to_csv(heatmap_df),
            file_name="heatmap_matrix.csv",
            mime="text/csv",
        )


def render_data_tables():
    raw_options_df = st.session_state.get("raw_options_df")
    greeks_df = st.session_state.get("greeks_df")
    heatmap_df = st.session_state.get("heatmap_df")

    if raw_options_df is not None and not raw_options_df.empty:
        with st.expander("Show raw options data"):
            st.dataframe(raw_options_df, use_container_width=True)

    if greeks_df is not None and not greeks_df.empty:
        with st.expander("Show computed Greek exposure data"):
            st.dataframe(greeks_df, use_container_width=True)

    if heatmap_df is not None and not heatmap_df.empty:
        with st.expander("Show heatmap matrix"):
            st.dataframe(heatmap_df, use_container_width=True)


def download_initial_snapshot(ticker):
    reset_data_state()
    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with st.spinner(f"Downloading {ticker} options snapshot..."):
        try:
            spot = get_spot_price(ticker)
            expiries = get_available_expiries(ticker)

            if not expiries:
                st.error("No available option expirations found for this ticker.")
                return

            sr3_price, risk_free_rate, sr3_ok = get_sr3_risk_free_rate()
            if not sr3_ok:
                st.warning("Could not fetch SR3=F. Using fallback risk-free rate of 5%.")

            default_expiries = expiries[:5]
            raw_options_df, failed_expiries = load_option_chains(
                ticker,
                default_expiries,
                spot,
                snapshot_time,
            )

            if raw_options_df.empty:
                st.error("No option chain data was returned for the nearest expirations.")
                return

            st.session_state["ticker"] = ticker
            st.session_state["spot"] = spot
            st.session_state["sr3_price"] = sr3_price
            st.session_state["risk_free_rate"] = risk_free_rate
            st.session_state["available_expiries"] = expiries
            st.session_state["selected_expiries"] = default_expiries
            st.session_state["loaded_expiries"] = [
                expiry for expiry in default_expiries if expiry not in failed_expiries
            ]
            st.session_state["raw_options_df"] = raw_options_df
            st.session_state["snapshot_time"] = snapshot_time

            if failed_expiries:
                st.warning(f"Some expiries returned no data: {', '.join(failed_expiries)}")

            st.success("Data loaded. Click Recompute / Redraw Heatmap to plot it.")
        except Exception as error:
            st.error(f"Could not download data for {ticker}: {error}")


def main():
    st.set_page_config(page_title="Options Greek Exposure Heatmap", layout="wide")

    with st.sidebar:
        ticker_input = st.text_input("Ticker", value=st.session_state.get("ticker", "SPY"))
        ticker = ticker_input.strip().upper()

        if not ticker:
            st.error("Enter a valid ticker.")
        elif st.session_state.get("ticker") != ticker or "raw_options_df" not in st.session_state:
            download_initial_snapshot(ticker)

        available_expiries = st.session_state.get("available_expiries", [])
        selected_expiries = []

        if available_expiries:
            selected_expiries = st.multiselect(
                "Expiration dates",
                options=available_expiries,
                default=st.session_state.get("selected_expiries", available_expiries[:5]),
            )
            st.session_state["selected_expiries"] = selected_expiries

            load_col, year_col = st.columns(2)
            if load_col.button("Load Selected Expiries"):
                load_missing_expiries(selected_expiries, "Loading selected expiries...")

            if year_col.button("Load This Year"):
                this_year_expiries = get_this_year_expiries(available_expiries)
                if not this_year_expiries:
                    st.info(f"No expiries found in {date.today().year}.")
                else:
                    st.session_state["selected_expiries"] = this_year_expiries
                    load_missing_expiries(this_year_expiries, "Loading this year's expiries...")
                    st.rerun()

        recompute_clicked = st.button("Recompute / Redraw Heatmap")

        spot = st.session_state.get("spot")
        if spot:
            strike_pct = st.slider(
                "Strike range around spot",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                format="+/-%d%%",
            )
            lower_strike = spot * (1 - strike_pct / 100)
            upper_strike = spot * (1 + strike_pct / 100)
            st.caption(f"Displaying strikes from {lower_strike:.2f} to {upper_strike:.2f}")
        else:
            lower_strike = None
            upper_strike = None

        color_scale_mode = st.selectbox(
            "Color Scale Mode",
            options=["Auto max", "90th percentile", "95th Percentile", "99th Percentile"],
            index=0,
        )

    if recompute_clicked:
        raw_options_df = st.session_state.get("raw_options_df")
        if raw_options_df is None or raw_options_df.empty:
            st.warning("Wait for data to load before recomputing the heatmap.")
        elif not st.session_state.get("selected_expiries"):
            st.warning("Select at least one expiration date.")
        else:
            with st.spinner("Computing Greek exposures and drawing heatmap..."):
                selected_expiries = st.session_state["selected_expiries"]
                working_raw_df = raw_options_df[raw_options_df["expiry"].isin(selected_expiries)].copy()

                greeks_df = compute_greek_exposures(
                    working_raw_df,
                    st.session_state["spot"],
                    st.session_state["risk_free_rate"],
                )
                st.session_state["greeks_df"] = greeks_df

    if "raw_options_df" not in st.session_state:
        st.info("Enter a ticker to begin.")
        return

    # Reserve plot space above controls, but evaluate the selector first so both
    # plots and the heatmap download use the selected metric on this same rerun.
    plot_container = st.container()
    control_cols = st.columns([1, 1, 1.2, 1])
    metric = control_cols[0].selectbox(
        "Metric", options=list(METRIC_COLUMNS.keys()), index=0,
        label_visibility="collapsed", key="heatmap_metric",
    )
    st.session_state.pop("heatmap_df", None)

    greeks_df = st.session_state.get("greeks_df")
    if (
        greeks_df is not None
        and not greeks_df.empty
        and lower_strike is not None
        and upper_strike is not None
    ):
        heatmap_df = build_listed_strikes_heatmap_matrix(
            greeks_df,
            st.session_state["selected_expiries"],
            metric,
            lower_strike,
            upper_strike,
        )
        st.session_state["heatmap_df"] = heatmap_df

        fig, warning_message = make_heatmap_fig(
            heatmap_df,
            metric,
            color_scale_mode,
            st.session_state["spot"],
            st.session_state["ticker"],
            st.session_state.get("snapshot_time"),
        )

        if warning_message:
            st.warning(warning_message)
        else:
            total_exposure_fig = make_total_exposure_by_strike_fig(
                greeks_df,
                st.session_state["selected_expiries"],
                metric,
                lower_strike,
                upper_strike,
                st.session_state["spot"],
            )
            align_strike_plots(fig, total_exposure_fig, heatmap_df, st.session_state["spot"])
            if metric == "Gamma":
                add_gamma_profile(
                    total_exposure_fig, st.session_state["raw_options_df"],
                    st.session_state["selected_expiries"],
                    st.session_state["risk_free_rate"], st.session_state["spot"],
                )
            heatmap_col, exposure_col = plot_container.columns([3, 2])

            with heatmap_col:
                st.plotly_chart(fig, use_container_width=True)

            with exposure_col:
                if total_exposure_fig is not None:
                    st.plotly_chart(total_exposure_fig, use_container_width=True)

    render_data_download_buttons(control_cols[1:])
    render_data_tables()

    st.code(
        "Exposure convention\n"
        "Gamma: dollar gamma for a 1% underlying move.\n"
        "Net Vanna Exposure: delta shares for a 1 volatility-point increase.\n"
        "Net Charm Exposure: delta shares gained (+) or lost (-) per calendar day.\n"
        "All net values use call exposure minus put exposure based on open interest.\n"
        "This is a positioning convention, not observed dealer positioning.\n"
        "Black Gamma curve: total net gamma repriced at hypothetical spot prices,\n"
        "using all usable strikes in selected, loaded expiries with fixed IV and OI.\n"
        "The curve uses the same 1-day minimum expiry time as the heatmap.",
        language=None,
    )

    show_snapshot_metrics()


if __name__ == "__main__":
    main()
