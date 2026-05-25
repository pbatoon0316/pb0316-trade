import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="Options GEX Heatmap", layout="wide")


METRIC_COLUMNS = {
    "Net GEX": "net_gex",
    "Call GEX": "call_gex",
    "Put GEX": "put_gex",
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
        "heatmap_df",
        "snapshot_time",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)


@st.cache_data(ttl=3600)
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


def calculate_black_scholes_gamma(S, K, T, sigma, r):
    if pd.isna(S) or pd.isna(K) or pd.isna(T) or pd.isna(sigma) or pd.isna(r):
        return np.nan
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    if sigma > 5:
        return np.nan

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    normal_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    gamma = normal_pdf / (S * sigma * math.sqrt(T))
    return gamma


def compute_gex(raw_options_df, spot, risk_free_rate):
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

    df["gamma"] = df.apply(
        lambda row: calculate_black_scholes_gamma(
            spot,
            row["strike"],
            row["T"],
            row["impliedVolatility"],
            risk_free_rate,
        ),
        axis=1,
    )

    df["call_gamma_value"] = np.where(df["option_type"] == "call", df["gamma"], np.nan)
    df["put_gamma_value"] = np.where(df["option_type"] == "put", df["gamma"], np.nan)
    df["call_oi_value"] = np.where(df["option_type"] == "call", df["openInterest"], 0)
    df["put_oi_value"] = np.where(df["option_type"] == "put", df["openInterest"], 0)
    df["call_volume_value"] = np.where(df["option_type"] == "call", df["volume"], 0)
    df["put_volume_value"] = np.where(df["option_type"] == "put", df["volume"], 0)

    def sum_gamma(values):
        valid_values = values.dropna()
        if valid_values.empty:
            return np.nan
        return valid_values.sum()

    grouped = (
        df.groupby(["ticker", "expiry", "dte", "strike"], dropna=False)
        .agg(
            call_gamma=("call_gamma_value", sum_gamma),
            put_gamma=("put_gamma_value", sum_gamma),
            call_oi=("call_oi_value", "sum"),
            put_oi=("put_oi_value", "sum"),
            call_volume=("call_volume_value", "sum"),
            put_volume=("put_volume_value", "sum"),
        )
        .reset_index()
    )

    grouped["call_gamma"] = np.where(grouped["call_oi"] == 0, 0, grouped["call_gamma"])
    grouped["put_gamma"] = np.where(grouped["put_oi"] == 0, 0, grouped["put_gamma"])

    # Gamma exposure is approximate dollar gamma per 1% move in the underlying.
    gex_multiplier = 100 * spot**2 * 0.01
    grouped["call_gex"] = grouped["call_gamma"] * grouped["call_oi"] * gex_multiplier
    grouped["put_gex"] = -1 * grouped["put_gamma"] * grouped["put_oi"] * gex_multiplier
    grouped["net_gex"] = grouped["call_gex"] + grouped["put_gex"]
    grouped["total_oi"] = grouped["call_oi"] + grouped["put_oi"]
    grouped["total_volume"] = grouped["call_volume"] + grouped["put_volume"]

    return grouped.sort_values(["expiry", "strike"]).reset_index(drop=True)


def build_listed_strikes_heatmap_matrix(gex_df, selected_expiries, metric, lower_strike, upper_strike):
    if gex_df.empty:
        return pd.DataFrame()

    metric_column = METRIC_COLUMNS[metric]

    filtered_df = gex_df[
        (gex_df["expiry"].isin(selected_expiries))
        & (gex_df["strike"] >= lower_strike)
        & (gex_df["strike"] <= upper_strike)
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


def format_gex_value(value):
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
                f"<br>{metric}: {format_gex_value(value)}"
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
            colorbar=dict(title=metric),
        )
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


def make_total_gex_by_strike_fig(gex_df, selected_expiries, lower_strike, upper_strike, spot):
    filtered_df = gex_df[
        (gex_df["expiry"].isin(selected_expiries))
        & (gex_df["strike"] >= lower_strike)
        & (gex_df["strike"] <= upper_strike)
    ].copy()

    if filtered_df.empty:
        return None

    total_gex_by_strike = filtered_df.groupby("strike")["net_gex"].sum(min_count=1)
    total_gex_by_strike = total_gex_by_strike.dropna()

    if total_gex_by_strike.empty:
        return None

    colors = np.where(total_gex_by_strike >= 0, "#276419", "#8e0152")

    fig = go.Figure(
        data=go.Bar(
            x=total_gex_by_strike.values,
            y=total_gex_by_strike.index,
            orientation="h",
            marker_color=colors,
            hovertemplate=(
                "Strike: %{y:g}<br>"
                "Total GEX: %{x:,.0f}"
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
        title="Total GEX by Strike",
        xaxis_title="GEX",
        yaxis_title="",
        height=720,
        margin=dict(l=20, r=20, t=70, b=40),
        showlegend=False,
    )

    return fig


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


def render_data_download_buttons():
    raw_options_df = st.session_state.get("raw_options_df")
    gex_df = st.session_state.get("gex_df")
    heatmap_df = st.session_state.get("heatmap_df")

    cols = st.columns(3)
    if raw_options_df is not None and not raw_options_df.empty:
        cols[0].download_button(
            "Download raw options CSV",
            dataframe_to_csv(raw_options_df),
            file_name="raw_options.csv",
            mime="text/csv",
        )

    if gex_df is not None and not gex_df.empty:
        cols[1].download_button(
            "Download GEX CSV",
            dataframe_to_csv(gex_df),
            file_name="computed_gex.csv",
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
    gex_df = st.session_state.get("gex_df")
    heatmap_df = st.session_state.get("heatmap_df")

    if raw_options_df is not None and not raw_options_df.empty:
        with st.expander("Show raw options data"):
            st.dataframe(raw_options_df, use_container_width=True)

    if gex_df is not None and not gex_df.empty:
        with st.expander("Show computed GEX data"):
            st.dataframe(gex_df, use_container_width=True)

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

        metric = st.selectbox("Metric", options=list(METRIC_COLUMNS.keys()), index=0)

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
            with st.spinner("Computing GEX and drawing heatmap..."):
                selected_expiries = st.session_state["selected_expiries"]
                working_raw_df = raw_options_df[raw_options_df["expiry"].isin(selected_expiries)].copy()

                gex_df = compute_gex(
                    working_raw_df,
                    st.session_state["spot"],
                    st.session_state["risk_free_rate"],
                )
                st.session_state["gex_df"] = gex_df

    if "raw_options_df" not in st.session_state:
        st.info("Enter a ticker to begin.")
        return

    show_snapshot_metrics()

    gex_df = st.session_state.get("gex_df")
    if (
        gex_df is not None
        and not gex_df.empty
        and lower_strike is not None
        and upper_strike is not None
    ):
        heatmap_df = build_listed_strikes_heatmap_matrix(
            gex_df,
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
            total_gex_fig = make_total_gex_by_strike_fig(
                gex_df,
                st.session_state["selected_expiries"],
                lower_strike,
                upper_strike,
                st.session_state["spot"],
            )
            heatmap_col, gex_col = st.columns([3, 2])

            with heatmap_col:
                st.plotly_chart(fig, use_container_width=True)

            with gex_col:
                if total_gex_fig is not None:
                    st.plotly_chart(total_gex_fig, use_container_width=True)

    render_data_download_buttons()
    render_data_tables()


if __name__ == "__main__":
    main()
