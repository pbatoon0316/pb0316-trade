import calendar
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yfinance as yf

from pages.gex_heatmap import (
    METRIC_COLUMNS,
    build_listed_strikes_heatmap_matrix,
    compute_greek_exposures,
    dataframe_to_csv,
    get_sr3_risk_free_rate,
    make_heatmap_fig,
    make_total_exposure_by_strike_fig,
)


TICKER = "^SPX"
STATE_PREFIX = "spx_gex_"
MARKET_TIMEZONE = ZoneInfo("America/New_York")


def state_key(name):
    return f"{STATE_PREFIX}{name}"


def market_today():
    return datetime.now(MARKET_TIMEZONE).date()


def expiry_dates_in_range(expiries, start_date, end_date):
    return [
        expiry
        for expiry in expiries
        if start_date <= datetime.strptime(expiry, "%Y-%m-%d").date() <= end_date
    ]


def get_expiry_preset(expiries, preset, today=None):
    today = today or market_today()

    if preset == "0DTE":
        end_date = today
    elif preset == "0-3 DTE":
        end_date = today + timedelta(days=3)
    elif preset == "This Week":
        end_date = today + timedelta(days=6 - today.weekday())
    elif preset == "This Month":
        end_date = today.replace(
            day=calendar.monthrange(today.year, today.month)[1]
        )
    elif preset == "This Year":
        end_date = today.replace(month=12, day=31)
    else:
        raise ValueError(f"Unknown expiry preset: {preset}")

    return expiry_dates_in_range(expiries, today, end_date)


@st.cache_data(ttl=3600)
def get_spx_expiries():
    return list(yf.Ticker(TICKER).options)


@st.cache_data(ttl=120)
def get_spx_spot_price():
    stock = yf.Ticker(TICKER)

    try:
        fast_info = getattr(stock, "fast_info", {})
        last_price = fast_info.get("last_price") if fast_info else None
    except Exception:
        last_price = None

    if last_price and last_price > 0:
        return float(last_price)

    history = stock.history(period="5d")
    if history.empty or "Close" not in history:
        raise ValueError("Could not fetch a valid ^SPX price.")

    close_values = history["Close"].dropna()
    if close_values.empty:
        raise ValueError("Could not fetch a valid ^SPX price.")

    return float(close_values.iloc[-1])


@st.cache_data(ttl=900)
def get_spx_option_chain(expiry):
    chain = yf.Ticker(TICKER).option_chain(expiry)
    frames = []

    for option_type, frame in (("call", chain.calls), ("put", chain.puts)):
        if frame is None or frame.empty:
            continue
        option_df = frame.copy()
        option_df["option_type"] = option_type
        frames.append(option_df)

    if not frames:
        return pd.DataFrame(), None

    fetched_at = datetime.now(MARKET_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
    return pd.concat(frames, ignore_index=True), fetched_at


def initialize_page_state():
    if state_key("available_expiries") in st.session_state:
        return True

    with st.spinner("Loading ^SPX expirations..."):
        try:
            expiries = get_spx_expiries()
            if not expiries:
                st.error("No available ^SPX option expirations were found.")
                return False

            sr3_price, risk_free_rate, sr3_ok = get_sr3_risk_free_rate()
            if not sr3_ok:
                st.warning(
                    "Could not fetch SR3=F. Using a fallback risk-free rate of 5%."
                )

            default_expiries = get_expiry_preset(expiries, "This Week")
            st.session_state[state_key("available_expiries")] = expiries
            st.session_state[state_key("selected_expiries")] = default_expiries
            st.session_state[state_key("expiry_widget")] = default_expiries
            st.session_state[state_key("loaded_expiries")] = []
            st.session_state[state_key("raw_options_df")] = pd.DataFrame()
            st.session_state[state_key("sr3_price")] = sr3_price
            st.session_state[state_key("risk_free_rate")] = risk_free_rate
            return True
        except Exception as error:
            st.error(f"Could not initialize the SPX GEX page: {error}")
            return False


def load_expiries(expiries, loading_message):
    if not expiries:
        st.info("No available expiries match that time range.")
        return False

    frames = []
    failed_expiries = []
    fetched_times = []

    with st.spinner(loading_message):
        for expiry in expiries:
            try:
                chain_df, fetched_at = get_spx_option_chain(expiry)
                if chain_df.empty:
                    failed_expiries.append(expiry)
                    continue

                chain_df = chain_df.copy()
                chain_df["ticker"] = TICKER
                chain_df["expiry"] = expiry
                chain_df["snapshot_time"] = fetched_at
                chain_df["spot"] = st.session_state[state_key("spot")]
                frames.append(chain_df)
                if fetched_at:
                    fetched_times.append(fetched_at)
            except Exception:
                failed_expiries.append(expiry)

    if frames:
        new_df = pd.concat(frames, ignore_index=True)
        current_df = st.session_state[state_key("raw_options_df")]
        if not current_df.empty:
            current_df = current_df[~current_df["expiry"].isin(expiries)]
        st.session_state[state_key("raw_options_df")] = pd.concat(
            [current_df, new_df], ignore_index=True
        )

        loaded = set(st.session_state[state_key("loaded_expiries")])
        loaded.update(expiry for expiry in expiries if expiry not in failed_expiries)
        st.session_state[state_key("loaded_expiries")] = [
            expiry
            for expiry in st.session_state[state_key("available_expiries")]
            if expiry in loaded
        ]
        st.session_state[state_key("snapshot_time")] = max(fetched_times, default=None)

    if failed_expiries:
        st.warning(f"Some expiries returned no data: {', '.join(failed_expiries)}")
    return bool(frames)


def select_and_load_preset(preset):
    expiries = get_expiry_preset(
        st.session_state[state_key("available_expiries")], preset
    )
    if not expiries:
        st.info(f"No available expiries match {preset}.")
        return

    st.session_state[state_key("selected_expiries")] = expiries
    st.session_state[state_key("expiry_widget")] = expiries
    load_expiries(expiries, f"Loading {preset} ^SPX expiries...")


def render_downloads_and_tables():
    raw_df = st.session_state[state_key("raw_options_df")]
    greeks_df = st.session_state.get(state_key("greeks_df"))
    heatmap_df = st.session_state.get(state_key("heatmap_df"))

    columns = st.columns(3)
    if not raw_df.empty:
        columns[0].download_button(
            "Download raw options CSV",
            dataframe_to_csv(raw_df),
            file_name="spx_raw_options.csv",
            mime="text/csv",
        )
    if greeks_df is not None and not greeks_df.empty:
        columns[1].download_button(
            "Download computed Greeks CSV",
            dataframe_to_csv(greeks_df),
            file_name="spx_computed_greek_exposures.csv",
            mime="text/csv",
        )
    if heatmap_df is not None and not heatmap_df.empty:
        columns[2].download_button(
            "Download heatmap CSV",
            dataframe_to_csv(heatmap_df),
            file_name="spx_heatmap_matrix.csv",
            mime="text/csv",
        )

    if not raw_df.empty:
        with st.expander("Show raw options data"):
            st.dataframe(raw_df, use_container_width=True)
    if greeks_df is not None and not greeks_df.empty:
        with st.expander("Show computed Greek exposure data"):
            st.dataframe(greeks_df, use_container_width=True)
    if heatmap_df is not None and not heatmap_df.empty:
        with st.expander("Show heatmap matrix"):
            st.dataframe(heatmap_df, use_container_width=True)


def main():
    st.set_page_config(page_title="SPX GEX", layout="wide")
    st.title("SPX GEX")

    if not initialize_page_state():
        return

    try:
        st.session_state[state_key("spot")] = get_spx_spot_price()
    except Exception as error:
        st.error(f"Could not update the ^SPX price: {error}")
        return

    with st.sidebar:
        st.caption("Ticker: ^SPX")
        available_expiries = st.session_state[state_key("available_expiries")]
        selected_expiries = st.multiselect(
            "Expiration dates",
            options=available_expiries,
            key=state_key("expiry_widget"),
        )
        st.session_state[state_key("selected_expiries")] = selected_expiries

        st.button(
            "0DTE",
            use_container_width=True,
            on_click=select_and_load_preset,
            args=("0DTE",),
        )
        st.button(
            "0-3 DTE",
            use_container_width=True,
            on_click=select_and_load_preset,
            args=("0-3 DTE",),
        )
        st.button(
            "This Week",
            use_container_width=True,
            on_click=select_and_load_preset,
            args=("This Week",),
        )
        st.button(
            "This Month",
            use_container_width=True,
            on_click=select_and_load_preset,
            args=("This Month",),
        )
        if st.button("Load Selected Expiries", use_container_width=True):
            load_expiries(selected_expiries, "Loading selected ^SPX expiries...")
        st.button(
            "Load This Year",
            use_container_width=True,
            on_click=select_and_load_preset,
            args=("This Year",),
        )

        recompute_clicked = st.button(
            "Recompute / Redraw Heatmap", use_container_width=True
        )

        spot = st.session_state[state_key("spot")]
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

        metric = st.selectbox("Metric", options=list(METRIC_COLUMNS), index=0)
        color_scale_mode = st.selectbox(
            "Color Scale Mode",
            options=[
                "Auto max",
                "90th percentile",
                "95th Percentile",
                "99th Percentile",
            ],
            index=0,
        )

    raw_df = st.session_state[state_key("raw_options_df")]
    loaded_expiries = st.session_state[state_key("loaded_expiries")]
    st.caption(
        f"{TICKER} | Spot {spot:.2f} | "
        f"{len(loaded_expiries)} loaded expiries / {len(available_expiries)} available | "
        f"{len(raw_df):,} option rows"
    )

    if raw_df.empty:
        st.info(
            "This Week is preselected. Use a time-range button or Load Selected "
            "Expiries to download option-chain data."
        )

    if recompute_clicked:
        if raw_df.empty:
            st.warning("Load at least one expiration before recomputing the heatmap.")
        elif not selected_expiries:
            st.warning("Select at least one expiration date.")
        else:
            unloaded = [
                expiry for expiry in selected_expiries if expiry not in loaded_expiries
            ]
            if unloaded:
                st.warning(
                    "Load all selected expiries first. Not loaded: " + ", ".join(unloaded)
                )
            else:
                with st.spinner("Computing Greek exposures and drawing heatmap..."):
                    working_df = raw_df[
                        raw_df["expiry"].isin(selected_expiries)
                    ].copy()
                    st.session_state[state_key("greeks_df")] = compute_greek_exposures(
                        working_df,
                        spot,
                        st.session_state[state_key("risk_free_rate")],
                    )

    greeks_df = st.session_state.get(state_key("greeks_df"))
    if greeks_df is not None and not greeks_df.empty:
        heatmap_df = build_listed_strikes_heatmap_matrix(
            greeks_df, selected_expiries, metric, lower_strike, upper_strike
        )
        st.session_state[state_key("heatmap_df")] = heatmap_df
        fig, warning_message = make_heatmap_fig(
            heatmap_df,
            metric,
            color_scale_mode,
            spot,
            TICKER,
            st.session_state.get(state_key("snapshot_time")),
        )

        if warning_message:
            st.warning(warning_message)
        else:
            exposure_fig = make_total_exposure_by_strike_fig(
                greeks_df,
                selected_expiries,
                metric,
                lower_strike,
                upper_strike,
                spot,
            )
            heatmap_col, exposure_col = st.columns([3, 2])
            with heatmap_col:
                st.plotly_chart(fig, use_container_width=True)
            with exposure_col:
                if exposure_fig is not None:
                    st.plotly_chart(exposure_fig, use_container_width=True)

    render_downloads_and_tables()
    st.caption(
        "The ^SPX price cache expires every 2 minutes. Expiration lists cache for "
        "1 hour, and each expiry's option-chain/GEX input data caches for 15 minutes."
    )
    st.code(
        "Exposure convention\n"
        "Net GEX: dollar gamma for a 1% underlying move.\n"
        "Net Vanna Exposure: delta shares for a 1 volatility-point increase.\n"
        "Net Charm Exposure: delta shares gained (+) or lost (-) per calendar day.\n"
        "All net values use call exposure minus put exposure based on open interest.\n"
        "This is a positioning convention, not observed dealer positioning.",
        language=None,
    )


if __name__ == "__main__":
    main()
