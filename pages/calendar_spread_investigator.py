from __future__ import annotations

import pandas as pd
import streamlit as st

from calendar_spread.analytics import (
    METRICS,
    add_pair_deltas,
    available_strikes,
    build_pair_metrics,
    compute_historical_volatility,
    directional_atm_strike,
    interpolated_atm_iv,
    optionstrat_calendar_url,
    rank_pairs,
    strike_term_structure,
    term_structure_strikes,
)
from calendar_spread.charts import (
    historical_volatility_figure,
    pair_heatmap_figure,
    term_structure_figure,
)
from calendar_spread.data import (
    load_calendar_snapshot,
    load_next_earnings,
    load_price_history,
)


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
