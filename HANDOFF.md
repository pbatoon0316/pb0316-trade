# Trading Tools App Handoff

This repository is a small Streamlit multipage app for personal trading research. It currently contains one lightweight home dashboard and three heavier analytical tools under `pages/`.

The main goal of this document is to let a future human programmer or LLM coding session understand the app quickly without rereading every file from scratch.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the multipage Streamlit app from the repo root:

```bash
streamlit run home.py
```

The checked-in `README.md` still says `streamlit run app.py`, but the current entry file in this repo is `home.py`.

## High-Level Structure

```text
home.py
pages/
  gex_heatmap.py
  rsi_screener.py
  ttm_screener.py
nasdaq_screener_1779080945125.csv
requirements.txt
.streamlit/config.toml
```

### `home.py`

Purpose: the lightweight home dashboard.

This page intentionally acts as a "wall" or neutral landing page. It prevents users from landing directly on a heavy tool that downloads market data, embeds charts, or computes screeners as soon as Streamlit starts.

Current behavior:

- Shows a simple title and instruction to select a tool from the sidebar.
- Does not download data.
- Does not import the heavy page modules.
- Keeps app startup cheap and predictable.

Decision point:

- Keep `home.py` minimal unless there is a strong reason to add shared navigation or status. Avoid adding imports from the page modules here, because that could defeat the purpose of the wall.

### `pages/gex_heatmap.py`

Purpose: options chain analysis tool for examining general options positioning over time.

Primary workflow:

1. User enters a ticker in the sidebar.
2. App downloads the spot price and available option expirations from `yfinance`.
3. App loads the nearest expiries by default.
4. User can select and load additional expiries.
5. App computes approximate Black-Scholes gamma and gamma exposure.
6. App displays a Plotly heatmap by strike and expiry, plus a side chart of total net GEX by strike.
7. User can download raw options data, computed GEX, and the heatmap matrix as CSV.

Important implementation details:

- Uses `st.cache_data` for `yfinance` calls.
- Uses `st.session_state` to store the active ticker, spot price, expiries, raw option chain data, computed GEX, and heatmap matrix.
- Uses `SR3=F` from `yfinance` as a rough short-rate proxy. If unavailable, falls back to a 5% risk-free rate.
- Calculates gamma from Black-Scholes using `impliedVolatility`, then estimates dollar gamma exposure per 1% move.
- Preserves listed strikes only. Missing heatmap cells mean the strike was not listed for that expiry, not that GEX is zero.

Decision points:

- Whether to keep `yfinance` as the data source or swap to a more reliable options data provider.
- Whether to support persisted snapshots. Today, snapshots live only in session state and downloads.
- Whether to add expiry-matched rates, dividends, or other model refinements.
- Whether to rename or reconcile user-facing references to "heapmap" versus the actual file name `gex_heatmap.py`.

### `pages/rsi_screener.py`

Purpose: weekly trending RSI screener.

Primary workflow:

1. Loads metadata from `nasdaq_screener_1779080945125.csv`.
2. Cleans symbols and keeps the top tickers by market cap.
3. Downloads one year of weekly OHLCV data with `yfinance`.
4. Computes weekly indicators, including moving averages, volume metrics, Keltner/Bollinger context, Awesome Oscillator, RSI, and RSI SMA.
5. Filters for tickers where RSI SMA is rising, RSI SMA is still below 50, and price is above the 50-period EMA.
6. Merges results with metadata.
7. Provides price and sector filters in the sidebar.
8. Embeds TradingView weekly charts for selected results.

Important implementation details:

- Uses `finta.TA.RSI`.
- Uses `st.cache_resource` for metadata and weekly data download.
- Uses `st.cache_data` for scanner output.
- Stores downloaded weekly price data in `st.session_state["data_wk"]`.
- Imports `streamlit.components.v1 as components` and uses `components.html` to render TradingView widgets.

Decision points:

- The screeners hard-code the top 1000 symbols by market cap. This balances coverage and download time, but it is a performance/data-quality tradeoff.
- The metadata CSV is local and may become stale. Updating it changes the screener universe.
- The scanner logic is currently embedded directly in the page file; extracting shared helpers would make future changes easier.

### `pages/ttm_screener.py`

Purpose: weekly TTM squeeze/consolidation screener.

Primary workflow:

1. Loads and cleans Nasdaq metadata from the local CSV.
2. Downloads one year of weekly OHLCV data with `yfinance`.
3. Computes EMA, Keltner Channel, Bollinger Band, Awesome Oscillator, volume, and squeeze-related fields.
4. Filters for tickers currently in a squeeze where price is above the 50-period EMA and the Awesome Oscillator is rising.
5. Merges results with metadata.
6. Provides price and sector filters.
7. Embeds TradingView weekly charts for selected results.

Important implementation details:

- This file is very similar to `rsi_screener.py`.
- It also stores downloaded weekly data in `st.session_state["data_wk"]`.
- Uses `threads=True` in the `yfinance.download` call.
- The page title is `Weekly Consolidation Screener`, and the sidebar expander says `Weekly Squeeze Reults`.

Decision points:

- Consider extracting shared metadata loading, weekly data downloading, chart rendering, price/sector filters, and layout helpers into a common module.
- Consider using distinct session state keys for the two screeners, such as `rsi_data_wk` and `ttm_data_wk`, if their data requirements diverge.
- Decide whether the page should be named or labeled "TTM Squeeze", "Weekly Consolidation", or both.

## Shared Data And Dependencies

### Local CSV

`nasdaq_screener_1779080945125.csv` is the local ticker universe for both weekly screeners.

Expected columns used by the code:

- `Symbol`
- `Name`
- `Market Cap`
- `Sector`
- `Industry`

The code replaces `/` with `-` in symbols to make tickers compatible with Yahoo Finance conventions.

### Python dependencies

Current `requirements.txt`:

```text
streamlit
yfinance==1.3.0
pandas
numpy
plotly
finta
```

Runtime services used:

- Yahoo Finance via `yfinance`.
- TradingView embedded widget scripts.
- Finviz and Profitviz links for manual ticker research.

## Caching And Loading Behavior

The app relies heavily on Streamlit caching and session state.

The home page should stay light. The heavy work happens only after navigating to a specific page.

`gex_heatmap.py`:

- Caches available expiries for 1 hour.
- Caches spot prices for 5 minutes.
- Caches SR3 risk-free proxy for 15 minutes.
- Caches each option chain by ticker/expiry for 15 minutes.
- Uses explicit "Load Selected Expiries" and "Recompute / Redraw Heatmap" buttons for user control.

`rsi_screener.py` and `ttm_screener.py`:

- Cache metadata and downloaded weekly data for 12 hours.
- Automatically download data when the page is opened and `st.session_state["data_wk"]` is missing.
- Reuse `data_wk` across pages if the session key already exists.

## Known Gotchas

- `README.md` references `app.py`, but this repo uses `home.py`.
- `rsi_screener.py` and `ttm_screener.py` share the same session state key, `data_wk`. This is currently acceptable because both use the same weekly ticker data, but it may become confusing later.
- The two screeners contain duplicated code. Fixes may need to be made in both files unless helpers are extracted.
- The screeners can fail or behave awkwardly if a filter leaves zero rows, because some sidebar inputs use dataframe min/max values.
- Several sidebar expander labels have typos such as `Reults`.
- Page icons currently display as mojibake in the source files, likely from an encoding issue.
- All market data from `yfinance` can be incomplete, delayed, throttled, or structurally different across tickers.
- Embedded TradingView widgets depend on network access and third-party script loading.
- None of the tools should be treated as financial advice.

## Suggested Next Refactors

1. Update `README.md` to run `streamlit run home.py`.
2. Extract shared screener utilities into something like `utils/screener_common.py`.
3. Rename shared session keys or intentionally document that the two screeners reuse weekly data.
4. Add empty-result guards around sidebar min/max controls.
5. Fix text typos and page icon encoding.
6. Add a small manual QA checklist for each page.
7. Consider adding snapshot persistence for GEX outputs if repeated comparison over time matters.

## Good Starting Prompts For Future Sessions

Use these when handing the repo to another LLM or programmer:

- "Read `HANDOFF.md` first, then inspect only the files relevant to the requested change."
- "Preserve `home.py` as a lightweight wall. Do not import heavy page modules from it."
- "If changing weekly screener logic, check both `rsi_screener.py` and `ttm_screener.py` because they duplicate structure."
- "If changing data loading, be mindful of Streamlit cache decorators and `st.session_state` keys."
- "If changing GEX calculations, document the model assumption in the UI or README."

## Manual QA Checklist

After changes, run:

```bash
streamlit run home.py
```

Check:

- Home page loads quickly and does not download market data.
- Sidebar shows all three tools.
- GEX Heatmap can load a liquid ticker such as `SPY`, load expiries, recompute, render charts, and download CSVs.
- RSI Screener loads weekly results, price filters work, sector filter works, mobile toggle works, and charts render or fall back to links.
- TTM Screener loads weekly results, price filters work, sector filter works, mobile toggle works, and charts render or fall back to links.
