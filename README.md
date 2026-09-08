# Trading Tools for pb0316

A multi-page Streamlit app for exploring options exposure, volatility term
structure, strategy payoffs, and technical stock screens. The tools use public
market data for research and do not place trades.

## Quick start

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install the pinned dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run home.py
```

Open the local URL shown by Streamlit and select a tool from the sidebar.
Internet access is required for Yahoo Finance data and embedded TradingView
charts.

### Windows launcher

If the dependencies are already installed, double-click `launch_app.bat`. The
launcher looks for Streamlit in the active environment, common Anaconda or
Miniconda locations, and Python installations on `PATH`. It does not create an
environment or install packages.

## Included tools

### Options Greek Exposure Heatmap

Loads option chains for a ticker and estimates Black-Scholes gamma, Vanna, and
Charm exposure by strike and expiry. The selected metric drives both the
heatmap and total-by-strike chart, and the underlying calculations can be
downloaded as CSV files.

- Net GEX: approximate dollar gamma for a 1% underlying move.
- Net Vanna Exposure: delta-equivalent shares for a one-percentage-point
  volatility increase.
- Net Charm Exposure: delta-equivalent shares gained or lost per calendar day.

The calculations use open interest and a call-positive/put-negative convention;
they are estimates, not observations of dealer positions.

### SPX GEX

A dedicated `^SPX` version of the exposure heatmap. Presets load 0DTE, the next
three calendar days, the rest of the week, month, or year, as well as manually
selected expiries. Date ranges use New York market time and include today.

The SPX spot price is cached for two minutes, expiry lists for one hour, and
individual option chains for 15 minutes to reduce repeated Yahoo Finance calls.

### Calendar Spread Investigator

Compares same-strike call or put implied volatility across listed expirations.
The analysis window supports 60, 90, 120, or 180 DTE and includes:

- term structures for the selected and nearby strikes;
- an optional next-earnings marker;
- expiration-pair heatmaps for signed IV gap, IV gap per day, or implied
  forward IV;
- a top-five table of positive front-minus-back IV gaps, estimated deltas, and
  OptionStrat diagram links;
- interpolated 30-day ATM IV and realized volatility over 5, 10, 20, 30, 50,
  and 100 trading days.

Historical IV Rank and IV Percentile are intentionally omitted because Yahoo
Finance does not provide the historical option-chain snapshots they require.

### Options Strategy Simulator

Builds long and short calls or puts, vertical spreads, short straddles and
strangles, iron condors, and single or double calendar/diagonal spreads from
Yahoo Finance option chains. It provides:

- strike and expiration selection with live midpoint-based entry pricing;
- Black-Scholes-Merton P&L curves for a selected date and front expiry;
- independent front- and back-expiry IV scenarios for time spreads;
- automatic chart ranges based on expiration breakevens;
- max profit/loss, entry Greeks, Vanna, Charm, Vomma, Vega/risk and Theta/risk
  measures, and IV-shock scenarios;
- quote-quality and model diagnostics.

The simulator is analytical only. It neither recommends nor submits orders.

### RSI Trend Screener

Screens a Nasdaq universe for weekly RSI trend setups using price, volume,
moving averages, and RSI conditions. Results can be filtered by price and
sector and reviewed with embedded TradingView charts.

### TTM Squeeze Screener

Finds daily or weekly consolidation setups using Bollinger Bands, Keltner
Channels, moving-average trend confirmation, and momentum. The Bollinger Band
width, price range, sector, and result count are configurable.

### Volatility-Momentum Surge

Screens for upside breakouts or downside breakdowns using price-change and
volume Z-scores, moving-average structure, average volume, and sector filters.
Results are displayed in a paginated TradingView chart grid.

## Project layout

```text
home.py                              Streamlit entry point
launch_app.bat                       Windows launcher
requirements.txt                     pinned Python dependencies
nasdaq_screener_1779080945125.csv    local screener universe
pages/
  calendar_spread_investigator.py
  gex_heatmap.py
  options_strategy_simulator.py
  rsi_screener.py
  spx_gex.py
  ttm_screener.py
  vomo_screener.py
tests/
  test_calendar_page.py
  test_calendar_spread.py
  test_options_strategy_simulator.py
```

Each file under `pages/` is intentionally self-contained project code. It may
use the Python standard library and packages in `requirements.txt`, but should
not depend on another private project folder. This keeps individual pages easy
to copy into another Streamlit app.

The local Nasdaq CSV supplies `Symbol`, `Name`, `Market Cap`, `Sector`, and
`Industry` metadata to the screeners. Replace it when a newer universe is
needed, while preserving those columns.

## Tests

Run the analytics and Streamlit page tests from the project root:

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic or mocked data and do not require a live market-data
request.

## Data and model limitations

- Yahoo Finance data can be delayed, incomplete, throttled, or unavailable.
- Embedded TradingView charts depend on third-party scripts and network access.
- Black-Scholes-based outputs depend on simplifying assumptions and user inputs.
- Open-interest exposure is not a direct measurement of market-maker inventory.
- The bundled Nasdaq metadata is a point-in-time universe and can become stale.

These tools are for research and education only and are not financial advice.
