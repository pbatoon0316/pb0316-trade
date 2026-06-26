# Trading Tools for pb0316

This Streamlit app contains a small collection of market screeners and options analysis tools. Launch the app from `home.py` and choose a tool from the Streamlit sidebar.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run home.py
```

## Pages

### Home

The home page is a simple entry point for the app. Use the sidebar to navigate to the available trading tools.

### Options GEX Heatmap

Downloads option chain data with `yfinance`, estimates Black-Scholes gamma exposure, and visualizes GEX across listed strikes and expirations. The page includes controls for expirations, color scaling, and GEX metric selection.

### RSI Trend Screener

Screens a Nasdaq ticker universe for weekly RSI trend setups using price, volume, moving averages, and RSI conditions. Results can be filtered by price and sector, then reviewed with embedded TradingView charts.

### Weekly Consolidation Screener

Finds weekly squeeze/consolidation setups using Bollinger Band and Keltner Channel logic, with trend confirmation from moving averages and momentum. The sidebar filters results by price and sector and displays matching weekly charts.

### Volatility-Momentum Surge

Screens for recent volatility-momentum breakouts or breakdowns using price-change Z-score, volume Z-score, moving-average trend structure, volume average, and sector filters. Results are shown in the sidebar and charted in a three-column TradingView grid.

## Notes

Data is downloaded from Yahoo Finance through `yfinance`, so availability, throttling, and data quality may vary. These tools are for research workflows and are not financial advice.
