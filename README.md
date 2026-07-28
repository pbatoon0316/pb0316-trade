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

### Options Greek Exposure Heatmap

Downloads option chain data with `yfinance` and estimates Black-Scholes gamma, Vanna, and Charm exposures across listed strikes and expirations. The sidebar metric selector switches both the heatmap and the total-by-strike chart between Net GEX, Net Vanna Exposure, and Net Charm Exposure. Computed CSV downloads include all three metrics, while the heatmap CSV contains the currently selected metric.

Net GEX is expressed as approximate dollar gamma for a 1% underlying move, Vanna exposure as delta-equivalent shares for a one-percentage-point volatility increase, and Charm exposure as delta-equivalent shares gained or lost per calendar day. These are open-interest-based estimates using a call-positive/put-negative positioning convention, not observed dealer positions.

### RSI Trend Screener

Screens a Nasdaq ticker universe for weekly RSI trend setups using price, volume, moving averages, and RSI conditions. Results can be filtered by price and sector, then reviewed with embedded TradingView charts.

### Weekly Consolidation Screener

Finds weekly squeeze/consolidation setups using Bollinger Band and Keltner Channel logic, with trend confirmation from moving averages and momentum. The sidebar filters results by price and sector and displays matching weekly charts.

### Volatility-Momentum Surge

Screens for recent volatility-momentum breakouts or breakdowns using price-change Z-score, volume Z-score, moving-average trend structure, volume average, and sector filters. Results are shown in the sidebar and charted in a three-column TradingView grid.

## Notes

Data is downloaded from Yahoo Finance through `yfinance`, so availability, throttling, and data quality may vary. These tools are for research workflows and are not financial advice.
