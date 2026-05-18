# Options GEX Heatmap - Listed Strikes MVP

This Streamlit app downloads option chain data with `yfinance`, estimates Black-Scholes gamma, and displays gamma exposure across option expirations.

The MVP implements **Listed Strikes Only** mode. It preserves the actual listed strikes for each expiration. Missing heatmap cells mean that strike is not listed for that expiry, not zero GEX.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## MVP Limitations

- Uses `yfinance`, so data quality and availability may vary.
- Greeks are estimated from Black-Scholes using `yfinance` implied volatility.
- Dividends are ignored.
- Risk-free rate is approximated from the `yfinance` `SR3=F` futures price using `(100 - price) / 100`.
- `SR3=F` is used as a simple short-rate proxy, not an expiry-matched rate curve.
- Listed Strikes Only mode preserves actual listed strikes.
- Missing heatmap cells mean no listed strike for that expiry, not zero GEX.
- Not financial advice.
