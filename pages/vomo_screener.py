import pandas as pd
import yfinance as yf
import warnings
import streamlit as st
from urllib.parse import quote


######################
# Set the display option to show 2 decimal places
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title='Volatility-Momentum Surge',
                   page_icon='⚡', 
                   layout="wide")

@st.cache_data(ttl='1d')
def download_metadata():
    url = 'nasdaq_screener_1779080945125.csv'
    metadata = pd.read_csv(url)
    return metadata

def get_tickers(metadata):
    tickers = metadata['Symbol'].dropna().astype(str).str.strip().str.replace('/', '-', regex=False)
    return tickers[tickers != ''].drop_duplicates().tolist()

def normalize_price_data(data, tickers):
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        return data

    if len(tickers) != 1:
        return pd.DataFrame()

    ticker = tickers[0]
    normalized = data.copy()
    normalized.columns = pd.MultiIndex.from_product([normalized.columns, [ticker]])
    return normalized

@st.cache_data(ttl='1hr')
def download_data():
    url = 'nasdaq_screener_1779080945125.csv'
    stocks = pd.read_csv(url)
    tickers = get_tickers(stocks)
    batch_size = 200
    chunks = []
    batch_count = (len(tickers) + batch_size - 1) // batch_size

    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        batch_number = (start // batch_size) + 1
        print(f'Downloading batch {batch_number} out of {batch_count} ({len(batch)} tickers)', flush=True)
        try:
            batch_data = yf.download(
                batch,
                period='1y',
                interval='1d',
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception:
            continue

        batch_data = normalize_price_data(batch_data, batch)
        if batch_data is not None and not batch_data.empty:
            chunks.append(batch_data)

    if not chunks:
        return pd.DataFrame()

    data = pd.concat(chunks, axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    return data

def scanner(data, threshold=2, lookback_days=20, screen_direction='Breakouts'):
    columns = [
        'Signal',
        'ticker',
        '%change_zscore',
        'volume_zscore',
        'volume_average',
        'Volume(M)',
        'SMA20',
        'SMA50',
        'SMA100',
        'SMA200',
    ]
    if data.empty or not isinstance(data.columns, pd.MultiIndex):
        empty_breakouts = pd.DataFrame(columns=columns)
        empty_breakouts.index.name = 'Date'
        return empty_breakouts

    tickers = list(data.columns.get_level_values(1).unique())
    breakout_rows = []
    period = 20
    longest_sma_period = 200
    dates = pd.Index(data.index).dropna().sort_values().unique()
    if len(dates) == 0:
        empty_breakouts = pd.DataFrame(columns=columns)
        empty_breakouts.index.name = 'Date'
        return empty_breakouts

    if lookback_days <= 0:
        screen_dates = dates[-1:]
    else:
        screen_dates = dates[-lookback_days:]

    ticker_count = len(tickers)
    print(f'Screening {ticker_count} tickers for {screen_direction} across {len(screen_dates)} recent trading days...', flush=True)

    for idx_ticker, ticker in enumerate(tickers, start=1):
        print(f'Screening {idx_ticker} out of {ticker_count}: {ticker}', flush=True)
        if ticker not in data.columns.get_level_values(1):
            continue

        df = data.loc[:, (slice(None), ticker)].copy()
        df.columns = df.columns.droplevel(1)
        if 'Close' not in df.columns or 'Volume' not in df.columns:
            continue

        df = df[['Close', 'Volume']].dropna()
        if len(df) < longest_sma_period:
            continue

        df['Signal'] = screen_direction
        df['ticker'] = ticker
        df['SMA20'] = df['Close'].rolling(20, min_periods=20).mean()
        df['SMA50'] = df['Close'].rolling(50, min_periods=50).mean()
        df['SMA100'] = df['Close'].rolling(100, min_periods=100).mean()
        df['SMA200'] = df['Close'].rolling(200, min_periods=200).mean()

        df['%change'] = df['Close'].pct_change()
        previous_changes = df['%change'].shift(1)
        change_mean = previous_changes.rolling(period, min_periods=period).mean()
        change_std = previous_changes.rolling(period, min_periods=period).std()
        df['%change_zscore'] = (df['%change'] - change_mean) / change_std

        df['Volume(M)'] = df['Volume'] / 1000000
        previous_volume = df['Volume(M)'].shift(1)
        volume_mean = previous_volume.rolling(period, min_periods=period).mean()
        volume_std = previous_volume.rolling(period, min_periods=period).std()
        df['volume_average'] = volume_mean
        df['volume_zscore'] = (df['Volume(M)'] - volume_mean) / volume_std

        df = df[['Signal','ticker','Close','%change_zscore','volume_zscore','volume_average','Volume(M)','SMA20','SMA50','SMA100','SMA200']]
        recent_df = df[df.index.isin(screen_dates)]
        if screen_direction == 'Breakdowns':
            signal_filter = (
                (recent_df['%change_zscore'] < -threshold)
                & (recent_df['volume_zscore'] > threshold)
                & (recent_df['Close'] < recent_df['SMA20'])
                & (recent_df['SMA20'] < recent_df['SMA50'])
                & (recent_df['SMA50'] < recent_df['SMA100'])
                & (recent_df['SMA100'] < recent_df['SMA200'])
            )
        else:
            signal_filter = (
                (recent_df['%change_zscore'] > threshold)
                & (recent_df['volume_zscore'] > threshold)
                & (recent_df['Close'] > recent_df['SMA20'])
                & (recent_df['SMA20'] > recent_df['SMA50'])
                & (recent_df['SMA50'] > recent_df['SMA100'])
                & (recent_df['SMA100'] > recent_df['SMA200'])
            )

        breakout_df = recent_df[
            signal_filter
        ].dropna(subset=['%change_zscore', 'volume_zscore', 'SMA20', 'SMA50', 'SMA100', 'SMA200'])

        breakout_df = breakout_df.drop(columns=['Close'])

        if not breakout_df.empty:
            breakout_rows.append(breakout_df)

    if not breakout_rows:
        empty_breakouts = pd.DataFrame(columns=columns)
        empty_breakouts.index.name = 'Date'
        return empty_breakouts

    breakouts = pd.concat(breakout_rows)
    breakouts = breakouts.sort_values('volume_average', ascending=False)

    return breakouts

CHART_HEIGHT = 320

def plot_ticker_iframe_src(ticker, metadata, breakouts):

    ticker_metadata = metadata[metadata['Symbol'].astype(str).str.replace('/', '-', regex=False) == ticker]
    company_sector = 'Unknown'
    if not ticker_metadata.empty and pd.notna(ticker_metadata['Sector'].iloc[0]):
        company_sector = ticker_metadata['Sector'].iloc[0]

    ticker_breakouts = breakouts[breakouts['ticker'] == ticker]
    avg_vol = ticker_breakouts['volume_average'].iloc[0] if not ticker_breakouts.empty else 0

    st.markdown(f'''{round(avg_vol,2)}M - {ticker} - {company_sector} [[Finviz]](https://finviz.com/quote.ashx?t={ticker}&p=d) [[Profitviz]](https://profitviz.com/{ticker})''')
    
    fig_html = f'''
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            html,
            body {{
                width: 100%;
                height: {CHART_HEIGHT}px;
                margin: 0;
                padding: 0;
                overflow: hidden;
                background: #ffffff;
            }}
            .tradingview-widget-container,
            .tradingview-widget-container__widget {{
                width: 100%;
                height: {CHART_HEIGHT}px;
                margin: 0;
                padding: 0;
                overflow: hidden;
            }}
        </style>
    </head>
    <body>
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
        {{
        "width": "100%",
        "height": "{CHART_HEIGHT}",
        "symbol": "{ticker}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "backgroundColor": "rgba(255, 255, 255, 1)",
        "gridColor": "rgba(0, 0, 0, 0.06)",
        "hide_top_toolbar": true,
        "allow_symbol_change": false,
        "save_image": false,
        "calendar": false,
        "studies": [
            "STD;SMA",
            "STD;MA%Ribbon"
        ],
        "support_host": "https://www.tradingview.com"
        }}
        </script>
    </div>
    <!-- TradingView Widget END -->
    </body>
    </html>
    '''
    return f'data:text/html;charset=utf-8,{quote(fig_html)}'

def add_metadata_columns(breakouts, metadata):
    if breakouts.empty:
        return breakouts

    metadata_lookup = metadata.copy()
    metadata_lookup['ticker'] = metadata_lookup['Symbol'].astype(str).str.replace('/', '-', regex=False)
    metadata_lookup = metadata_lookup[['ticker', 'Name', 'Sector', 'Industry']]
    return breakouts.merge(metadata_lookup, how='left', on='ticker')

def prepare_breakouts_for_display(breakouts, metadata, min_volume_average, sector_filter):
    columns = [
        'Signal',
        'ticker',
        'Name',
        'Sector',
        'Industry',
        '%change_zscore',
        'volume_zscore',
        'volume_average',
        'Volume(M)',
        'SMA20',
        'SMA50',
        'SMA100',
        'SMA200',
    ]
    if breakouts.empty:
        return pd.DataFrame(columns=columns).rename_axis('Date')

    display_breakouts = breakouts.reset_index()
    if 'Date' not in display_breakouts.columns:
        display_breakouts = display_breakouts.rename(columns={display_breakouts.columns[0]: 'Date'})

    display_breakouts = display_breakouts.sort_values(by=['Date','volume_average'], ascending=False)
    display_breakouts = display_breakouts.drop_duplicates(subset=['Date', 'ticker'])
    display_breakouts = add_metadata_columns(display_breakouts, metadata)
    display_breakouts = display_breakouts[display_breakouts['volume_average'] > min_volume_average]
    if sector_filter != 'All':
        display_breakouts = display_breakouts[display_breakouts['Sector'] == sector_filter]

    display_breakouts = display_breakouts.set_index('Date')
    return display_breakouts

def get_sector_options(raw_breakouts, metadata):
    if raw_breakouts.empty:
        return ['All']

    breakouts_with_metadata = add_metadata_columns(raw_breakouts.reset_index(), metadata)
    sectors = breakouts_with_metadata['Sector'].dropna().sort_values().unique().tolist()
    return ['All'] + sectors

######################

empty_breakouts = pd.DataFrame(
    columns=['Signal', 'ticker', '%change_zscore', 'volume_zscore', 'volume_average', 'Volume(M)', 'SMA20', 'SMA50', 'SMA100', 'SMA200']
).rename_axis('Date')
raw_breakouts_by_signal = st.session_state.get('raw_breakouts_by_signal', {})
legacy_raw_breakouts = st.session_state.get('raw_breakouts', empty_breakouts)
if not legacy_raw_breakouts.empty and 'Breakouts' not in raw_breakouts_by_signal:
    raw_breakouts_by_signal['Breakouts'] = legacy_raw_breakouts
metadata = download_metadata()
st.session_state.metadata = metadata


##### Data download & Calculations #####

with st.sidebar:
    st.caption(f'{len(metadata)} tickers in universe')

    screen_direction = st.radio('Signal', ['Breakouts', 'Breakdowns'], horizontal=True)
    raw_breakouts = raw_breakouts_by_signal.get(screen_direction, empty_breakouts)
    threshold = st.number_input('Z-Score (default=2)', value=2.0)
    lookback = st.number_input('Lookback days (default=10)', value=10, min_value=0, max_value=120)
    volavg = st.number_input('Volume Average', value=5.0)

    data = st.session_state.get('data', pd.DataFrame())
    if data.empty:
        st.caption('Price data not loaded')
    else:
        st.caption(f'Price data loaded: {data.shape[0]} rows x {data.shape[1]} columns')

    start_screen = st.button('Start screen', type='primary', use_container_width=True)
    if start_screen:
        with st.spinner('Downloading price data and screening recent signals...'):
            st.session_state.data = download_data()
            data = st.session_state.data
            if data.empty:
                st.warning('No price data was downloaded. Yahoo Finance may be rate limiting requests; try again after the cache expires or reduce the ticker universe.')
                raw_breakouts = empty_breakouts
            else:
                raw_breakouts = scanner(data, threshold, lookback, screen_direction)

            st.session_state.raw_breakouts = raw_breakouts
            raw_breakouts_by_signal[screen_direction] = raw_breakouts
            st.session_state.raw_breakouts_by_signal = raw_breakouts_by_signal

    sector_options = get_sector_options(raw_breakouts, metadata)
    sector_filter = st.selectbox('Sector', sector_options)
    charts_per_page = st.number_input('Charts per page', value=12, min_value=3, max_value=24, step=3)

    breakouts = prepare_breakouts_for_display(raw_breakouts, metadata, volavg, sector_filter)
    chart_tickers = breakouts.ticker.unique().tolist() if not breakouts.empty else []
    if chart_tickers:
        total_pages = max(1, (len(chart_tickers) + charts_per_page - 1) // charts_per_page)
        chart_page = st.number_input('Chart page', value=1, min_value=1, max_value=total_pages)
        start_idx = (chart_page - 1) * charts_per_page
        end_idx = start_idx + charts_per_page
        visible_tickers = chart_tickers[start_idx:end_idx]
    else:
        visible_tickers = []

    st.markdown(screen_direction)
    if breakouts.empty:
        if start_screen:
            st.info(f'No {screen_direction.lower()} found for the selected Z-score, volume average, sector, and lookback settings.')
        else:
            st.info('Press "Start screen" to look for recent volatility-momentum signals.')
    st.dataframe(breakouts, hide_index=False)



##### Plotting charts #####

if visible_tickers:
    st.caption(f'Showing charts {start_idx + 1}-{min(end_idx, len(chart_tickers))} of {len(chart_tickers)}')

chart_columns = st.columns(3)
for i, ticker in enumerate(visible_tickers):
    with chart_columns[i % 3]:
        try:
            fig = plot_ticker_iframe_src(ticker, metadata, breakouts)
            st.iframe(fig, height=CHART_HEIGHT)
        except:
            st.markdown(f'{ticker} - [[Finviz]](https://finviz.com/quote.ashx?t={ticker}&p=d) [[Profitviz]](https://profitviz.com/{ticker})')
