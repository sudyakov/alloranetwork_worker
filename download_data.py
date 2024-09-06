import requests
import pandas as pd
import os
from glob import glob
from datetime import datetime, timedelta
from utils import prepare_data

# List of pairs for which data needs to be downloaded
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ARBUSDT"]

# Dictionary of intervals and corresponding periods in days
intervals_periods = {
    # "1s": 1,    # 1 second - for the last day
    "1m": 30,   # 1 minute - for the last 30 days
    "3m": 90,   # 3 minutes - for the last 90 days
    # "5m": 180,  # 5 minutes - for the last 180 days
    # "15m": 180, # 15 minutes - for the last year
    # "30m": 365, # 30 minutes - for the last year
    # "1h": 365,  # 1 hour - for the last year
    # "1d": 730,  # 1 day - for the last 2 years
    # "3d": 1095, # 3 days - for the last 3 years
    # "1w": 1825, # 1 week - for the last 5 years
    # "1mo": 3650 # 1 month - for the last 10 years
}

seq_length = 60  # или другое подходящее значение

def load_binance_data_from_folder(folder):
    data = {}
    csv_files = glob(os.path.join(folder, "*.csv"))
    for file in csv_files:
        symbol, interval, _ = os.path.basename(file).split("_")
        df = pd.read_csv(file, parse_dates=["timestamp"], index_col="timestamp")
        data[(symbol, interval)] = df
    return data

def get_binance_data(symbol, interval, start_time=None, end_time=None, limit=1000):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        if start_time:
            url += f"&startTime={start_time}"
        if end_time:
            url += f"&endTime={end_time}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume",
                                         "close_time", "quote_asset_volume", "number_of_trades",
                                         "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
        df = df[["open_time", "open", "high", "low", "close", "volume"]]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        return df.set_index("timestamp")
    except requests.exceptions.RequestException as e:
        print(f"Error loading data for pair {symbol} and interval {interval}: {e}")
        return None

def get_all_binance_data(symbol, interval, start_date, end_date):
    csv_filename = f"binance-data/{symbol}_{interval}_data.csv"
    
    if not os.path.exists(csv_filename):
        df_existing = pd.DataFrame()
    else:
        df_existing = pd.read_csv(csv_filename, parse_dates=["timestamp"], index_col="timestamp")
        start_date = df_existing.index[-1] + pd.Timedelta(minutes=1)
    
    if start_date >= end_date:
        return df_existing
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df_list = []
    
    for start_time, end_time in zip(date_range, date_range[1:]):
        start_time = int(start_time.timestamp() * 1000)
        end_time = int(end_time.timestamp() * 1000)
        df = get_binance_data(symbol, interval, start_time, end_time)
        if df is None or df.empty:
            print(f"Failed to load data for pair {symbol} and interval {interval} from {start_time} to {end_time}")
            break
        df_list.append(df)
    
    if df_list:
        df_new = pd.concat(df_list)
        df_final = pd.concat([df_existing, df_new]).drop_duplicates()
        df_final.to_csv(csv_filename)
        return df_final
    return df_existing

def save_to_csv(df, filename):
    symbol, interval = os.path.basename(filename).split('_')[:2]
    df = df.reset_index()
    df['symbol'] = symbol
    df['interval'] = interval
    df = df[['timestamp', 'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume']]
    df.to_csv(filename, index=False)
    print(f"Data saved to file {filename}")

def save_combined_dataset(data, filename):
    combined_data = pd.concat(data.values(), ignore_index=True)
    combined_data.to_csv(filename, index=False)
    print(f"Combined dataset saved to file {filename}")

def load_data():
    return pd.read_csv("combined_dataset.csv")

def update_data(df, scaler, symbols):
    new_data = {symbol: get_binance_data(symbol) for symbol in symbols}
    
    df = pd.concat([df] + list(new_data.values())).drop_duplicates().reset_index(drop=True)
    
    x_new, y_new, _ = prepare_data(new_data, seq_length, symbols)
    return df, x_new, y_new

# Create "binance-data" folder if it doesn't exist
if not os.path.exists("binance-data"):
    os.makedirs("binance-data")

# Load data for each pair and interval
for symbol in symbols:
    for interval, period in intervals_periods.items():
        print(f"Starting data download for pair {symbol} and interval {interval}")
        start_date = pd.Timestamp.now().floor('D') - pd.Timedelta(days=period)
        end_date = pd.Timestamp.now().floor('D')
        df = get_all_binance_data(symbol, interval, start_date, end_date)
        if df is not None and not df.empty:
            print(f"Data download for pair {symbol} and interval {interval} completed")
            print(f"First row of data for pair {symbol} and interval {interval}:")
            print(df.head(1))
            print(f"Last row of data for pair {symbol} and interval {interval}:")
            print(df.tail(1))
            filename = f"binance-data/{symbol}_{interval}_data.csv"
            save_to_csv(df, filename)
        else:
            print(f"Failed to load data for pair {symbol} and interval {interval}")
        print("------------------------")

# Load data from "binance-data" folder recursively
folder = "binance-data"
binance_data = load_binance_data_from_folder(folder)

# Save combined dataset to file
save_combined_dataset(binance_data, "combined_dataset.csv")
