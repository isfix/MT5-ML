#--- Single Symbol Data Acquisition ---
import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime

# Configuration
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Create data directory
os.makedirs('data', exist_ok=True)

# Initialize MT5 connection
if not mt5.initialize():
    print("MT5 initialization failed")
    quit()

print(f"Acquiring data for {SYMBOL}...")

# Get historical data
rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, START_DATE, END_DATE)

if rates is None:
    print(f"Failed to get rates for {SYMBOL}")
    quit()

# Convert to DataFrame
df = pd.DataFrame(rates)

# Time standardization to UTC
df['time'] = pd.to_datetime(df['time'], unit='s')
df['time'] = df['time'].dt.tz_localize('UTC')

# Clean column names
df.rename(columns={
    'open': 'Open',
    'high': 'High', 
    'low': 'Low',
    'close': 'Close',
    'tick_volume': 'Volume'
}, inplace=True)

# Set time as index
df.set_index('time', inplace=True)

# Save data
df.to_csv('data/raw_dataset.csv')

print(f"=== Data Acquisition Complete ===")
print(f"{SYMBOL}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
print(f"Data saved to: data/raw_dataset.csv")

mt5.shutdown()