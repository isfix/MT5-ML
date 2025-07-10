#--- Multi-Symbol Data Acquisition Module ---
import MetaTrader5 as mt5
import pandas as pd
import os
from config import SYMBOLS, TIMEFRAME, START_DATE, END_DATE, DATA_DIR

# Create data directory
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize MT5 connection
if not mt5.initialize():
    print("MT5 initialization failed")
    quit()

print(f"Acquiring data for {len(SYMBOLS)} symbols...")
all_data = []

for symbol in SYMBOLS:
    print(f"\nProcessing {symbol}...")
    
    # Get historical data
    rates = mt5.copy_rates_range(symbol, TIMEFRAME, START_DATE, END_DATE)
    
    if rates is None:
        print(f"Failed to get rates for {symbol}")
        continue
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    
    # Critical: Time standardization to UTC
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
    
    # Add symbol column
    df['Symbol'] = symbol
    
    # Set time as index
    df.set_index('time', inplace=True)
    
    # Save individual symbol data
    df.to_csv(f'{DATA_DIR}/raw_data_{symbol}.csv')
    
    # Add to combined dataset
    all_data.append(df)
    
    print(f"{symbol}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

# Combine all symbols into one dataset
if all_data:
    combined_df = pd.concat(all_data, axis=0)
    combined_df.sort_index(inplace=True)
    combined_df.to_csv(f'{DATA_DIR}/raw_data_combined.csv')
    
    print(f"\n=== Data Acquisition Complete ===")
    print(f"Total bars: {len(combined_df)}")
    print(f"Symbols processed: {len(all_data)}")
    print(f"Combined data saved to: {DATA_DIR}/raw_data_combined.csv")
else:
    print("No data acquired for any symbol")

mt5.shutdown()