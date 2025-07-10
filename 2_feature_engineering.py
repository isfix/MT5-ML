#--- Feature Engineering Module ---
import pandas as pd
import pandas_ta as ta
import numpy as np

# Load combined raw data
df = pd.read_csv('data/raw_data_combined.csv', index_col='time', parse_dates=True)

print(f"Loaded {len(df)} bars for feature engineering")

# Layer 1: Basic Indicators
df['RSI_14'] = ta.rsi(df['Close'], length=14)
df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

# Layer 2: Structural Context (Dynamic Support/Resistance)
df['support_100'] = df['Low'].rolling(window=100).min()
df['resistance_100'] = df['High'].rolling(window=100).max()

# Distance features normalized by ATR
df['dist_to_support_atr'] = (df['Close'] - df['support_100']) / df['ATR_14']
df['dist_to_resistance_atr'] = (df['resistance_100'] - df['Close']) / df['ATR_14']

# Layer 3: Temporal Context (Market Sessions)
df['hour'] = df.index.hour
df['is_london_session'] = ((df['hour'] >= 7) & (df['hour'] <= 15)).astype(int)
df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)

# Layer 4: Multi-Timeframe Context
# Resample to H1 and calculate RSI
df_h1 = df.resample('1H').agg({
    'Open': 'first',
    'High': 'max', 
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Calculate RSI on H1 data
df_h1['RSI_14_H1'] = ta.rsi(df_h1['Close'], length=14)

# Merge H1 RSI back to M5 data with forward fill
df = df.merge(df_h1[['RSI_14_H1']], left_index=True, right_index=True, how='left')
df['RSI_14_H1'] = df['RSI_14_H1'].fillna(method='ffill')

# Clean up temporary columns
df.drop(['hour'], axis=1, inplace=True)

# Remove rows with NaN values (initial periods where indicators can't be calculated)
df.dropna(inplace=True)

# Save featured data
df.to_csv('data/featured_data.csv')

print(f"Feature engineering complete. {len(df)} bars with {len(df.columns)} features saved")
print("Features created:", list(df.columns))

# Verify multi-timeframe feature
print(f"\nVerification - RSI_14_H1 sample (should show same value for 12 consecutive M5 bars):")
print(df['RSI_14_H1'].head(15))