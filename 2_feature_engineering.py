#--- Feature Engineering Module ---
import pandas as pd
import pandas_ta as ta

# Load raw data
df = pd.read_csv('data/raw_dataset.csv', index_col='time', parse_dates=True)

print(f"Loaded {len(df)} bars for feature engineering")

# Basic Indicators
df['RSI_14'] = ta.rsi(df['Close'], length=14)
df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

# Dynamic Support/Resistance
df['support_100'] = df['Low'].rolling(window=100).min()
df['resistance_100'] = df['High'].rolling(window=100).max()

# Distance features normalized by ATR
df['dist_to_support_atr'] = (df['Close'] - df['support_100']) / df['ATR_14']
df['dist_to_resistance_atr'] = (df['resistance_100'] - df['Close']) / df['ATR_14']

# Market Sessions
df['hour'] = df.index.hour
df['is_london_session'] = ((df['hour'] >= 7) & (df['hour'] <= 15)).astype(int)
df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)

# Multi-Timeframe Context
df_h1 = df.resample('1H').agg({
    'Open': 'first',
    'High': 'max', 
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

df_h1['RSI_14_H1'] = ta.rsi(df_h1['Close'], length=14)

# Merge H1 RSI back to M5 data
df = df.merge(df_h1[['RSI_14_H1']], left_index=True, right_index=True, how='left')
df['RSI_14_H1'] = df['RSI_14_H1'].fillna(method='ffill')

# Clean up
df.drop(['hour'], axis=1, inplace=True)
df.dropna(inplace=True)

# Save featured data
df.to_csv('data/featured_dataset.csv')

print(f"Feature engineering complete. {len(df)} bars saved to data/featured_dataset.csv")