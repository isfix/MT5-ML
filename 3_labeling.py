#--- Trade Simulation Labeling ---
import pandas as pd
import numpy as np

# Load featured data
df = pd.read_csv('data/featured_dataset.csv', index_col='time', parse_dates=True)

print(f"Loaded {len(df)} bars for labeling")

# Labeling parameters
RR_RATIO = 2.0
SL_ATR_MULTI = 1.5
LOOKAHEAD_BARS = 60

def simulate_trade(df, idx):
    if idx >= len(df) - LOOKAHEAD_BARS:
        return -1
    
    entry_price = df.iloc[idx]['Close']
    atr_value = df.iloc[idx]['ATR_14']
    
    stop_loss = entry_price - (atr_value * SL_ATR_MULTI)
    take_profit = entry_price + (atr_value * SL_ATR_MULTI * RR_RATIO)
    
    future_data = df.iloc[idx+1:idx+1+LOOKAHEAD_BARS]
    
    for _, row in future_data.iterrows():
        if row['High'] >= take_profit:
            return 1  # Win
        if row['Low'] <= stop_loss:
            return 0  # Loss
    
    return -1  # Timeout

# Apply labeling
print("Starting trade simulation labeling...")
labels = []

for i in range(len(df)):
    if i % 10000 == 0:
        print(f"Processing bar {i}/{len(df)}")
    
    label = simulate_trade(df, i)
    labels.append(label)

df['label'] = labels

# Filter out timeout cases
df = df[df['label'] != -1]

print(f"Labeling complete: {len(df)} labeled examples")
print(f"Win rate: {df['label'].mean():.3f}")

# Save the final labeled dataset
output_path = 'data/final_dataset.csv'
df.to_csv(output_path)
print(f"\nFinal dataset saved to {output_path}")
