#--- Simulation-Based Labeling Module ---
import pandas as pd
import numpy as np

# Load featured data
df = pd.read_csv('data/featured_data.csv', index_col='time', parse_dates=True)

print(f"Loaded {len(df)} bars for labeling")

# Import parameters from config
from config import RR_RATIO, SL_ATR_MULTI, LOOKAHEAD_BARS

def simulate_trade(df, idx):
    """
    Simulate a long trade from the given index
    Returns: 1 (Win), 0 (Loss), -1 (Timeout)
    """
    if idx >= len(df) - LOOKAHEAD_BARS:
        return -1  # Not enough future data
    
    entry_price = df.iloc[idx]['Close']
    atr_value = df.iloc[idx]['ATR_14']
    
    # Calculate exit levels
    stop_loss = entry_price - (atr_value * SL_ATR_MULTI)
    take_profit = entry_price + (atr_value * SL_ATR_MULTI * RR_RATIO)
    
    # Look ahead for the next LOOKAHEAD_BARS
    future_data = df.iloc[idx+1:idx+1+LOOKAHEAD_BARS]
    
    tp_hit = False
    sl_hit = False
    tp_bar = None
    sl_bar = None
    
    for i, (_, row) in enumerate(future_data.iterrows()):
        # Check if TP hit first
        if not tp_hit and row['High'] >= take_profit:
            tp_hit = True
            tp_bar = i
        
        # Check if SL hit first  
        if not sl_hit and row['Low'] <= stop_loss:
            sl_hit = True
            sl_bar = i
        
        # If both hit, determine which came first
        if tp_hit and sl_hit:
            if tp_bar <= sl_bar:
                return 1  # TP hit first - Win
            else:
                return 0  # SL hit first - Loss
    
    # Only one hit
    if tp_hit:
        return 1  # Win
    elif sl_hit:
        return 0  # Loss
    else:
        return -1  # Timeout - neither hit
    
# Apply labeling function
print("Starting trade simulation labeling...")
labels = []

for i in range(len(df)):
    if i % 10000 == 0:
        print(f"Processing bar {i}/{len(df)}")
    
    label = simulate_trade(df, i)
    labels.append(label)

df['label'] = labels

# Filter out timeout cases (label == -1)
initial_count = len(df)
df = df[df['label'] != -1]
final_count = len(df)

print(f"Labeling complete:")
print(f"Initial bars: {initial_count}")
print(f"Final bars (after removing timeouts): {final_count}")
print(f"Removed {initial_count - final_count} timeout cases")

# Check class balance
print(f"\nClass distribution:")
print(df['label'].value_counts())
print(f"Win rate: {df['label'].mean():.3f}")

# Save final dataset
df.to_csv('data/final_dataset.csv')

print(f"\nFinal golden dataset saved with {len(df)} labeled examples")