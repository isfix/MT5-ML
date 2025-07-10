#--- Configuration File for ML Entry Validator ---
from datetime import datetime
import MetaTrader5 as mt5

# Pairs Configuration
SYMBOLS = [
    "XAUUSD",    # Gold
    "EURUSD",    # Euro/USD
    "GBPUSD",    # Pound/USD
    "USDJPY",    # USD/Yen
    "AUDUSD",    # Aussie/USD
    "USDCAD",    # USD/Canadian
    "NZDUSD",    # Kiwi/USD
    "USDCHF",    # USD/Swiss
]

# Data Parameters
TIMEFRAME = mt5.TIMEFRAME_M5
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Model Parameters
FEATURE_COUNT = 14
RR_RATIO = 2.0
SL_ATR_MULTI = 1.5
LOOKAHEAD_BARS = 60

# Technical Indicators
ATR_PERIOD = 14
RSI_PERIOD = 14
SUPPORT_RESISTANCE_PERIOD = 100

# Training Parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
GA_POPULATION = 20
GA_GENERATIONS = 10

# File Paths
DATA_DIR = "data"
MODELS_DIR = "models"