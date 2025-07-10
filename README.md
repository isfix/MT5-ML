# Hybrid ML Entry Validator EA

A sophisticated machine learning-powered Expert Advisor for MetaTrader 5 that combines rule-based triggers with ML validation for high-probability trade entries.

## Project Overview

This project implements a **data-driven trading system** that:
- Uses ML models to validate trade entries (not predict direction)
- Employs trigger-based efficiency (ML only runs on high-potential setups)
- Features comprehensive risk management and position tracking
- Supports multi-symbol training for robust generalization
- Includes fallback logic for reliability

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │───▶│   ML Pipeline   │───▶│  Trading Layer  │
│                 │    │                 │    │                 │
│ • Multi-symbol  │    │ • Feature Eng.  │    │ • MQL5 EA       │
│ • M5 timeframe  │    │ • Labeling      │    │ • ONNX Runtime  │
│ • UTC standard  │    │ • Training      │    │ • Risk Mgmt     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Project Structure

```
MT5ML2/
├── config.py                    # Central configuration
├── 1_data_acquisition.py        # Multi-symbol data extraction
├── 2_feature_engineering.py     # 14-feature creation
├── 3_labeling.py                # Trade simulation labeling
├── 4_model_training.py          # Baseline model training
├── 5_genetic_optimization.py    # GA hyperparameter tuning
├── 6_onnx_export.py            # Model export to ONNX
├── 7_run_complete_pipeline.py  # Master execution script
├── ML_Entry_Validator.mq5      # MetaTrader 5 Expert Advisor
├── feature_schema.md           # Feature documentation
├── data/                       # Generated datasets
│   ├── raw_data_*.csv         # Individual symbol data
│   ├── raw_data_combined.csv  # Multi-symbol dataset
│   ├── featured_data.csv      # Engineered features
│   └── final_dataset.csv      # Labeled training data
└── models/                     # Trained models
    ├── entry_model.onnx       # Deployable ONNX model
    ├── model_info.json        # Model metadata
    ├── optimized_params.json  # Best hyperparameters
    └── final_model_metadata.json
```

## Quick Start

### Prerequisites

```bash
pip install MetaTrader5 pandas pandas-ta xgboost lightgbm scikit-learn
pip install skl2onnx onnx onnxmltools deap
```

### 1. Configuration

Edit `config.py` to select your trading pairs:

```python
SYMBOLS = [
    "XAUUSD",    # Gold
    "EURUSD",    # Euro/USD
    "GBPUSD",    # Add/remove as needed
]
```

### 2. Training Pipeline

Run the complete pipeline:

```bash
python 7_run_complete_pipeline.py
```

Or execute steps individually:

```bash
python 1_data_acquisition.py      # Extract data from MT5
python 2_feature_engineering.py   # Create 14 features
python 3_labeling.py              # Simulate trades for labels
python 4_model_training.py        # Train baseline models
python 5_genetic_optimization.py  # Optimize hyperparameters
python 6_onnx_export.py           # Export to ONNX format
```

### 3. MetaTrader 5 Deployment

1. Copy `entry_model.onnx` to `MQL5/Files/`
2. Copy `ML_Entry_Validator.mq5` to `MQL5/Experts/`
3. Compile and attach to chart
4. Configure EA parameters

## Machine Learning Pipeline

### Feature Engineering (14 Features)

| Category | Features | Description |
|----------|----------|-------------|
| **OHLCV** | Open, High, Low, Close, Volume | Basic price data |
| **Technical** | RSI_14, ATR_14 | Momentum and volatility |
| **Structural** | support_100, resistance_100 | Dynamic S/R levels |
| **Normalized** | dist_to_support_atr, dist_to_resistance_atr | ATR-normalized distances |
| **Temporal** | is_london_session, is_ny_session | Market session flags |
| **Multi-TF** | RSI_14_H1 | Higher timeframe context |

### Labeling Strategy

- **Entry**: Current bar close price
- **Stop Loss**: Entry - (ATR × 1.5)
- **Take Profit**: Entry + (ATR × 1.5 × 2.0) [2:1 RR]
- **Lookahead**: 60 bars (5 hours on M5)
- **Label**: 1 (Win) if TP hit first, 0 (Loss) if SL hit first

### Model Training

1. **Baseline**: XGBoost vs LightGBM comparison
2. **Optimization**: Genetic Algorithm hyperparameter tuning
3. **Export**: ONNX format for MQL5 compatibility
4. **Validation**: Stratified train/test split with precision focus

## Expert Advisor Features

### Core Functionality

- **Trigger-Based**: Only activates when price approaches support levels
- **ML Validation**: Confidence threshold filtering (default: 0.65)
- **Risk Management**: Position sizing based on account risk percentage
- **Position Tracking**: Multi-position monitoring with metadata
- **Trailing Stops**: Dynamic stop loss adjustment
- **Fallback Logic**: Rule-based system if ONNX fails

### Key Parameters

```cpp
input double   InpConfidenceThreshold = 0.65;  // ML confidence threshold
input double   InpMaxRiskPercent = 2.0;        // Risk per trade (%)
input int      InpMaxPositions = 1;            // Concurrent positions
input bool     InpUseTrailingStop = true;      // Enable trailing stops
```

### Trading Logic

1. **New Bar Check**: Process only on completed M5 bars
2. **Trigger Evaluation**: Price near support + RSI oversold + volatility OK
3. **Feature Calculation**: Real-time computation of 14 features
4. **ML Prediction**: ONNX model inference with confidence score
5. **Trade Execution**: If confidence ≥ threshold, execute with calculated SL/TP
6. **Position Management**: Track, trail, and log all positions

## Configuration Options

### Trading Pairs (`config.py`)

```python
SYMBOLS = [
    "XAUUSD",    # Gold - High volatility
    "EURUSD",    # Major forex pair
    "GBPUSD",    # Volatile forex pair
    # Add any MT5 symbol
]
```

### Model Parameters

```python
RR_RATIO = 2.0              # Risk-reward ratio
SL_ATR_MULTI = 1.5          # Stop loss ATR multiplier
LOOKAHEAD_BARS = 60         # Label lookahead period
```

### Training Parameters

```python
GA_POPULATION = 20          # Genetic algorithm population
GA_GENERATIONS = 10         # GA generations
TEST_SIZE = 0.2            # Train/test split ratio
```

## Backtesting

### MetaTrader 5 Strategy Tester

1. **EA**: ML_Entry_Validator
2. **Symbol**: Match your training data (e.g., XAUUSD)
3. **Timeframe**: M5
4. **Model**: "Every tick based on real ticks"
5. **Period**: Use subset of training date range

### Expected Behavior

- **Triggers**: "TRIGGER: Conditions met" messages
- **ML Predictions**: Confidence scores logged
- **Trade Decisions**: Execute/Reject based on threshold
- **Risk Management**: Position sizing and trailing stops

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| No trades in backtest | Lower confidence threshold (0.55) |
| ONNX model fails to load | Check file path and enable fallback logic |
| Poor performance | Retrain with more recent data |
| Compilation errors | Ensure ONNX model is in MQL5/Files/ |

### Debug Mode

Enable detailed logging by setting:
```cpp
input bool InpUseFallbackLogic = true;  // Shows fallback predictions
```

## Performance Metrics

The system tracks:
- **Precision**: Percentage of predicted wins that were actual wins
- **Win Rate**: Overall percentage of winning trades
- **Risk-Reward**: Actual vs target RR ratios
- **Confidence Distribution**: ML prediction score analysis

## Retraining Workflow

### When to Retrain

- **Quarterly**: Regular model updates
- **Performance Degradation**: When live results decline
- **Market Regime Changes**: After major market shifts
- **New Symbols**: When adding trading pairs

### Retraining Process

1. Update date ranges in `config.py`
2. Run complete pipeline: `python 7_run_complete_pipeline.py`
3. Validate new model against old model on hold-out data
4. Deploy only if statistically significant improvement

## Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk and never risk more than you can afford to lose.

## Acknowledgments

- MetaTrader 5 for the trading platform
- ONNX community for model portability standards
- scikit-learn and XGBoost teams for ML frameworks
- MQL5 community for trading insights