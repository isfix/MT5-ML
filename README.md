# ML Entry Validator EA

A machine learning-powered Expert Advisor for MetaTrader 5 that validates trade entries using ONNX models with proper time-based data splitting.

## Project Overview

This project implements a **data-driven trading system** that:
- Uses ML models to validate trade entries (not predict direction)
- Employs trigger-based efficiency (ML only runs on high-potential setups)
- Features comprehensive risk management and position tracking
- Uses proper time-based data splitting to prevent overfitting
- Includes fallback logic for reliability

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │───▶│   ML Pipeline   │───▶│  Trading Layer  │
│                 │    │                 │    │                 │
│ • XAUUSD only   │    │ • Feature Eng.  │    │ • MQL5 EA       │
│ • M5 timeframe  │    │ • Labeling      │    │ • ONNX Runtime  │
│ • UTC standard  │    │ • Training      │    │ • Risk Mgmt     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Project Structure

```
MT5ML2/
├── 1_data_acquisition.py        # XAUUSD data extraction
├── 2_feature_engineering.py     # 14-feature creation
├── 3_labeling.py                # Trade simulation + time splits
├── 4_model_training.py          # Baseline model training
├── 5_genetic_optimization.py    # GA hyperparameter tuning
├── 6_onnx_export.py            # Model export to ONNX
├── 7_run_complete_pipeline.py  # Master execution script
├── ML_Entry_Validator.mq5      # MetaTrader 5 Expert Advisor
├── feature_schema.md           # Feature documentation
├── data/                       # Generated datasets
│   ├── raw_dataset.csv         # Raw XAUUSD data
│   ├── featured_dataset.csv    # Engineered features
│   ├── labeled_dataset.csv     # Labeled training data
│   ├── train_2020_2022.csv     # Training set (60%)
│   ├── val_2023.csv           # Validation set (20%)
│   └── test_2024.csv          # Test set (20%)
└── models/                     # Trained models
    ├── entry_model.onnx       # Deployable ONNX model
    ├── model_info.json        # Model metadata
    ├── optimized_params.json  # Best hyperparameters
    └── baseline_model.pkl     # Trained baseline model
```

## Quick Start

### Prerequisites

```bash
pip install MetaTrader5 pandas pandas-ta xgboost lightgbm scikit-learn
pip install skl2onnx onnx onnxmltools deap
```

### 1. Data Pipeline

Run the 3-step data pipeline:

```bash
python 1_data_acquisition.py      # Extract XAUUSD data from MT5
python 2_feature_engineering.py   # Create 14 features
python 3_labeling.py              # Simulate trades + time splits
```

### 2. Model Training

Train and optimize the model:

```bash
python 4_model_training.py        # Train baseline models
python 5_genetic_optimization.py  # Optimize hyperparameters
python 6_onnx_export.py           # Export to ONNX format
```

Or run complete pipeline:

```bash
python 7_run_complete_pipeline.py
```

### 3. MetaTrader 5 Deployment

1. Copy `entry_model.onnx` to `MQL5/Files/`
2. Copy `ML_Entry_Validator.mq5` to `MQL5/Experts/`
3. Compile and attach to XAUUSD M5 chart
4. Configure EA parameters

## Time-Based Data Splitting

### Proper ML Workflow
- **Training Set (2020-2022)**: Model learns patterns from historical data
- **Validation Set (2023)**: Genetic Algorithm optimizes hyperparameters
- **Test Set (2024+)**: Final evaluation on unseen recent data

### Why Time-Based?
- ✅ **No Data Leakage**: Model can't learn from future data
- ✅ **Realistic Performance**: Tests on chronologically future data
- ✅ **Prevents Overfitting**: Each phase uses distinct time periods
- ✅ **Mirrors Live Trading**: Models trained on past, tested on present

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

1. **Baseline**: XGBoost vs LightGBM comparison on training data
2. **Optimization**: Genetic Algorithm tunes hyperparameters using validation data
3. **Export**: ONNX format for MQL5 compatibility
4. **Evaluation**: Final test on unseen 2024+ data

## Expert Advisor Features

### Core Functionality

- **Trigger-Based**: Only activates when price approaches support levels
- **ML Validation**: Confidence threshold filtering (default: 0.50)
- **Risk Management**: Position sizing based on account risk percentage
- **Position Tracking**: Multi-position monitoring with metadata
- **Trailing Stops**: Dynamic stop loss adjustment
- **Fallback Logic**: Rule-based system if ONNX fails

### Key Parameters

```cpp
input double   InpConfidenceThreshold = 0.50;  // ML confidence threshold
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

## Configuration

### Symbol Configuration

The system is configured for **XAUUSD only** in `1_data_acquisition.py`:

```python
SYMBOL = "XAUUSD"           # Gold trading pair
TIMEFRAME = mt5.TIMEFRAME_M5 # 5-minute timeframe
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2024, 12, 31)
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
```

## Backtesting

### MetaTrader 5 Strategy Tester

1. **EA**: ML_Entry_Validator
2. **Symbol**: XAUUSD
3. **Timeframe**: M5
4. **Model**: "Every tick based on real ticks"
5. **Period**: Use 2024+ data (test set period)

### Expected Behavior

- **Triggers**: "TRIGGER: Conditions met" messages
- **Feature Debug**: All 14 feature values logged
- **ML Predictions**: Confidence scores logged
- **Trade Decisions**: Execute/Reject based on threshold
- **Risk Management**: Position sizing and trailing stops

## Performance Analysis

### Model Validation

Test your model predictions match between Python and MQL5:

```bash
python test_mql5_features.py
```

This verifies:
- ✅ **Feature Consistency**: Same values in Python and MQL5
- ✅ **Prediction Accuracy**: ONNX model produces identical results
- ✅ **Pipeline Integrity**: End-to-end validation

### Confidence Threshold Tuning

- **0.65+**: Very conservative, few trades
- **0.50-0.65**: Balanced approach
- **0.40-0.50**: More aggressive trading
- **<0.40**: High frequency, lower quality

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| No trades in backtest | Lower confidence threshold (0.45-0.50) |
| ONNX model fails to load | Check file path and enable fallback logic |
| Poor performance | Retrain with more recent data |
| Feature mismatch | Run test_mql5_features.py to verify |

### Debug Mode

Enable detailed logging in MQL5:
```cpp
// Add this to ProcessTradingSignals() for feature debugging
string feature_log = "Features: ";
for(int i = 0; i < FEATURE_COUNT; i++) {
    feature_log += "[" + IntegerToString(i) + "]=" + DoubleToString(m_features[i], 4) + "; ";
}
Print(feature_log);
```

## Retraining Workflow

### When to Retrain

- **Quarterly**: Regular model updates
- **Performance Degradation**: When live results decline
- **Market Regime Changes**: After major market shifts

### Retraining Process

1. Update date ranges in data acquisition
2. Run complete pipeline: `python 7_run_complete_pipeline.py`
3. Validate new model against old model on test data
4. Deploy only if statistically significant improvement

## Key Improvements

### vs Original Version

- ✅ **Time-Based Splitting**: Prevents data leakage and overfitting
- ✅ **Single Symbol Focus**: Simplified to XAUUSD only
- ✅ **No Config Dependency**: Self-contained scripts
- ✅ **Proper Validation**: Separate validation set for hyperparameter tuning
- ✅ **ONNX Compatibility**: Fixed data type issues for MQL5
- ✅ **Feature Debugging**: Real-time feature value logging

## Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk and never risk more than you can afford to lose.

## Acknowledgments

- MetaTrader 5 for the trading platform
- ONNX community for model portability standards
- scikit-learn and XGBoost teams for ML frameworks
- MQL5 community for trading insights