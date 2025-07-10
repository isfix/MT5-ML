# Unified Feature Schema

This document is the single source of truth for all features used in the model.
The order of features listed here **must** be maintained in the MQL5 implementation.

| Feature Name           | Data Type | Description                                                                 |
|------------------------|-----------|-----------------------------------------------------------------------------|
| Open                   | float     | Opening price of the candle                                                 |
| High                   | float     | Highest price of the candle                                                 |
| Low                    | float     | Lowest price of the candle                                                  |
| Close                  | float     | Closing price of the candle                                                 |
| Volume                 | float     | Tick volume for the period                                                  |
| RSI_14                 | float     | 14-period Relative Strength Index                                           |
| ATR_14                 | float     | 14-period Average True Range                                                |
| support_100            | float     | 100-period rolling minimum (dynamic support)                                |
| resistance_100         | float     | 100-period rolling maximum (dynamic resistance)                             |
| dist_to_support_atr    | float     | Distance to support normalized by ATR: (Close - support_100) / ATR_14       |
| dist_to_resistance_atr | float     | Distance to resistance normalized by ATR: (resistance_100 - Close) / ATR_14 |
| is_london_session      | float     | Binary flag: 1 if hour is 7-15 UTC (London session), 0 otherwise            |
| is_ny_session          | float     | Binary flag: 1 if hour is 13-21 UTC (New York session), 0 otherwise         |
| RSI_14_H1              | float     | 14-period RSI calculated on H1 timeframe, forward-filled to M5              |

**Total Features: 14**

## Feature Engineering Notes

- All distance features are normalized by ATR to ensure robustness across volatility regimes
- Session flags based on UTC time to maintain consistency  
- Multi-timeframe features use forward-fill to propagate H1 values to M5 bars
- Support/resistance levels are dynamic, recalculated every bar
- Feature order is critical and must match exactly between Python and MQL5 implementations