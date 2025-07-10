//+------------------------------------------------------------------+
//|                                           ML_Entry_Validator.mq5 |
//|                        Advanced ML Entry Validator Expert Advisor|
//+------------------------------------------------------------------+
#property copyright "ML Entry Validator Project"
#property version   "2.00"

#include <Trade\Trade.mqh>

#define   ModelPath "entry_model.onnx"
#resource ModelPath as const uchar ExtModel[];

//--- Input parameters
input group "=== ML Model Settings ==="
input double   InpConfidenceThreshold = 0.65;     // ML confidence threshold
input bool     InpUseFallbackLogic = true;        // Use fallback if model fails

input group "=== Trading Parameters ==="
input double   InpLotSize = 0.01;                 // Position size
input double   InpSLMultiplier = 1.5;             // Stop Loss ATR multiplier
input double   InpRRRatio = 2.0;                  // Risk-Reward ratio
input int      InpMaxPositions = 1;               // Maximum concurrent positions

input group "=== Technical Indicators ==="
input int      InpATRPeriod = 14;                 // ATR calculation period
input int      InpRSIPeriod = 14;                 // RSI calculation period
input int      InpSRPeriod = 100;                 // Support/Resistance lookback

input group "=== Risk Management ==="
input double   InpMaxRiskPercent = 2.0;           // Maximum risk per trade (%)
input bool     InpUseTrailingStop = true;         // Enable trailing stop
input double   InpTrailingDistance = 50;          // Trailing stop distance (points)

//--- Global variables
CTrade         m_trade;
long           m_onnxModel;
datetime       m_lastBarTime;
int            m_rsiHandle;
int            m_atrHandle;
bool           m_modelLoaded;

//--- Feature array
double         m_features[14];
const int      FEATURE_COUNT = 14;

//--- Position management
struct PositionInfo
{
   ulong    ticket;
   double   entryPrice;
   double   stopLoss;
   double   takeProfit;
   datetime openTime;
   double   confidence;
};

PositionInfo m_positions[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== Advanced ML Entry Validator EA v2.0 ===");
   
   // Initialize trading class
   m_trade.SetExpertMagicNumber(12345);
   m_trade.SetDeviationInPoints(10);
   m_trade.SetTypeFilling(ORDER_FILLING_FOK);
   
   // Initialize technical indicators
   m_rsiHandle = iRSI(_Symbol, PERIOD_CURRENT, InpRSIPeriod, PRICE_CLOSE);
   m_atrHandle = iATR(_Symbol, PERIOD_CURRENT, InpATRPeriod);
   
   if(m_rsiHandle == INVALID_HANDLE || m_atrHandle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicator handles");
      return INIT_FAILED;
   }
   
   // Initialize ONNX model
   m_onnxModel = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   m_modelLoaded = (m_onnxModel != INVALID_HANDLE);
   
   if(!m_modelLoaded)
   {
      if(InpUseFallbackLogic)
      {
         Print("WARNING: ONNX model failed to load, using fallback logic");
      }
      else
      {
         Print("ERROR: ONNX model required but failed to load");
         return INIT_FAILED;
      }
   }
   else
   {
      Print("SUCCESS: ONNX model loaded successfully");
   }
   
   // Initialize position tracking
   ArrayResize(m_positions, InpMaxPositions);
   for(int i = 0; i < ArraySize(m_positions); i++)
   {
      m_positions[i].ticket = 0;
   }
   
   Print("SUCCESS: EA initialized successfully");
   Print("- Model loaded: ", m_modelLoaded ? "YES" : "NO (Fallback)");
   Print("- Confidence threshold: ", InpConfidenceThreshold);
   Print("- Risk per trade: ", InpMaxRiskPercent, "%");
   Print("- Max positions: ", InpMaxPositions);
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Release ONNX model
   if(m_onnxModel != INVALID_HANDLE)
      OnnxRelease(m_onnxModel);
   
   // Release indicator handles
   if(m_rsiHandle != INVALID_HANDLE) IndicatorRelease(m_rsiHandle);
   if(m_atrHandle != INVALID_HANDLE) IndicatorRelease(m_atrHandle);
   
   Print("ML Entry Validator EA deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new bar
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBarTime == m_lastBarTime) return;
   m_lastBarTime = currentBarTime;
   
   // Update position tracking
   UpdatePositionTracking();
   
   // Apply trailing stops
   if(InpUseTrailingStop) ApplyTrailingStops();
   
   // Check for new trade opportunities
   if(GetActivePositionsCount() < InpMaxPositions)
   {
      ProcessTradingSignals();
   }
}

//+------------------------------------------------------------------+
//| Process trading signals                                          |
//+------------------------------------------------------------------+
void ProcessTradingSignals()
{
   // Calculate all features
   if(!CalculateFeatures()) return;
   
   // Check trigger conditions
   if(!CheckTriggerConditions()) return;
   
   Print("TRIGGER: Market conditions met at ", TimeToString(TimeCurrent()));
   
   // Get ML prediction
   double confidence = GetMLPrediction();
   
   Print("ML Prediction - Confidence: ", DoubleToString(confidence, 4));
   
   // Execute trade if confidence is high enough
   if(confidence >= InpConfidenceThreshold)
   {
      ExecuteTrade(confidence);
   }
   else
   {
      Print("REJECT: Low confidence (", DoubleToString(confidence, 4), 
            " < ", DoubleToString(InpConfidenceThreshold, 2), ")");
   }
}

//+------------------------------------------------------------------+
//| Calculate all features for ML model                              |
//+------------------------------------------------------------------+
bool CalculateFeatures()
{
   // Get basic OHLCV data (last completed bar)
   m_features[0] = iOpen(_Symbol, PERIOD_CURRENT, 1);   // Open
   m_features[1] = iHigh(_Symbol, PERIOD_CURRENT, 1);   // High
   m_features[2] = iLow(_Symbol, PERIOD_CURRENT, 1);    // Low
   m_features[3] = iClose(_Symbol, PERIOD_CURRENT, 1);  // Close
   m_features[4] = (double)iVolume(_Symbol, PERIOD_CURRENT, 1); // Volume
   
   // Get RSI
   double rsiBuffer[1];
   if(CopyBuffer(m_rsiHandle, 0, 1, 1, rsiBuffer) <= 0) return false;
   m_features[5] = rsiBuffer[0]; // RSI_14
   
   // Get ATR
   double atrBuffer[1];
   if(CopyBuffer(m_atrHandle, 0, 1, 1, atrBuffer) <= 0) return false;
   m_features[6] = atrBuffer[0]; // ATR_14
   
   // Calculate dynamic support and resistance
   double support = m_features[3];
   double resistance = m_features[3];
   
   for(int i = 1; i <= InpSRPeriod; i++)
   {
      double high = iHigh(_Symbol, PERIOD_CURRENT, i);
      double low = iLow(_Symbol, PERIOD_CURRENT, i);
      
      if(low < support) support = low;
      if(high > resistance) resistance = high;
   }
   
   m_features[7] = support;    // support_100
   m_features[8] = resistance; // resistance_100
   
   // Distance features normalized by ATR
   m_features[9] = (m_features[3] - support) / m_features[6];     // dist_to_support_atr
   m_features[10] = (resistance - m_features[3]) / m_features[6]; // dist_to_resistance_atr
   
   // Market session flags
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int hour = dt.hour;
   
   m_features[11] = (hour >= 7 && hour <= 15) ? 1.0 : 0.0;  // is_london_session
   m_features[12] = (hour >= 13 && hour <= 21) ? 1.0 : 0.0; // is_ny_session
   
   // H1 RSI (approximated)
   m_features[13] = m_features[5]; // RSI_14_H1
   
   return true;
}

//+------------------------------------------------------------------+
//| Check trigger conditions                                         |
//+------------------------------------------------------------------+
bool CheckTriggerConditions()
{
   double close = m_features[3];
   double support = m_features[7];
   double atr = m_features[6];
   
   // Trigger: Price near support level
   double distanceToSupport = (close - support) / atr;
   
   // Additional filters
   bool nearSupport = (distanceToSupport >= 0.0 && distanceToSupport <= 0.5);
   bool rsiOversold = (m_features[5] < 40); // RSI oversold
   bool volatilityOK = (atr > 0.0001); // Minimum volatility
   
   return nearSupport && rsiOversold && volatilityOK;
}

//+------------------------------------------------------------------+
//| Run ONNX model prediction                                        |
//+------------------------------------------------------------------+
bool RunONNXModel(double &features[], double &predictions[])
{
   if(m_onnxModel == INVALID_HANDLE) return false;
   
   // Set input shape [1, 14] - batch size 1, 14 features
   ulong input_shape[] = {1, FEATURE_COUNT};
   if(!OnnxSetInputShape(m_onnxModel, 0, input_shape))
   {
      Print("ERROR: Failed to set input shape");
      return false;
   }
   
   // Set output shape [1, 1] - batch size 1, 1 prediction
   ulong output_shape[] = {1, 1};
   if(!OnnxSetOutputShape(m_onnxModel, 0, output_shape))
   {
      Print("ERROR: Failed to set output shape");
      return false;
   }
   
   // Run inference
   return OnnxRun(m_onnxModel, ONNX_NO_CONVERSION, features, predictions);
}

//+------------------------------------------------------------------+
//| Get ML prediction                                                |
//+------------------------------------------------------------------+
double GetMLPrediction()
{
   if(m_modelLoaded)
   {
      // Use ONNX model
      double predictions[1];
      if(RunONNXModel(m_features, predictions))
      {
         return predictions[0];
      }
      else
      {
         Print("WARNING: ONNX prediction failed, using fallback");
         if(!InpUseFallbackLogic) return 0.0;
      }
   }
   
   // Fallback: Advanced rule-based prediction
   double score = 0.5; // Base score
   
   // RSI factor (stronger weight)
   if(m_features[5] < 20) score += 0.25;      // Extremely oversold
   else if(m_features[5] < 30) score += 0.20; // Very oversold
   else if(m_features[5] < 40) score += 0.10; // Oversold
   
   // Support distance factor
   double supportDist = m_features[9];
   if(supportDist < 0.1) score += 0.20;      // Very close to support
   else if(supportDist < 0.3) score += 0.15; // Close to support
   else if(supportDist < 0.5) score += 0.10; // Near support
   
   // Resistance distance factor (confluence)
   double resistanceDist = m_features[10];
   if(resistanceDist > 2.0) score += 0.10; // Far from resistance
   
   // Session factor
   if(m_features[11] == 1.0 || m_features[12] == 1.0) score += 0.05; // Active session
   
   // Volatility factor
   if(m_features[6] > 0.001) score += 0.05; // Good volatility
   
   return MathMin(score, 0.95); // Cap at 95%
}

//+------------------------------------------------------------------+
//| Execute trade                                                    |
//+------------------------------------------------------------------+
void ExecuteTrade(double confidence)
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double atr = m_features[6];
   
   // Calculate position size based on risk
   double lotSize = CalculatePositionSize(atr);
   
   // Calculate SL and TP
   double stopLoss = ask - (atr * InpSLMultiplier);
   double takeProfit = ask + (atr * InpSLMultiplier * InpRRRatio);
   
   // Normalize prices
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   stopLoss = NormalizeDouble(stopLoss, digits);
   takeProfit = NormalizeDouble(takeProfit, digits);
   
   Print("=== TRADE EXECUTION ===");
   Print("Confidence: ", DoubleToString(confidence, 4));
   Print("Entry: ", DoubleToString(ask, digits));
   Print("SL: ", DoubleToString(stopLoss, digits));
   Print("TP: ", DoubleToString(takeProfit, digits));
   Print("Lot Size: ", DoubleToString(lotSize, 2));
   Print("Risk Amount: $", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE) * InpMaxRiskPercent / 100.0, 2));
   
   // Execute order
   string comment = "ML_Entry_" + DoubleToString(confidence, 2);
   if(m_trade.Buy(lotSize, _Symbol, ask, stopLoss, takeProfit, comment))
   {
      ulong ticket = m_trade.ResultOrder();
      Print("SUCCESS: Buy order placed. Ticket: ", ticket);
      
      // Store position info
      StorePositionInfo(ticket, ask, stopLoss, takeProfit, confidence);
   }
   else
   {
      Print("ERROR: Failed to place order. Error: ", GetLastError());
      Print("Trade result: ", m_trade.ResultRetcode(), " - ", m_trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Calculate position size based on risk                           |
//+------------------------------------------------------------------+
double CalculatePositionSize(double atr)
{
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = accountBalance * InpMaxRiskPercent / 100.0;
   
   double stopLossDistance = atr * InpSLMultiplier;
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   
   double lotSize = riskAmount / (stopLossDistance / tickSize * tickValue);
   
   // Apply lot size limits
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lotSize = MathMax(lotSize, minLot);
   lotSize = MathMin(lotSize, maxLot);
   lotSize = NormalizeDouble(lotSize / lotStep, 0) * lotStep;
   
   return lotSize;
}

//+------------------------------------------------------------------+
//| Store position information                                       |
//+------------------------------------------------------------------+
void StorePositionInfo(ulong ticket, double entry, double sl, double tp, double confidence)
{
   for(int i = 0; i < ArraySize(m_positions); i++)
   {
      if(m_positions[i].ticket == 0)
      {
         m_positions[i].ticket = ticket;
         m_positions[i].entryPrice = entry;
         m_positions[i].stopLoss = sl;
         m_positions[i].takeProfit = tp;
         m_positions[i].openTime = TimeCurrent();
         m_positions[i].confidence = confidence;
         break;
      }
   }
}

//+------------------------------------------------------------------+
//| Update position tracking                                         |
//+------------------------------------------------------------------+
void UpdatePositionTracking()
{
   for(int i = 0; i < ArraySize(m_positions); i++)
   {
      if(m_positions[i].ticket > 0)
      {
         if(!PositionSelectByTicket(m_positions[i].ticket))
         {
            // Position closed
            Print("Position closed: Ticket ", m_positions[i].ticket, 
                  ", Confidence: ", DoubleToString(m_positions[i].confidence, 4));
            m_positions[i].ticket = 0;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Apply trailing stops                                             |
//+------------------------------------------------------------------+
void ApplyTrailingStops()
{
   for(int i = 0; i < ArraySize(m_positions); i++)
   {
      if(m_positions[i].ticket > 0)
      {
         if(PositionSelectByTicket(m_positions[i].ticket))
         {
            double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double currentSL = PositionGetDouble(POSITION_SL);
            double newSL = currentPrice - InpTrailingDistance * _Point;
            
            if(newSL > currentSL + _Point)
            {
               if(m_trade.PositionModify(m_positions[i].ticket, newSL, PositionGetDouble(POSITION_TP)))
               {
                  Print("Trailing stop updated for ticket: ", m_positions[i].ticket, 
                        " New SL: ", DoubleToString(newSL, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)));
                  m_positions[i].stopLoss = newSL;
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Get active positions count                                       |
//+------------------------------------------------------------------+
int GetActivePositionsCount()
{
   int count = 0;
   for(int i = 0; i < ArraySize(m_positions); i++)
   {
      if(m_positions[i].ticket > 0) count++;
   }
   return count;
}